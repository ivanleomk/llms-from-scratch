"""Parallel BPE implementation with chunked pretokenization."""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import BinaryIO

import regex

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
WordCounts = dict[tuple[bytes, ...], int]
PairCounts = dict[tuple[bytes, bytes], int]

GPT2_REGEX = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 4,
) -> tuple[Vocab, Merges]:
    """Train a BPE tokenizer and return vocabulary and merges."""
    word_counts = pretokenize_parallel(input_path, special_tokens, num_workers)
    vocab = build_initial_vocab(special_tokens)
    merges: Merges = []

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        best_pair = find_most_frequent_pair(word_counts, vocab)
        if best_pair is None:
            break
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_counts = apply_merge(word_counts, best_pair)

    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> WordCounts:
    """Process a single chunk of the file."""
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    corpus_split = [chunk]

    if special_tokens:
        pattern = (
            r"(" + "|".join(regex.escape(token) for token in special_tokens) + r")"
        )
        corpus_split = regex.split(pattern, chunk)
        corpus_split = [x for x in corpus_split if x]

    word_count: WordCounts = {}
    for corpus in corpus_split:
        if corpus in special_tokens:
            continue
        words = GPT2_REGEX.findall(corpus)
        for word in words:
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_count[word_bytes] = word_count.get(word_bytes, 0) + 1

    return word_count


def merge_word_counts(counts_list: list[WordCounts]) -> WordCounts:
    """Merge multiple WordCounts dicts into one."""
    merged: WordCounts = {}
    for counts in counts_list:
        for word, count in counts.items():
            merged[word] = merged.get(word, 0) + count
    return merged


def pretokenize_parallel(
    input_path: str,
    special_tokens: list[str],
    num_workers: int = 4,
) -> WordCounts:
    """Pre-tokenize input in parallel using chunking."""
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    chunks = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(pretokenize_chunk, chunks)

    return merge_word_counts(results)


def build_initial_vocab(special_tokens: list[str]) -> Vocab:
    """Build initial vocabulary with 256 single bytes + special tokens."""
    vocab = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    n = len(vocab)

    for i in range(256):
        vocab[i + n] = bytes([i])

    return vocab


def find_most_frequent_pair(
    word_counts: WordCounts, vocab: Vocab
) -> tuple[bytes, bytes] | None:
    """Find the most frequent adjacent pair across all words."""
    counts: PairCounts = {}
    for word in word_counts:
        for x, y in zip(word, word[1:]):
            if (x, y) not in vocab:
                counts[(x, y)] = counts.get((x, y), 0) + word_counts[word]

    if not counts:
        return None

    return max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))


def apply_merge(word_counts: WordCounts, pair: tuple[bytes, bytes]) -> WordCounts:
    """Apply a merge to all words, combining the pair into a single token."""
    new_word_counts = {}
    merged = pair[0] + pair[1]
    for word, count in word_counts.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_counts[tuple(new_word)] = (
            new_word_counts.get(tuple(new_word), 0) + count
        )

    return new_word_counts
