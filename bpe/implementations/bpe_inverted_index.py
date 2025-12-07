"""BPE implementation with inverted index for faster merges."""

from __future__ import annotations
import os
from typing import BinaryIO
from multiprocessing import Pool

import regex

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
PairCounts = dict[tuple[bytes, bytes], int]
PairToWords = dict[tuple[bytes, bytes], set[int]]  # NEW: inverted index

Words = list[list[bytes]]
WordFreq = list[int]

WordCounts = dict[tuple[bytes, ...], int]

GPT2_REGEX = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def merge_word_counts(
    counts_list: list[WordCounts],
) -> tuple[Words, WordFreq, PairCounts, PairToWords]:
    """Merge multiple WordCounts dicts into one, building pair counts and inverted index."""

    word_to_id: dict[tuple[bytes, ...], int] = {}
    words: Words = []
    word_freq: WordFreq = []

    # First pass: merge word counts (same as before)
    for counts in counts_list:
        for word_bytes, count in counts.items():
            if word_bytes in word_to_id:
                word_id = word_to_id[word_bytes]
                word_freq[word_id] += count
            else:
                word_id = len(words)
                word_to_id[word_bytes] = word_id
                words.append(list(word_bytes))
                word_freq.append(count)

    # NEW: Build initial pair_counts and pair_to_words
    pair_counts: PairCounts = {}
    pair_to_words: PairToWords = {}

    for word_id, word in enumerate(words):
        freq = word_freq[word_id]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            # Update count
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            # Update inverted index
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word_id)

    return words, word_freq, pair_counts, pair_to_words


def find_most_frequent_pair(
    pair_counts: PairCounts,
) -> tuple[bytes, bytes] | None:
    """Find the most frequent pair - now O(n) over pairs, not over all tokens."""
    if not pair_counts:
        return None
    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))


def get_word_pairs(word: list[bytes]) -> list[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word."""
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def apply_merge(
    words: Words,
    word_freq: WordFreq,
    pair: tuple[bytes, bytes],
    pair_counts: PairCounts,
    pair_to_words: PairToWords,
) -> None:
    """Apply merge and update pair_counts and pair_to_words by comparing old/new pairs."""
    merged = pair[0] + pair[1]
    affected_word_ids = pair_to_words.pop(pair, set())
    del pair_counts[pair]

    for word_id in affected_word_ids:
        word = words[word_id]
        freq = word_freq[word_id]

        old_pairs = get_word_pairs(word)

        i = 0
        while i < len(word) - 1:
            if word[i] == pair[0] and word[i + 1] == pair[1]:
                word[i] = merged
                del word[i + 1]
            else:
                i += 1

        new_pairs = get_word_pairs(word)

        for p in old_pairs:
            if p == pair:
                continue
            pair_counts[p] -= freq
            if pair_counts[p] == 0:
                del pair_counts[p]
                pair_to_words.pop(p, None)
            else:
                pair_to_words[p].discard(word_id)

        for p in new_pairs:
            pair_counts[p] = pair_counts.get(p, 0) + freq
            if p not in pair_to_words:
                pair_to_words[p] = set()
            pair_to_words[p].add(word_id)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    """Train a BPE tokenizer and return vocabulary and merges."""
    words, word_freq, pair_counts, pair_to_words = pretokenize(input_path, special_tokens)
    vocab = build_initial_vocab(special_tokens)
    merges: Merges = []

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        best_pair = find_most_frequent_pair(pair_counts)
        if best_pair is None:
            break
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        apply_merge(words, word_freq, best_pair, pair_counts, pair_to_words)

    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently."""
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

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


def pretokenize(
    input_path: str, special_tokens: list[str], num_workers: int = 4
) -> tuple[Words, WordFreq, PairCounts, PairToWords]:
    """Pre-tokenize input, returning words, frequencies, pair counts, and inverted index."""
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
