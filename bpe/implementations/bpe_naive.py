"""Naive BPE implementation extracted from assignment1-basics."""

from __future__ import annotations

import regex

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
WordCounts = dict[tuple[bytes, ...], int]
PairCounts = dict[tuple[bytes, bytes], int]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    """Train a BPE tokenizer and return vocabulary and merges."""
    word_counts = pretokenize(input_path, special_tokens)
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


def pretokenize(input_path: str, special_tokens: list[str]) -> WordCounts:
    """Pre-tokenize input into words/chunks, returning counts of each word as tuple of bytes."""
    with open(input_path) as f:
        corpus = f.read()

    corpus_split = [corpus]

    if special_tokens:
        pattern = r"(" + "|".join(regex.escape(token) for token in special_tokens) + r")"
        corpus_split = regex.split(pattern, corpus)
        corpus_split = [x for x in corpus_split if x]

    gpt_2_regex = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    word_count: WordCounts = {}
    for corpus in corpus_split:
        if corpus in special_tokens:
            continue
        words = gpt_2_regex.findall(corpus)
        for word in words:
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_count[word_bytes] = word_count.get(word_bytes, 0) + 1

    return word_count


def build_initial_vocab(special_tokens: list[str]) -> Vocab:
    """Build initial vocabulary with 256 single bytes + special tokens."""
    vocab = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    n = len(vocab)

    for i in range(256):
        vocab[i + n] = bytes([i])

    return vocab


def find_most_frequent_pair(word_counts: WordCounts, vocab: Vocab) -> tuple[bytes, bytes] | None:
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
        new_word_counts[tuple(new_word)] = new_word_counts.get(tuple(new_word), 0) + count

    return new_word_counts
