"""Naive BPE implementation extracted from assignment1-basics."""

from __future__ import annotations

import regex

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
PairCounts = dict[tuple[bytes, bytes], int]

Words = list[list[bytes]]  # words[word_id] = list of tokens
WordFreq = list[int]  # word_freq[word_id] = corpus frequency


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    """Train a BPE tokenizer and return vocabulary and merges."""
    words, word_freq = pretokenize(input_path, special_tokens)
    vocab = build_initial_vocab(special_tokens)
    merges: Merges = []

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        best_pair = find_most_frequent_pair(words, word_freq, vocab)
        if best_pair is None:
            break
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        apply_merge(words, best_pair)

    return vocab, merges


def pretokenize(input_path: str, special_tokens: list[str]) -> tuple[Words, WordFreq]:
    """Pre-tokenize input into words/chunks, returning words and their frequencies."""
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

    word_to_id: dict[tuple[bytes, ...], int] = {}
    words: Words = []
    word_freq: WordFreq = []

    for corpus_chunk in corpus_split:
        if corpus_chunk in special_tokens:
            continue
        matched_words = gpt_2_regex.findall(corpus_chunk)
        for word in matched_words:
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            if word_bytes in word_to_id:
                word_id = word_to_id[word_bytes]
                word_freq[word_id] += 1
            else:
                word_id = len(words)
                word_to_id[word_bytes] = word_id
                words.append(list(word_bytes))
                word_freq.append(1)

    return words, word_freq


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
    words: Words, word_freq: WordFreq, vocab: Vocab
) -> tuple[bytes, bytes] | None:
    """Find the most frequent adjacent pair across all words."""
    counts: PairCounts = {}
    for word_id, word in enumerate(words):
        freq = word_freq[word_id]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in vocab:
                counts[pair] = counts.get(pair, 0) + freq

    if not counts:
        return None

    return max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))


def apply_merge(words: Words, pair: tuple[bytes, bytes]) -> None:
    """Apply a merge to all words in-place, combining the pair into a single token."""
    merged = pair[0] + pair[1]
    for word in words:
        i = 0
        while i < len(word) - 1:
            if word[i] == pair[0] and word[i + 1] == pair[1]:
                word[i] = merged
                del word[i + 1]
            else:
                i += 1
