# Introduction

This is a repository called llms-from-scratch that contain the code for building a language model from scratch. I'm using this to benchmark different changes that I've made and using them to learn more about the inner workings about a language model.

> 📖 Check out my blog post [Implementing BPE](https://ivanleo.com/blog/implementing-bpe) for a detailed walkthrough of how BPE tokenization works.

## Implementations

### [BPE Tokenizer](./bpe/)

A collection of BPE (Byte Pair Encoding) tokenizer implementations with progressive optimizations:

| Implementation          | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `bpe_naive`             | Basic BPE implementation - creates new word counts dict on each merge     |
| `bpe_in_place`          | In-place merges on mutable word lists - avoids dict recreation            |
| `bpe_parallel`          | Parallel pretokenization using multiprocessing                            |
| `bpe_in_place_parallel` | Combines in-place merges with parallel pretokenization                    |
| `bpe_inverted_index`    | Uses inverted index for O(affected) merge updates instead of O(all words) |
