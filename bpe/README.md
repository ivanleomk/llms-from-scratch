# BPE Tokenizer

Benchmark different BPE training implementations.

> 📖 Check out [Implementing BPE](https://ivanleo.com/blog/implementing-bpe) for a detailed walkthrough of how BPE tokenization works.

## Implementations

| Implementation | Description |
|----------------|-------------|
| `bpe_naive` | Basic BPE implementation - creates new word counts dict on each merge |
| `bpe_in_place` | In-place merges on mutable word lists - avoids dict recreation |
| `bpe_parallel` | Parallel pretokenization using multiprocessing with chunked file reading |
| `bpe_in_place_parallel` | Combines in-place merges with parallel pretokenization |
| `bpe_inverted_index` | Uses inverted index for O(affected) merge updates instead of O(all words) |

## Setup

```bash
uv venv
uv pip install -e .
```

## Usage

```bash
# Basic usage (corpus_en dataset, 1 run, bpe_naive implementation)
python benchmark.py

# Benchmark on tinystories dataset with 5 repetitions
python benchmark.py --tests tinystories --repeat 5

# Full options
python benchmark.py --tests corpus_en --repeat 3 --implementation bpe_naive --vocab-size 500
```

## Options

- `--tests`: Dataset to use (`corpus_en`, `tinystories`)
- `--repeat`: Number of times to repeat the benchmark
- `--implementation`: BPE implementation to use (see table above)
- `--vocab-size`: Target vocabulary size (default: 500)
- `--special-tokens`: Special tokens to add (default: `<|endoftext|>`)

## Adding New Implementations

1. Create a new file in `implementations/` (e.g., `bpe_optimized.py`)
2. Implement a `train_bpe(input_path, vocab_size, special_tokens)` function
3. Register it in `implementations/__init__.py`
