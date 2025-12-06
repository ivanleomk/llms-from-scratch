# BPE Benchmark

Benchmark different BPE training implementations.

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
- `--implementation`: BPE implementation to use (`bpe_naive`)
- `--vocab-size`: Target vocabulary size (default: 500)
- `--special-tokens`: Special tokens to add (default: `<|endoftext|>`)

## Adding New Implementations

1. Create a new file in `implementations/` (e.g., `bpe_optimized.py`)
2. Implement a `train_bpe(input_path, vocab_size, special_tokens)` function
3. Register it in `implementations/__init__.py`
