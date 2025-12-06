#!/usr/bin/env python3
"""BPE Training Benchmark CLI."""

import argparse
import time
from pathlib import Path

from implementations import IMPLEMENTATIONS

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "corpus": DATA_DIR / "corpus.en",
    "tinystory": DATA_DIR / "tinystories_sample_5M.txt",
}

ALL_DATASETS = ["corpus", "tinystory"]


def run_benchmark(
    dataset: str,
    implementation: str,
    repeat: int,
    vocab_size: int,
    special_tokens: list[str],
) -> None:
    input_path = DATASETS[dataset]
    train_fn = IMPLEMENTATIONS[implementation]

    times = []
    for i in range(repeat):
        start = time.perf_counter()
        vocab, merges = train_fn(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"Run {i + 1}/{repeat}: {elapsed:.4f}s (vocab={len(vocab)}, merges={len(merges)})"
        )

    print(f"\n--- Results for {implementation} on {dataset} ---")
    print(f"Runs: {repeat}")
    print(f"Mean: {sum(times) / len(times):.4f}s")
    print(f"Min:  {min(times):.4f}s")
    print(f"Max:  {max(times):.4f}s")


def main():
    parser = argparse.ArgumentParser(description="BPE Training Benchmark")
    parser.add_argument(
        "--tests",
        choices=list(DATASETS.keys()) + ["all"],
        nargs="+",
        default=["corpus"],
        help="Dataset(s) to use for benchmarking",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark",
    )
    parser.add_argument(
        "--implementation",
        choices=list(IMPLEMENTATIONS.keys()),
        default="bpe_naive",
        help="BPE implementation to benchmark",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to add to vocabulary",
    )

    args = parser.parse_args()

    if "all" in args.tests:
        args.tests = ALL_DATASETS

    for dataset in args.tests:
        run_benchmark(
            dataset=dataset,
            implementation=args.implementation,
            repeat=args.repeat,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
        )


if __name__ == "__main__":
    main()
