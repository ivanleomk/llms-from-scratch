.PHONY: install benchmark clean

install:
	uv sync

benchmark:
	uv run python benchmark.py

benchmark-all:
	uv run python benchmark.py --tests corpus_en tinystories

clean:
	rm -rf __pycache__ .venv bpe_benchmark.egg-info
