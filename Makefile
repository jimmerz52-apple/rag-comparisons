.PHONY: run chart install setup

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -e .
	.venv/bin/python -m spacy download en_core_web_sm

setup: install
	brew services start ollama || true
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text

run:
	.venv/bin/python -m rag_benchmark run

chart:
	.venv/bin/python -m rag_benchmark chart

run-fresh:
	.venv/bin/python -m rag_benchmark run --no-reuse

semantic-only:
	.venv/bin/python -m rag_benchmark run --skip-fetch --methods semantic_rag
