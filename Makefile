# Local development targets for the ICD-10 coding agent.
# By default the CLI talks to a local Ollama server.

VENV := .venv
ICD10 := $(VENV)/bin/icd10

# Chat backend (llama3.1:8b via Ollama's OpenAI-compatible endpoint).
BASE_URL ?= http://localhost:11434/v1
MODEL ?= llama3.1:8b

# Embedding backend.
RAG_EMBED_BASE_URL ?= http://localhost:11434/v1
RAG_EMBED_MODEL ?= mxbai-embed-large

# Corpus / index files.
CORPUS ?= data/icd10_sample.json
ALIASES ?= data/aliases.json
RAG_INDEX ?= index.db

# Case inputs (override on the command line):
#   make run QUOTED="patient presents with ..." CODE="I10"
QUOTED ?= DM2 with neuropathy
CODE ?= E11.9

.PHONY: install index-build run test clean

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

index-build:
	$(ICD10) index-build --corpus $(CORPUS) --aliases $(ALIASES) --index $(RAG_INDEX) \
		--rag-embed-base-url $(RAG_EMBED_BASE_URL) --rag-embed-model $(RAG_EMBED_MODEL)

run:
	$(ICD10) run "$(QUOTED)" "$(CODE)" --base-url $(BASE_URL) --model $(MODEL) \
		--index $(RAG_INDEX) --verbose

test:
	$(VENV)/bin/python -m pytest

clean:
	rm -rf $(RAG_INDEX) .pytest_cache src/*.egg-info
