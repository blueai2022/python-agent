# Local development targets for the ICD-10 coding agent.
# By default the CLI talks to a local Ollama server.

VENV := .venv
ICD10 := $(VENV)/bin/icd10

# Embedding backend.
RAG_EMBED_BASE_URL ?= http://localhost:11434/v1
RAG_EMBED_MODEL ?= mxbai-embed-large

# Corpus / index files.
CORPUS ?= data/icd10_sample.json
ALIASES ?= data/aliases.json
RAG_INDEX ?= index.db

.PHONY: install index-build

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

index-build:
	$(ICD10) index-build --corpus $(CORPUS) --aliases $(ALIASES) --index $(RAG_INDEX) \
		--rag-embed-base-url $(RAG_EMBED_BASE_URL) --rag-embed-model $(RAG_EMBED_MODEL)
