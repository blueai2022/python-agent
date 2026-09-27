# icd10-agent (RAG as a tool call)

A Python agent that calls RAG-based retrieval as a *tool*, so the upstream
agent does not make a mistake due to poor "memory" — a non-existent ICD-10
code, or an imprecise one.

## A production RAG-calling agent pipeline

A production RAG + LLM-selector pipeline (the shape this repo is exported
from) typically has extracted ICDs and quoted text. This agent adds a
corrective step: instead of trusting the upstream candidate code, it
retrieves grounded candidates from a semantic index and lets the model reason
over them before deciding.

## Requirements

- Python 3.12+
- An OpenAI-compatible endpoint with **tool-calling support** (chat) and an
  OpenAI-compatible **embeddings** endpoint.

## Running it

Install the project and its dev dependencies:

```bash
make install   # python3 -m venv .venv && pip install -e ".[dev]"
```

**Local, via Ollama** (no API key; the models must support tools/embeddings):

```bash
ollama pull llama3.1:8b        # tool-calling capable model
ollama pull mxbai-embed-large  # embedding model

make index-build
```

Or run the commands directly:

```bash
.venv/bin/icd10 index-build --corpus data/icd10_sample.json --index index.db
```
