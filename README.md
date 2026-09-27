# ICD10-agent (RAG as a tool call)

A Python agent that calls RAG-based retrieval as a *tool*, so the upstream
agent does not make a mistake due to poor "memory" — a non-existent ICD-10
code, or an imprecise one.

## A production RAG-calling agent pipeline

A production RAG + LLM-selector pipeline (the shape this repo is exported
from) typically has extracted ICDs and quoted text. This agent adds a
corrective step: instead of trusting the upstream candidate code, it
retrieves grounded candidates from a semantic index and lets the model reason
over them before deciding.

## What's in here

- `src/icd10_coding_agent/icd10_corpus.py` — the corpus model and the
  JSON/CSV loader: category derivation, leaf-only filtering, and
  case-insensitive code lookup.
- `src/icd10_coding_agent/code_aliasing.py` — acronym/abbreviation aliases,
  validated against the corpus and projected as searchable entries.
- `src/icd10_coding_agent/semantic_index_build.py` — builds a persisted
  sqlite-vec index (a single `.db` file) from the corpus + aliases, in
  bounded batches, recording the embedding model and dimensionality.
- `src/icd10_coding_agent/semantic_retrieval.py` — top-k cosine-similarity
  search over the built index.
- `src/icd10_coding_agent/llm_tool_calling.py` — a minimal OpenAI-compatible
  chat-completions client with tool calling, modeled with Pydantic.
- `src/icd10_coding_agent/coding_agent.py` — the loop itself: send messages +
  tool schema, execute any tool calls the model makes, append results, repeat
  until the model answers with JSON instead of a tool call (bounded by a
  max-steps guard).
- `src/icd10_coding_agent/cli.py` — the Typer CLI: `index-build` and `run`.
- `src/icd10_coding_agent/_embeddings.py` — an OpenAI-compatible embeddings
  client (`openai` SDK), configurable by base URL / API key / model.

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
make run QUOTED="diabetic neuropathy" CODE="E11.9"
```

Or run the commands directly:

```bash
.venv/bin/icd10 index-build --corpus data/icd10_sample.json --index index.db
.venv/bin/icd10 run "diabetic neuropathy" "E11.9" --index index.db --verbose
```

**OpenAI**:

```bash
.venv/bin/icd10 index-build \
  --corpus data/icd10_sample.json --index index.db \
  --rag-embed-base-url https://api.openai.com/v1 \
  --rag-embed-model text-embedding-3-small \
  --rag-embed-api-key "$OPENAI_API_KEY"

.venv/bin/icd10 run "DM2 with neuropathy" "E11.9" \
  --index index.db \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"
```

`--verbose` prints the full transcript (system/user/assistant/tool messages,
including every tool call) so you can watch the loop happen.

### Configuration

Defaults target a local Ollama server (`http://localhost:11434/v1`,
`llama3.1:8b`, `mxbai-embed-large`). Every connection setting can be
overridden on the command line or via environment variables under the
`ICD10_AGENT_` prefix — notably `ICD10_AGENT_BASE_URL` and `ICD10_AGENT_MODEL`.

### Example transcript (llama3.1:8b, local)

```
--- user ---
Clinical evidence:
diabetic neuropathy
Upstream candidate code: E11.9
--- assistant ---
tool_call: search({"query":"diabetic nerve damage"})
--- tool ---
[{"code":"E11.42","display_text":"Type 2 diabetes mellitus with diabetic polyneuropathy","score":0.82}, ...]
--- assistant ---
{"code":"E11.42","reason":"evidence names diabetic polyneuropathy, a specific complication"}
```

The model chose its own search phrasing ("diabetic nerve damage", not the
literal input text), got back candidates, and only then answered. Answer
*quality* depends heavily on the model. An 8B-class model (`llama3.1:8b`
above) is the recommended minimum for reliable results: it converges in a
single search + answer, with valid JSON and a code that's actually justified
by the evidence.

Smaller models (e.g. `llama3.2:1b`) still demonstrate the tool-calling
*mechanics* — they'll call `search` and eventually stop — but in practice
they're unreliable at the *reasoning*.

## Makefile targets

- `make install` — creates the virtualenv and installs the package + dev deps.
- `make index-build` — embeds `data/icd10_sample.json` plus its acronym aliases
  (`data/aliases.json`) and writes the sqlite-vec `index.db` artifact the
  agent loads at runtime.
- `make run` — runs the agent CLI; override defaults on the command line,
  e.g. `make run QUOTED="chest pain" CODE=R07.9`.
- `make test` — runs the test suite (`pytest`).
- `make clean` — removes the built index and Python build artifacts.

## Data

`data/icd10_sample.json` — 61 ICD-10-CM codes and their official short titles
(a US government work, not subject to copyright), covering common conditions
plus near-neighbor distractors. This is a small hand-curated educational
subset, **not** a complete or authoritative code set — do not use it for
actual clinical coding or billing.

`data/aliases.json` — 26 acronym/abbreviation aliases (e.g. `HTN` → `I10`),
enriched into the index at build time so a query can match by short form.
Aliases are baked into the index; they are **only** consumed by `index-build`,
not at run time.
