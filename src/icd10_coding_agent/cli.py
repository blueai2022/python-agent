"""Typer CLI: build the semantic index and run a single case."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._embeddings import EmbeddingBackend
from .code_aliasing import load_aliases, validate_aliases
from .coding_agent import CaseInput, CodingAgent
from .icd10_corpus import load_corpus
from .llm_tool_calling import ChatBackend
from .semantic_index_build import build_index

app = typer.Typer()


class Settings(BaseSettings):
    """Resolved connection settings: command line > environment > built-in defaults."""

    model_config = SettingsConfigDict(env_prefix="ICD10_AGENT_", extra="ignore")

    corpus_path: Path = Path("corpus.json")
    alias_path: Path | None = None
    index_path: Path = Path("index.db")
    system_prompt_path: Path | None = None

    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    model: str = "llama3.1:8b"

    rag_embed_base_url: str = "http://localhost:11434/v1"
    rag_embed_api_key: str | None = None
    rag_embed_model: str = "mxbai-embed-large"


def resolve_settings(**overrides) -> Settings:
    """Layer command-line overrides over environment values over built-in defaults."""
    settings = Settings()
    for name, value in overrides.items():
        if value is not None:
            setattr(settings, name, value)
    return settings


@app.command("index-build")
def index_build(
    corpus: Path | None = typer.Option(None, "--corpus", help="Corpus JSON/CSV path"),
    aliases: Path | None = typer.Option(None, "--aliases", help="Alias JSON/CSV path"),
    index: Path | None = typer.Option(None, "--index", help="Output index path"),
    rag_embed_base_url: str | None = typer.Option(None, "--rag-embed-base-url"),
    rag_embed_api_key: str | None = typer.Option(None, "--rag-embed-api-key"),
    rag_embed_model: str | None = typer.Option(None, "--rag-embed-model"),
) -> None:
    settings = resolve_settings(
        corpus_path=corpus,
        alias_path=aliases,
        index_path=index,
        rag_embed_base_url=rag_embed_base_url,
        rag_embed_api_key=rag_embed_api_key,
        rag_embed_model=rag_embed_model,
    )
    corpus_obj = load_corpus(settings.corpus_path)
    alias_objs = load_aliases(settings.alias_path) if settings.alias_path else []
    if alias_objs:
        validate_aliases(alias_objs, corpus_obj)
    embeddings = EmbeddingBackend(
        model=settings.rag_embed_model,
        base_url=settings.rag_embed_base_url,
        api_key=settings.rag_embed_api_key,
    )
    build_index(corpus_obj, embeddings, settings.index_path, aliases=alias_objs)
    typer.echo(f"Built index at {settings.index_path}")


@app.command("run")
def run(
    evidence: str = typer.Argument(..., help="Quoted clinical evidence text"),
    upstream_code: str = typer.Argument(..., help="Upstream candidate code"),
    index: Path | None = typer.Option(None, "--index", help="Index artifact path"),
    system_prompt: Path | None = typer.Option(None, "--system-prompt"),
    base_url: str | None = typer.Option(None, "--base-url"),
    api_key: str | None = typer.Option(None, "--api-key"),
    model: str | None = typer.Option(None, "--model"),
    rag_embed_base_url: str | None = typer.Option(None, "--rag-embed-base-url"),
    rag_embed_api_key: str | None = typer.Option(None, "--rag-embed-api-key"),
    rag_embed_model: str | None = typer.Option(None, "--rag-embed-model"),
    verbose: bool = typer.Option(False, "--verbose", help="Print the full transcript"),
) -> None:
    settings = resolve_settings(
        index_path=index,
        system_prompt_path=system_prompt,
        base_url=base_url,
        api_key=api_key,
        model=model,
        rag_embed_base_url=rag_embed_base_url,
        rag_embed_api_key=rag_embed_api_key,
        rag_embed_model=rag_embed_model,
    )
    backend = ChatBackend(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
    embeddings = EmbeddingBackend(
        model=settings.rag_embed_model,
        base_url=settings.rag_embed_base_url,
        api_key=settings.rag_embed_api_key,
    )
    system_prompt_text = (
        settings.system_prompt_path.read_text(encoding="utf-8")
        if settings.system_prompt_path
        else None
    )
    agent = CodingAgent(
        backend,
        embeddings,
        settings.index_path,
        system_prompt=system_prompt_text,
    )
    answer = agent.decide(CaseInput(evidence=evidence, upstream_candidate_code=upstream_code))
    if verbose:
        _print_transcript(agent.messages)
    typer.echo(json.dumps(answer.model_dump()))


def _print_transcript(messages) -> None:
    typer.echo("Transcript:")
    for message in messages:
        typer.echo(f"[{message.role}] {message.content or ''}")
        for tool_call in message.tool_calls:
            typer.echo(
                f"  -> tool call {tool_call.name}({json.dumps(tool_call.arguments)})"
            )


if __name__ == "__main__":
    app()
