"""Top-k semantic search over a persisted index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._embeddings import Embedder
from .semantic_index_build import load_index


class RetrievalError(Exception):
    """Raised when a retrieval request is invalid or cannot be served."""


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval hit."""

    code: str
    display_text: str
    cross_references: tuple[str, ...]
    score: float


def retrieve(
    query: str,
    k: int,
    embeddings: Embedder,
    index_path: str | Path,
) -> list[RetrievalResult]:
    """Return up to ``k`` entries ranked by descending cosine similarity."""
    if k <= 0:
        raise RetrievalError("k must be positive")

    index = load_index(index_path)
    try:
        if len(index) == 0:
            raise RetrievalError("the index has no entries")
        if embeddings.model != index.model:
            raise RetrievalError(
                f"query embedding model {embeddings.model!r} does not match "
                f"index model {index.model!r}"
            )
        query_vectors = embeddings.embed([query])
        if len(query_vectors) != 1:
            raise RetrievalError(
                f"embedding backend returned {len(query_vectors)} vectors for one query"
            )
        hits = index.search(query_vectors[0], k)
        return [
            RetrievalResult(
                code=entry.code,
                display_text=entry.display_text,
                cross_references=entry.cross_references,
                score=score,
            )
            for entry, score in hits
        ]
    finally:
        index.close()
