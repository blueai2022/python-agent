"""Build a persisted sqlite-vec semantic index from the corpus and aliases."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ._embeddings import Embedder
from .code_aliasing import Alias
from .icd10_corpus import Corpus

FORMAT_VERSION = 1
DEFAULT_BATCH_SIZE = 32

_META_FORMAT_VERSION = "format_version"
_META_MODEL = "model"
_META_DIMENSION = "dimension"


class IndexBuildError(Exception):
    """Raised when an index cannot be built or loaded."""


@dataclass(frozen=True)
class IndexEntry:
    """One searchable row: a corpus entry or an alias projected for search."""

    code: str
    match_text: str
    display_text: str
    cross_references: tuple[str, ...] = ()


def build_index_entries(corpus: Corpus, aliases: Iterable[Alias] = ()) -> list[IndexEntry]:
    """Project corpus entries and aliases into one searchable row each."""
    entries: list[IndexEntry] = []
    for entry in corpus:
        entries.append(
            IndexEntry(
                code=entry.code,
                match_text=entry.description,
                display_text=entry.description,
                cross_references=entry.clinical_notes.cross_references,
            )
        )
    for alias in aliases:
        entries.append(
            IndexEntry(
                code=alias.target_code,
                match_text=alias.short_form,
                display_text=alias.display_text,
            )
        )
    return entries


def build_index(
    corpus: Corpus,
    embeddings: Embedder,
    output_path: str | Path,
    aliases: Iterable[Alias] = (),
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Build and persist a semantic index from the corpus and optional aliases."""
    entries = build_index_entries(corpus, aliases)
    vectors = _embed_in_batches(embeddings, [entry.match_text for entry in entries], batch_size)
    _write_index(Path(output_path), entries, vectors, embeddings.model)
    return Path(output_path)


def _embed_in_batches(
    embeddings: Embedder, texts: Sequence[str], batch_size: int
) -> list[list[float]]:
    if batch_size <= 0:
        raise IndexBuildError("batch size must be positive")
    vectors: list[list[float]] = []
    dimension: Optional[int] = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = embeddings.embed(batch)
        if len(batch_vectors) != len(batch):
            raise IndexBuildError(
                f"embedding backend returned {len(batch_vectors)} vectors "
                f"for a batch of {len(batch)}"
            )
        for vector in batch_vectors:
            if len(vector) == 0:
                raise IndexBuildError("embedding backend returned an empty vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise IndexBuildError(
                    f"embedding dimensionality changed mid-build: expected "
                    f"{dimension}, got {len(vector)}"
                )
        vectors.extend(batch_vectors)
    return vectors


def _write_index(
    path: Path, entries: Sequence[IndexEntry], vectors: Sequence[list[float]], model: str
) -> None:
    dimension = len(vectors[0]) if vectors else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        _write_index_file(tmp, entries, vectors, model, dimension)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_index_file(
    path: Path,
    entries: Sequence[IndexEntry],
    vectors: Sequence[list[float]],
    model: str,
    dimension: int,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        _load_vec_extension(conn)
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE entries (
                id INTEGER PRIMARY KEY,
                code TEXT NOT NULL,
                match_text TEXT NOT NULL,
                display_text TEXT NOT NULL,
                cross_references TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?), (?, ?), (?, ?)",
            [
                _META_FORMAT_VERSION,
                str(FORMAT_VERSION),
                _META_MODEL,
                model,
                _META_DIMENSION,
                str(dimension),
            ],
        )
        if entries:
            conn.execute(
                f"CREATE VIRTUAL TABLE vectors USING vec0("
                f"embedding float[{dimension}] distance_metric=cosine)"
            )
            for row_id, (entry, vector) in enumerate(zip(entries, vectors), start=1):
                conn.execute(
                    "INSERT INTO entries (id, code, match_text, display_text, "
                    "cross_references) VALUES (?, ?, ?, ?, ?)",
                    [
                        row_id,
                        entry.code,
                        entry.match_text,
                        entry.display_text,
                        json.dumps(list(entry.cross_references)),
                    ],
                )
                conn.execute(
                    "INSERT INTO vectors (rowid, embedding) VALUES (?, ?)",
                    [row_id, json.dumps(vector)],
                )
        conn.commit()
    finally:
        conn.close()


class Index:
    """A loaded, reloadable semantic index."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str,
        dimension: int,
        entries: tuple[IndexEntry, ...],
    ) -> None:
        self._conn = conn
        self._model = model
        self._dimension = dimension
        self._entries = entries

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def entries(self) -> tuple[IndexEntry, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, query_vector: Sequence[float], k: int) -> list[tuple[IndexEntry, float]]:
        """Return up to ``k`` (entry, cosine-similarity) pairs, most similar first."""
        if not self._entries:
            return []
        rows = self._conn.execute(
            "SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = ?",
            [json.dumps(list(query_vector)), k],
        ).fetchall()
        results: list[tuple[IndexEntry, float]] = []
        for row_id, distance in rows:
            idx = int(row_id) - 1
            if 0 <= idx < len(self._entries):
                results.append((self._entries[idx], 1.0 - float(distance)))
        return results

    def close(self) -> None:
        self._conn.close()


def load_index(path: str | Path) -> Index:
    """Load a persisted index artifact without re-embedding anything."""
    path = Path(path)
    if not path.is_file():
        raise IndexBuildError(f"index artifact not found: {path}")
    conn = sqlite3.connect(str(path))
    try:
        _load_vec_extension(conn)
        meta = _read_meta(conn)
        version = meta.get(_META_FORMAT_VERSION)
        if version != str(FORMAT_VERSION):
            raise IndexBuildError(f"unsupported index format version: {version!r}")
        model = meta.get(_META_MODEL)
        if model is None:
            raise IndexBuildError("index artifact is missing its embedding model")
        try:
            dimension = int(meta.get(_META_DIMENSION, "0"))
        except ValueError as exc:
            raise IndexBuildError("index artifact has an invalid dimension") from exc
        entries = _read_entries(conn)
    except IndexBuildError:
        conn.close()
        raise
    except Exception as exc:
        conn.close()
        raise IndexBuildError(f"failed to load index artifact: {exc}") from exc
    return Index(conn, model, dimension, entries)


def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.OperationalError as exc:
        raise IndexBuildError(f"not a valid index artifact: {exc}") from exc
    return dict(rows)


def _read_entries(conn: sqlite3.Connection) -> tuple[IndexEntry, ...]:
    try:
        rows = conn.execute(
            "SELECT id, code, match_text, display_text, cross_references "
            "FROM entries ORDER BY id"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise IndexBuildError(f"not a valid index artifact: {exc}") from exc
    return tuple(
        IndexEntry(
            code=code,
            match_text=match_text,
            display_text=display_text,
            cross_references=tuple(json.loads(xrefs)),
        )
        for (_row_id, code, match_text, display_text, xrefs) in rows
    )


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise IndexBuildError("the sqlite-vec package is not installed") from exc
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        raise IndexBuildError(f"could not load the sqlite-vec extension: {exc}") from exc
