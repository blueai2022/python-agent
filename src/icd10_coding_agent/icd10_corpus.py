"""Load and represent the ICD-10 corpus of billable, leaf-level codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from ._source import load_records, optional_string

CATEGORY_PREFIX_LENGTH = 3

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


class CorpusError(Exception):
    """Raised when a corpus source cannot be loaded."""


@dataclass(frozen=True)
class ClinicalNotes:
    """Optional clinical notes attached to a corpus entry."""

    inclusion_terms: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    cross_references: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusEntry:
    """A single billable, leaf-level corpus entry."""

    code: str
    description: str
    category: str
    billable: bool
    clinical_notes: ClinicalNotes = field(default_factory=ClinicalNotes)


class Corpus:
    """An immutable collection of corpus entries with case-insensitive lookup."""

    def __init__(self, entries: Iterable[CorpusEntry]) -> None:
        self._entries: tuple[CorpusEntry, ...] = tuple(entries)
        self._by_code: Mapping[str, CorpusEntry] = {
            entry.code.casefold(): entry for entry in self._entries
        }

    @property
    def entries(self) -> tuple[CorpusEntry, ...]:
        return self._entries

    def lookup(self, code: str) -> Optional[CorpusEntry]:
        """Return the entry for ``code`` (case-insensitive), or ``None`` if absent."""
        return self._by_code.get(code.casefold())

    def __iter__(self) -> Iterator[CorpusEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class _RawRecord:
    code: str
    description: str
    category: Optional[str]
    billable: Optional[bool]
    parent: Optional[str]
    inclusion_terms: tuple[str, ...]
    excludes: tuple[str, ...]
    cross_references: tuple[str, ...]


def load_corpus(source: str | Path) -> Corpus:
    """Load a corpus from a JSON or CSV export and build a leaf-only ``Corpus``."""
    records = [
        _mapping_to_record(mapping, index)
        for index, mapping in enumerate(load_records(source, "corpus"), start=1)
    ]
    return _build_corpus(records)


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _parse_bool(value: Any, index: int) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise CorpusError(f"record {index}: invalid billable value {value!r}")


def _mapping_to_record(mapping: Mapping[str, Any], index: int) -> _RawRecord:
    code = optional_string(mapping.get("code"))
    description = optional_string(mapping.get("description"))
    if not code:
        raise CorpusError(f"record {index}: missing required field 'code'")
    if not description:
        raise CorpusError(f"record {index}: missing required field 'description'")
    return _RawRecord(
        code=code,
        description=description,
        category=optional_string(mapping.get("category")),
        billable=_parse_bool(mapping.get("billable"), index),
        parent=optional_string(mapping.get("parent")),
        inclusion_terms=_string_list(mapping.get("inclusion_terms")),
        excludes=_string_list(mapping.get("excludes")),
        cross_references=_string_list(mapping.get("cross_references")),
    )


def _build_corpus(records: Sequence[_RawRecord]) -> Corpus:
    parent_codes = {record.parent for record in records if record.parent}
    entries: list[CorpusEntry] = []
    for record in records:
        billable = (
            record.billable
            if record.billable is not None
            else record.code not in parent_codes
        )
        if not billable:
            continue
        entries.append(
            CorpusEntry(
                code=record.code,
                description=record.description,
                category=record.category or record.code[:CATEGORY_PREFIX_LENGTH],
                billable=True,
                clinical_notes=ClinicalNotes(
                    inclusion_terms=record.inclusion_terms,
                    excludes=record.excludes,
                    cross_references=record.cross_references,
                ),
            )
        )
    return Corpus(entries)
