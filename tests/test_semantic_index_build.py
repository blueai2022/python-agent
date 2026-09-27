import sqlite3

import pytest

from icd10_coding_agent.code_aliasing import Alias
from icd10_coding_agent.icd10_corpus import Corpus, CorpusEntry
from icd10_coding_agent.semantic_index_build import (
    IndexBuildError,
    build_index,
    build_index_entries,
    load_index,
)


class FakeEmbedder:
    """Deterministic per-text vectors, independent of batching."""

    def __init__(self, model="test-model", dim=3):
        self.model = model
        self.dim = dim
        self.batch_count = 0
        self._index = 0

    def embed(self, texts):
        self.batch_count += 1
        vectors = []
        for _ in texts:
            vector = [0.0] * self.dim
            vector[self._index % self.dim] = 1.0
            self._index += 1
            vectors.append(vector)
        return vectors


class EmptyVectorEmbedder:
    model = "test-model"

    def embed(self, texts):
        return [[] for _ in texts]


class ChangingDimEmbedder:
    model = "test-model"

    def __init__(self):
        self.dim = 3

    def embed(self, texts):
        vectors = [[0.0] * self.dim for _ in texts]
        self.dim = 2
        return vectors


def _corpus():
    return Corpus(
        [
            CorpusEntry(
                code="I10",
                description="Essential hypertension",
                category="I10",
                billable=True,
            ),
            CorpusEntry(
                code="E11.9",
                description="Type 2 diabetes",
                category="E11",
                billable=True,
            ),
        ]
    )


def test_build_corpus_only(tmp_path):
    out = tmp_path / "index.db"
    build_index(_corpus(), FakeEmbedder(), out)
    index = load_index(out)
    try:
        assert len(index) == 2
        assert [entry.code for entry in index.entries] == ["I10", "E11.9"]
    finally:
        index.close()


def test_build_corpus_and_aliases(tmp_path):
    out = tmp_path / "index.db"
    aliases = [Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")]
    build_index(_corpus(), FakeEmbedder(), out, aliases=aliases)
    index = load_index(out)
    try:
        assert len(index) == 3
        alias_entry = index.entries[-1]
        assert alias_entry.code == "I10"
        assert alias_entry.match_text == "HTN"
        assert alias_entry.display_text == "Hypertension - HTN"
    finally:
        index.close()


def test_build_index_entries_counts():
    corpus = _corpus()
    aliases = [Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")]
    assert len(build_index_entries(corpus)) == 2
    assert len(build_index_entries(corpus, aliases)) == 3


def test_build_in_multiple_batches(tmp_path):
    corpus = Corpus(
        [
            CorpusEntry(code=f"C{i}", description=f"desc {i}", category="C", billable=True)
            for i in range(5)
        ]
    )
    small_path = tmp_path / "small.db"
    large_path = tmp_path / "large.db"
    build_index(corpus, FakeEmbedder(), small_path, batch_size=2)
    build_index(corpus, FakeEmbedder(), large_path, batch_size=10)

    small = load_index(small_path)
    large = load_index(large_path)
    try:
        assert [entry.code for entry in small.entries] == [
            entry.code for entry in large.entries
        ]
    finally:
        small.close()
        large.close()


def test_build_uses_multiple_batches(tmp_path):
    corpus = Corpus(
        [
            CorpusEntry(code=f"C{i}", description=f"desc {i}", category="C", billable=True)
            for i in range(5)
        ]
    )
    embedder = FakeEmbedder()
    build_index(corpus, embedder, tmp_path / "index.db", batch_size=2)
    assert embedder.batch_count == 3


def test_empty_vector_fails(tmp_path):
    with pytest.raises(IndexBuildError):
        build_index(_corpus(), EmptyVectorEmbedder(), tmp_path / "index.db")


def test_dimension_mismatch_fails(tmp_path):
    with pytest.raises(IndexBuildError) as exc:
        build_index(_corpus(), ChangingDimEmbedder(), tmp_path / "index.db", batch_size=1)
    assert "dimensionality" in str(exc.value)


def test_reload_and_search_without_embedder(tmp_path):
    out = tmp_path / "index.db"
    build_index(_corpus(), FakeEmbedder(dim=3), out)
    index = load_index(out)
    try:
        hits = index.search([1.0, 0.0, 0.0], 1)
        assert len(hits) == 1
        entry, score = hits[0]
        assert entry.code == "I10"
        assert score > 0.0
    finally:
        index.close()


def test_load_missing_file(tmp_path):
    with pytest.raises(IndexBuildError):
        load_index(tmp_path / "does-not-exist.db")


def test_load_incompatible_artifact(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a sqlite database", encoding="utf-8")
    with pytest.raises(IndexBuildError):
        load_index(bad)


def test_load_version_mismatch(tmp_path):
    out = tmp_path / "index.db"
    build_index(_corpus(), FakeEmbedder(), out)
    conn = sqlite3.connect(out)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    with pytest.raises(IndexBuildError) as exc:
        load_index(out)
    assert "version" in str(exc.value)
