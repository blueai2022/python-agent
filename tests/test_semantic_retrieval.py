import pytest

from icd10_coding_agent.code_aliasing import Alias
from icd10_coding_agent.icd10_corpus import Corpus, CorpusEntry
from icd10_coding_agent.semantic_index_build import build_index
from icd10_coding_agent.semantic_retrieval import RetrievalError, retrieve


class DictEmbedder:
    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[text] for text in texts]


@pytest.fixture
def index_path(tmp_path):
    corpus = Corpus(
        [
            CorpusEntry(code="I10", description="alpha", category="I10", billable=True),
            CorpusEntry(code="E11.9", description="beta", category="E11", billable=True),
        ]
    )
    aliases = [Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")]
    mapping = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "HTN": [0.0, 0.0, 1.0],
    }
    path = tmp_path / "index.db"
    build_index(corpus, DictEmbedder("model-A", mapping), path, aliases=aliases)
    return path


def test_retrieve_returns_ordered_results(index_path):
    embedder = DictEmbedder("model-A", {"q": [0.9, 0.1, 0.0]})
    results = retrieve("q", 2, embedder, index_path)
    assert [result.code for result in results] == ["I10", "E11.9"]
    assert results[0].score > results[1].score


def test_retrieve_k_exceeds_size(index_path):
    embedder = DictEmbedder("model-A", {"q": [0.9, 0.1, 0.0]})
    results = retrieve("q", 10, embedder, index_path)
    assert len(results) == 3


def test_retrieve_model_mismatch(index_path):
    embedder = DictEmbedder("model-B", {"q": [0.9, 0.1, 0.0]})
    with pytest.raises(RetrievalError):
        retrieve("q", 2, embedder, index_path)


def test_retrieve_plain_entry_display_text(index_path):
    embedder = DictEmbedder("model-A", {"q": [1.0, 0.0, 0.0]})
    results = retrieve("q", 1, embedder, index_path)
    assert results[0].display_text == "alpha"
    assert results[0].code == "I10"


def test_retrieve_alias_display_text(index_path):
    embedder = DictEmbedder("model-A", {"q": [0.0, 0.0, 1.0]})
    results = retrieve("q", 1, embedder, index_path)
    assert results[0].display_text == "Hypertension - HTN"
    assert results[0].code == "I10"


def test_retrieve_empty_index(tmp_path):
    path = tmp_path / "empty.db"
    build_index(Corpus([]), DictEmbedder("model-A", {}), path)
    embedder = DictEmbedder("model-A", {"q": [0.0, 0.0, 0.0]})
    with pytest.raises(RetrievalError):
        retrieve("q", 1, embedder, path)


def test_retrieve_rejects_non_positive_k(index_path):
    embedder = DictEmbedder("model-A", {"q": [1.0, 0.0, 0.0]})
    with pytest.raises(RetrievalError):
        retrieve("q", 0, embedder, index_path)
    with pytest.raises(RetrievalError):
        retrieve("q", -1, embedder, index_path)
