from icd10_coding_agent._embeddings import EmbeddingBackend


class FakeEmbedding:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeEmbeddings:
    def __init__(self, received):
        self.received = received

    def create(self, *, model, input):
        self.received["model"] = model
        self.received["input"] = input
        return FakeResponse(
            [FakeEmbedding(i, [float(i)] * 3) for i in range(len(input))]
        )


class FakeOpenAI:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key


def test_embedding_backend_config(monkeypatch):
    received = {}

    def fake_openai(base_url=None, api_key=None):
        received["base_url"] = base_url
        received["api_key"] = api_key
        client = FakeOpenAI(base_url=base_url, api_key=api_key)
        client.embeddings = FakeEmbeddings(received)
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    backend = EmbeddingBackend(
        model="text-embedding-3-small",
        base_url="http://localhost:11434/v1",
        api_key="sk-test",
    )
    vectors = backend.embed(["a", "b"])

    assert received["base_url"] == "http://localhost:11434/v1"
    assert received["api_key"] == "sk-test"
    assert received["model"] == "text-embedding-3-small"
    assert received["input"] == ["a", "b"]
    assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_embedding_backend_orders_by_index(monkeypatch):
    class OutOfOrderEmbeddings:
        def create(self, *, model, input):
            return FakeResponse([FakeEmbedding(1, [1.0, 1.0, 1.0]), FakeEmbedding(0, [0.0, 0.0, 0.0])])

    class Client:
        def __init__(self, base_url=None, api_key=None):
            self.embeddings = OutOfOrderEmbeddings()

    monkeypatch.setattr("openai.OpenAI", Client)

    backend = EmbeddingBackend(model="m")
    assert backend.embed(["a", "b"]) == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_embedding_backend_placeholder_key_when_none(monkeypatch):
    captured = {}

    class CaptureEmbeddings:
        def create(self, *, model, input):
            return FakeResponse([FakeEmbedding(0, [1.0, 1.0, 1.0])])

    def fake_openai(base_url=None, api_key=None):
        captured["api_key"] = api_key
        client = FakeOpenAI(base_url=base_url, api_key=api_key)
        client.embeddings = CaptureEmbeddings()
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    EmbeddingBackend(model="m").embed(["a"])
    assert captured["api_key"] == "sk-local"
