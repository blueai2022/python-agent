"""OpenAI-compatible embeddings backend."""

from __future__ import annotations

from typing import Protocol, Sequence


class Embedder(Protocol):
    """Minimal embeddings interface consumed by index build and retrieval."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingBackend:
    """Embeds text via an OpenAI-compatible embeddings endpoint (`openai` SDK).

    ``base_url``, ``api_key``, and ``model`` are configurable so that OpenAI,
    Azure OpenAI, or any OpenAI-compatible local proxy (Ollama, vLLM, LiteLLM)
    can be used interchangeably.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        from openai import OpenAI

        # Local OpenAI-compatible servers (Ollama, vLLM, LiteLLM) don't require a
        # real key, but the SDK rejects a None key — send a placeholder.
        client = OpenAI(base_url=self.base_url, api_key=self.api_key or "sk-local")
        response = client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]
