from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 100
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.total_tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            self.total_tokens += response.usage.total_tokens
            vectors.extend(item.embedding for item in response.data)
        return vectors
