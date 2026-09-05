import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import chromadb
from rank_bm25 import BM25Okapi

from henet_kb.index.embeddings import Embedder
from henet_kb.index.text import tokenize
from henet_kb.ingest.chunking import Chunk

if TYPE_CHECKING:
    from henet_kb.config import Settings

RRF_K = 60


@dataclass
class SearchHit:
    chunk_id: str
    url: str
    title: str
    section: str
    chunk_index: int
    text: str
    score: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class HybridIndex:
    """Chroma for the vectors, BM25 over the same chunks, fused with reciprocal rank."""

    def __init__(self, collection: chromadb.Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._records: dict[str, dict] = {}

    @classmethod
    def persistent(cls, path: str, name: str, embedder: Embedder) -> "HybridIndex":
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        return cls(collection, embedder)

    @classmethod
    def remote(cls, host: str, port: int, name: str, embedder: Embedder) -> "HybridIndex":
        client = chromadb.HttpClient(host=host, port=port)
        collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        return cls(collection, embedder)

    @classmethod
    def from_settings(cls, settings: "Settings", embedder: Embedder) -> "HybridIndex":
        if settings.chroma_host:
            return cls.remote(
                settings.chroma_host, settings.chroma_port, settings.chroma_collection, embedder
            )
        return cls.persistent(settings.chroma_path, settings.chroma_collection, embedder)

    @classmethod
    def ephemeral(cls, embedder: Embedder, name: str | None = None) -> "HybridIndex":
        # Ephemeral clients share one database per process, so each caller gets its own name.
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name or f"index-{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"}
        )
        return cls(collection, embedder)

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=self.embedder.embed([chunk.text for chunk in batch]),
            )
        self._bm25 = None
        return len(chunks)

    def delete_url(self, url: str) -> None:
        self.collection.delete(where={"url": url})
        self._bm25 = None

    def replace_document_chunks(self, url: str, chunks: list[Chunk]) -> int:
        self.delete_url(url)
        return self.add_chunks(chunks)

    def clear(self) -> None:
        existing = self.collection.get(include=[])
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        self._bm25 = None

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        rows = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = list(rows["ids"])
        self._records = {
            chunk_id: {"text": text, **metadata}
            for chunk_id, text, metadata in zip(
                rows["ids"], rows["documents"] or [], rows["metadatas"] or [], strict=True
            )
        }
        corpus = [tokenize(self._records[chunk_id]["text"]) for chunk_id in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _vector_ranking(self, query: str, limit: int) -> list[str]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(limit, self.count()),
            include=["documents", "metadatas"],
        )
        ids = result["ids"][0]
        for chunk_id, text, metadata in zip(
            ids, result["documents"][0], result["metadatas"][0], strict=True
        ):
            self._records.setdefault(chunk_id, {"text": text, **metadata})
        return ids

    def _keyword_ranking(self, query: str, limit: int) -> list[str]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self._bm25_ids[index] for index in ranked[:limit] if scores[index] > 0]

    def search(self, query: str, top_k: int = 6) -> list[SearchHit]:
        candidates = max(top_k * 3, 10)
        fused: dict[str, float] = {}
        for ranking in (
            self._vector_ranking(query, candidates),
            self._keyword_ranking(query, candidates),
        ):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        hits: list[SearchHit] = []
        for chunk_id, score in ordered:
            record = self._records[chunk_id]
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    url=str(record["url"]),
                    title=str(record["title"]),
                    section=str(record.get("section", "")),
                    chunk_index=int(record["chunk_index"]),
                    text=str(record["text"]),
                    score=round(score, 6),
                )
            )
        return hits
