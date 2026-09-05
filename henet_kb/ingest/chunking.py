import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from henet_kb.ingest.base import Document

SENTENCE_RE = re.compile(r"(?<=[.!?:;])\s+(?=[A-ZÀ-Ú0-9\"“(])")


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    section: str
    chunk_index: int
    text: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }


@lru_cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _split_sentences(text: str) -> list[str]:
    return [part for part in SENTENCE_RE.split(text) if part.strip()]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _encoder().encode(text)
    return [
        _encoder().decode(tokens[start : start + max_tokens])
        for start in range(0, len(tokens), max_tokens)
    ]


def _pack(sentences: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack sentences greedily and carry a short tail into the next chunk."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def emit() -> None:
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        pieces = [sentence]
        if count_tokens(sentence) > max_tokens:
            pieces = _hard_split(sentence, max_tokens)
        for piece in pieces:
            piece_tokens = count_tokens(piece)
            if current and current_tokens + piece_tokens > max_tokens:
                emit()
                tail: list[str] = []
                tail_tokens = 0
                for previous in reversed(current):
                    previous_tokens = count_tokens(previous)
                    if tail_tokens + previous_tokens > overlap_tokens:
                        break
                    tail.insert(0, previous)
                    tail_tokens += previous_tokens
                current = tail
                current_tokens = tail_tokens
            current.append(piece)
            current_tokens += piece_tokens
    emit()
    return chunks


def chunk_document(
    document: Document, max_tokens: int = 400, overlap_tokens: int = 40
) -> list[Chunk]:
    """Split a document into chunks that never cross a section boundary."""
    chunks: list[Chunk] = []
    for section in document.sections:
        for text in _pack(_split_sentences(section.text), max_tokens, overlap_tokens):
            index = len(chunks)
            digest = hashlib.sha1(f"{document.url}|{index}|{text}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest,
                    url=document.url,
                    title=document.title,
                    section=section.heading,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks
