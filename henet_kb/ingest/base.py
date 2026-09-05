from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Document:
    url: str
    title: str
    sections: list[Section] = field(default_factory=list)
    source: str = ""
    modified: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)


class Source(Protocol):
    """Anything that yields documents for the index."""

    name: str

    def documents(self) -> Iterator[Document]: ...
