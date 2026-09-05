import json
from typing import Any


def encode_event(event: str, data: dict[str, Any]) -> str:
    """One SSE event. Data is always a single JSON line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class SSEParser:
    """Incremental SSE parser. Handles events split across chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[tuple[str, dict[str, Any]]]:
        self._buffer += chunk.replace("\r\n", "\n")
        events: list[tuple[str, dict[str, Any]]] = []
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            parsed = self._parse_block(block)
            if parsed is not None:
                events.append(parsed)
        return events

    @staticmethod
    def _parse_block(block: str) -> tuple[str, dict[str, Any]] | None:
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data_lines.append(value)
        if not data_lines:
            return None
        return event, json.loads("\n".join(data_lines))
