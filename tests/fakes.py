import hashlib
import math
from collections.abc import Callable, Iterator

from henet_kb.index.text import tokenize
from henet_kb.llm.base import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage


class HashEmbedder:
    """Deterministic bag of words embedding so vector search can be tested offline."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
                vector[slot] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


Script = Callable[[list[Message], list[ToolSpec] | None], LLMResponse]


def text_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage(input_tokens, output_tokens, 0.001))


def tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        text="", tool_calls=list(calls), usage=Usage(10, 5, 0.001), stop_reason="tool_use"
    )


class FakeProvider:
    """Scripted model. The script decides the reply from the messages it receives."""

    model = "fake-model"

    def __init__(self, script: Script) -> None:
        self.script = script
        self.calls: list[tuple[list[Message], list[ToolSpec] | None]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        return self.script(messages, tools)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        response = self.complete(messages, tools, max_tokens, temperature)
        for word in response.text.split(" "):
            if word:
                yield StreamEvent(kind="text_delta", text=word + " ")
        yield StreamEvent(kind="final", response=response)
