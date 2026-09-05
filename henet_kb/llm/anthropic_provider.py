import json
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic

from henet_kb.llm.base import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage
from henet_kb.llm.pricing import estimate_cost


class AnthropicProvider:
    """Messages API with strict tool use and streaming through messages.stream."""

    def __init__(
        self, api_key: str, model: str = "claude-opus-5", client: Anthropic | None = None
    ) -> None:
        self.model = model
        self.client = client or Anthropic(api_key=api_key)

    @staticmethod
    def _to_tool(spec: ToolSpec) -> dict[str, Any]:
        schema = dict(spec.parameters)
        schema.setdefault("additionalProperties", False)
        return {
            "name": spec.name,
            "description": spec.description,
            "strict": True,
            "input_schema": schema,
        }

    @staticmethod
    def _to_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Split out the system prompt and fold tool results into user turns."""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": message.is_error,
                }
                if (
                    converted
                    and converted[-1]["role"] == "user"
                    and isinstance(converted[-1]["content"], list)
                ):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
                continue
            if message.role == "assistant" and message.tool_calls:
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.extend(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    for call in message.tool_calls
                )
                converted.append({"role": "assistant", "content": content})
                continue
            converted.append({"role": message.role, "content": message.content})
        return "\n\n".join(system_parts), converted

    def _request(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        system, converted = self._to_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": converted,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [self._to_tool(spec) for spec in tools]
        return request

    def _usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage()
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        input_tokens += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        input_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return Usage(
            input_tokens, output_tokens, estimate_cost(self.model, input_tokens, output_tokens)
        )

    def _to_response(self, message: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = (
                    block.input if isinstance(block.input, dict) else json.loads(block.input)
                )
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=self._usage(message.usage),
            stop_reason=message.stop_reason or "",
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Sampling parameters are not accepted alongside adaptive thinking, so
        # temperature is intentionally not forwarded.
        message = self.client.messages.create(**self._request(messages, tools, max_tokens))
        return self._to_response(message)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        with self.client.messages.stream(**self._request(messages, tools, max_tokens)) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield StreamEvent(kind="text_delta", text=event.delta.text)
            final = stream.get_final_message()
        yield StreamEvent(kind="final", response=self._to_response(final))
