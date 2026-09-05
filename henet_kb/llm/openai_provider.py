import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from henet_kb.llm.base import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage
from henet_kb.llm.pricing import estimate_cost


class OpenAIProvider:
    """Chat Completions with function calling, strict schemas and streaming."""

    def __init__(
        self, api_key: str, model: str = "gpt-4.1-mini", client: OpenAI | None = None
    ) -> None:
        self.model = model
        self.client = client or OpenAI(api_key=api_key)

    @staticmethod
    def _to_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "strict": True,
            },
        }

    @staticmethod
    def _to_message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
            if not message.content:
                payload["content"] = None
        return payload

    def _request(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_message(message) for message in messages],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = [self._to_tool(spec) for spec in tools]
            request["tool_choice"] = "auto"
        return request

    def _usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage()
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return Usage(
            input_tokens, output_tokens, estimate_cost(self.model, input_tokens, output_tokens)
        )

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        completion = self.client.chat.completions.create(
            **self._request(messages, tools, max_tokens, temperature)
        )
        choice = completion.choices[0]
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=self._parse_arguments(call.function.arguments),
            )
            for call in (choice.message.tool_calls or [])
        ]
        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            usage=self._usage(completion.usage),
            stop_reason=choice.finish_reason or "",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        request = self._request(messages, tools, max_tokens, temperature)
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        pending: dict[int, dict[str, str]] = {}
        usage = Usage()
        stop_reason = ""
        for chunk in self.client.chat.completions.create(**request):
            if chunk.usage is not None:
                usage = self._usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                stop_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                text_parts.append(delta.content)
                yield StreamEvent(kind="text_delta", text=delta.content)
            for call in delta.tool_calls or []:
                slot = pending.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function is not None:
                    slot["name"] += call.function.name or ""
                    slot["arguments"] += call.function.arguments or ""

        tool_calls = [
            ToolCall(
                id=slot["id"], name=slot["name"], arguments=self._parse_arguments(slot["arguments"])
            )
            for _, slot in sorted(pending.items())
        ]
        yield StreamEvent(
            kind="final",
            response=LLMResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                usage=usage,
                stop_reason=stop_reason,
            ),
        )
