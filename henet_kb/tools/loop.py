import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from henet_kb.llm.base import LLMProvider, Message, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass
class ToolLoopResult:
    text: str
    messages: list[Message]
    usage: Usage = field(default_factory=Usage)
    rounds: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)


def _execute(call: ToolCall, handlers: dict[str, ToolHandler]) -> tuple[str, bool]:
    handler = handlers.get(call.name)
    if handler is None:
        return json.dumps({"error": f"unknown tool: {call.name}"}), True
    try:
        result = handler(call.arguments)
    except Exception as exc:
        log.warning("tool %s failed: %s", call.name, exc)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
    return json.dumps(result, ensure_ascii=False, default=str), False


def run_tool_loop(
    provider: LLMProvider,
    messages: list[Message],
    tools: list[ToolSpec],
    handlers: dict[str, ToolHandler],
    max_rounds: int = 4,
    max_tokens: int = 1024,
    on_text: Callable[[str], None] | None = None,
) -> ToolLoopResult:
    """Call the model, run whatever tools it asks for, feed the results back, repeat.

    Stops when the model answers without tool calls. Past max_rounds the tools are no
    longer offered, so the model has to answer with what it already has.
    """
    history = list(messages)
    result = ToolLoopResult(text="", messages=history)

    while True:
        offer_tools = tools if result.rounds < max_rounds else None
        response = None
        for event in provider.stream(history, tools=offer_tools, max_tokens=max_tokens):
            if event.kind == "text_delta" and on_text is not None and event.text:
                on_text(event.text)
            elif event.kind == "final":
                response = event.response
        if response is None:
            raise RuntimeError("provider stream ended without a final response")

        result.usage = result.usage + response.usage
        result.rounds += 1
        history.append(Message.assistant(response.text, response.tool_calls))

        if not response.wants_tools:
            result.text = response.text
            return result

        for call in response.tool_calls:
            result.tool_calls.append(call)
            content, is_error = _execute(call, handlers)
            if is_error:
                result.tool_errors.append(content)
            history.append(Message.tool_result(call, content, is_error=is_error))
