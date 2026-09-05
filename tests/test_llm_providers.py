import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from henet_kb.config import Settings
from henet_kb.llm import Message, ToolCall, ToolSpec, estimate_cost, make_provider
from henet_kb.llm.anthropic_provider import AnthropicProvider
from henet_kb.llm.openai_provider import OpenAIProvider

SEARCH_TOOL = ToolSpec(
    name="search_knowledge_base",
    description="Search the index.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
)

CONVERSATION = [
    Message.system("You answer about He-Net."),
    Message.user("Quais planos existem?"),
    Message.assistant(
        tool_calls=[ToolCall("call_1", "search_knowledge_base", {"query": "planos", "top_k": 3})]
    ),
    Message.tool_result(ToolCall("call_1", "search_knowledge_base", {}), '{"hits": []}'),
]


def test_estimate_cost_handles_snapshots_and_unknown_models():
    assert estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000) == 2.0
    assert estimate_cost("gpt-4.1-mini-2025-04-14", 500_000, 0) == 0.2
    assert estimate_cost("some-model", 10, 10) == 0.0


class TestOpenAI:
    def make(self) -> tuple[OpenAIProvider, MagicMock]:
        client = MagicMock()
        return OpenAIProvider("key", "gpt-4.1-mini", client=client), client

    def test_complete_maps_tool_calls_and_usage(self):
        provider, client = self.make()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_9",
                                function=SimpleNamespace(
                                    name="search_knowledge_base",
                                    arguments='{"query": "tv", "top_k": 2}',
                                ),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )

        response = provider.complete(CONVERSATION, tools=[SEARCH_TOOL])

        assert response.wants_tools
        assert response.tool_calls[0] == ToolCall(
            "call_9", "search_knowledge_base", {"query": "tv", "top_k": 2}
        )
        assert response.usage.input_tokens == 100
        assert response.usage.cost_usd == estimate_cost("gpt-4.1-mini", 100, 20)

        request = client.chat.completions.create.call_args.kwargs
        assert request["tools"][0]["function"]["strict"] is True
        assert request["messages"][0] == {"role": "system", "content": "You answer about He-Net."}
        assert (
            request["messages"][2]["tool_calls"][0]["function"]["arguments"]
            == '{"query": "planos", "top_k": 3}'
        )
        assert request["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"hits": []}',
        }

    def test_stream_accumulates_text_and_tool_arguments(self):
        provider, client = self.make()

        def chunk(content=None, tool=None, finish=None, usage=None):
            delta = SimpleNamespace(content=content, tool_calls=tool)
            choices = [SimpleNamespace(delta=delta, finish_reason=finish)] if usage is None else []
            return SimpleNamespace(choices=choices, usage=usage)

        def tool_delta(index, id=None, name=None, arguments=None):
            return SimpleNamespace(
                index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
            )

        client.chat.completions.create.return_value = iter(
            [
                chunk(content="Ola"),
                chunk(content=", tudo bem"),
                chunk(
                    tool=[
                        tool_delta(
                            0, id="call_1", name="search_knowledge_base", arguments='{"query": '
                        )
                    ]
                ),
                chunk(tool=[tool_delta(0, arguments='"fibra", "top_k": 3}')]),
                chunk(finish="tool_calls"),
                chunk(usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10)),
            ]
        )

        events = list(provider.stream(CONVERSATION, tools=[SEARCH_TOOL]))

        deltas = [event.text for event in events if event.kind == "text_delta"]
        assert deltas == ["Ola", ", tudo bem"]
        final = events[-1].response
        assert final.text == "Ola, tudo bem"
        assert final.tool_calls == [
            ToolCall("call_1", "search_knowledge_base", {"query": "fibra", "top_k": 3})
        ]
        assert final.usage.output_tokens == 10
        assert client.chat.completions.create.call_args.kwargs["stream_options"] == {
            "include_usage": True
        }

    def test_invalid_json_arguments_are_kept_raw(self):
        assert OpenAIProvider._parse_arguments("{not json") == {"_raw": "{not json"}


class TestAnthropic:
    def make(self) -> tuple[AnthropicProvider, MagicMock]:
        client = MagicMock()
        return AnthropicProvider("key", "claude-opus-5", client=client), client

    def test_messages_are_converted_to_the_anthropic_shape(self):
        provider, _ = self.make()

        system, messages = provider._to_messages(CONVERSATION)

        assert system == "You answer about He-Net."
        assert messages[0] == {"role": "user", "content": "Quais planos existem?"}
        assert messages[1]["content"][0] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search_knowledge_base",
            "input": {"query": "planos", "top_k": 3},
        }
        assert messages[2] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"hits": []}',
                    "is_error": False,
                }
            ],
        }

    def test_complete_uses_strict_tools_and_reads_tool_use_blocks(self):
        provider, client = self.make()
        client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Vou procurar."),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="search_knowledge_base",
                    input={"query": "tv", "top_k": 2},
                ),
            ],
            usage=SimpleNamespace(input_tokens=80, output_tokens=15, cache_read_input_tokens=20),
            stop_reason="tool_use",
        )

        response = provider.complete(CONVERSATION, tools=[SEARCH_TOOL])

        assert response.text == "Vou procurar."
        assert response.tool_calls == [
            ToolCall("toolu_1", "search_knowledge_base", {"query": "tv", "top_k": 2})
        ]
        assert response.usage.input_tokens == 100
        assert response.stop_reason == "tool_use"
        request = client.messages.create.call_args.kwargs
        assert request["tools"][0]["strict"] is True
        assert request["tools"][0]["input_schema"]["additionalProperties"] is False
        assert request["system"] == "You answer about He-Net."
        assert "temperature" not in request

    def test_stream_yields_text_deltas_then_final(self):
        provider, client = self.make()
        events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="A He-Net "),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="oferece fibra."),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        stream = MagicMock()
        stream.__enter__.return_value = stream
        stream.__iter__.return_value = iter(events)
        stream.get_final_message.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="A He-Net oferece fibra.")],
            usage=SimpleNamespace(input_tokens=30, output_tokens=8),
            stop_reason="end_turn",
        )
        client.messages.stream.return_value = stream

        received = list(provider.stream([Message.user("O que a He-Net oferece?")]))

        assert [event.text for event in received[:-1]] == ["A He-Net ", "oferece fibra."]
        assert received[-1].response.text == "A He-Net oferece fibra."
        assert received[-1].response.usage.cost_usd == estimate_cost("claude-opus-5", 30, 8)
        assert not received[-1].response.wants_tools


def test_factory_picks_provider_from_settings():
    openai_provider = make_provider(Settings(openai_api_key="x", _env_file=None))
    assert isinstance(openai_provider, OpenAIProvider)

    anthropic_provider = make_provider(
        Settings(llm_provider="anthropic", anthropic_api_key="y", _env_file=None)
    )
    assert isinstance(anthropic_provider, AnthropicProvider)
    assert anthropic_provider.model == "claude-opus-5"

    with pytest.raises(RuntimeError):
        make_provider(Settings(llm_provider="anthropic", _env_file=None))


def test_tool_call_arguments_round_trip_as_json():
    call = ToolCall("c", "search_knowledge_base", {"query": "x", "top_k": 1})
    assert json.loads(json.dumps(call.arguments)) == call.arguments
