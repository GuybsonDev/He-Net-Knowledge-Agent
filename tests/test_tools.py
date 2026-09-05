import json

import pytest

from henet_kb.llm import Message, ToolCall
from henet_kb.tools import SEARCH_TOOL, SearchTool, run_tool_loop
from tests.fakes import FakeProvider, HashEmbedder, text_response, tool_response
from tests.test_hybrid import build_index


def test_search_tool_schema_is_strict():
    schema = SEARCH_TOOL.parameters
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == {"query", "top_k"}


def test_search_tool_returns_hits_and_records_them():
    tool = SearchTool(build_index())

    result = tool({"query": "canais de esporte", "top_k": 2})

    assert result["query"] == "canais de esporte"
    assert result["results"][0]["url"] == "https://henet.com.br/tv"
    assert {"url", "title", "section", "text", "score"} <= set(result["results"][0])
    assert len(tool.hits) == len(result["results"])


def test_search_tool_clamps_top_k_and_rejects_bad_input():
    tool = SearchTool(build_index())

    assert len(tool({"query": "plano", "top_k": 999})["results"]) <= 20
    with pytest.raises(ValueError):
        tool({"query": "   ", "top_k": 3})
    with pytest.raises(ValueError):
        tool({"query": "plano", "top_k": "3"})


def test_loop_executes_tools_until_the_model_stops():
    def script(messages, tools):
        tool_messages = [m for m in messages if m.role == "tool"]
        if not tool_messages:
            return tool_response(
                ToolCall("c1", "search_knowledge_base", {"query": "fibra", "top_k": 2})
            )
        payload = json.loads(tool_messages[-1].content)
        return text_response(f"Encontrei {len(payload['results'])} trechos.")

    provider = FakeProvider(script)
    deltas: list[str] = []
    tool = SearchTool(build_index())

    result = run_tool_loop(
        provider,
        [Message.user("fibra?")],
        tools=[SEARCH_TOOL],
        handlers={SEARCH_TOOL.name: tool},
        on_text=deltas.append,
    )

    assert result.rounds == 2
    assert result.text.startswith("Encontrei 2 trechos")
    assert "".join(deltas).strip() == result.text
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    assert result.tool_errors == []
    assert result.usage.input_tokens == 20


def test_tool_errors_are_returned_to_the_model_not_raised():
    def script(messages, tools):
        tool_messages = [m for m in messages if m.role == "tool"]
        if not tool_messages:
            return tool_response(
                ToolCall("c1", "search_knowledge_base", {"query": "", "top_k": 2}),
                ToolCall("c2", "does_not_exist", {}),
            )
        assert all(m.is_error for m in tool_messages)
        return text_response("Nao consegui buscar.")

    result = run_tool_loop(
        FakeProvider(script),
        [Message.user("?")],
        tools=[SEARCH_TOOL],
        handlers={SEARCH_TOOL.name: SearchTool(build_index())},
    )

    assert len(result.tool_errors) == 2
    assert "ValueError" in result.tool_errors[0]
    assert "unknown tool" in result.tool_errors[1]
    assert result.text == "Nao consegui buscar."


def test_loop_withholds_tools_after_max_rounds():
    def script(messages, tools):
        if tools:
            return tool_response(
                ToolCall("c", "search_knowledge_base", {"query": "tv", "top_k": 1})
            )
        return text_response("Resposta final.")

    provider = FakeProvider(script)
    result = run_tool_loop(
        provider,
        [Message.user("?")],
        tools=[SEARCH_TOOL],
        handlers={SEARCH_TOOL.name: SearchTool(build_index())},
        max_rounds=2,
    )

    assert result.rounds == 3
    assert result.text == "Resposta final."
    assert provider.calls[-1][1] is None


def test_hash_embedder_is_deterministic():
    embedder = HashEmbedder()
    assert embedder.embed(["fibra optica"]) == embedder.embed(["fibra optica"])
