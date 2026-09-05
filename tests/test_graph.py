import json

from henet_kb.agent import AgentService
from henet_kb.agent.graph import _parse_grade
from henet_kb.config import Settings
from henet_kb.llm import Message, ToolCall
from tests.fakes import FakeProvider, text_response, tool_response
from tests.test_hybrid import build_index


def settings(**overrides) -> Settings:
    return Settings(openai_api_key="x", top_k=3, max_rewrites=2, _env_file=None, **overrides)


def role_of(messages: list[Message]) -> str:
    system = messages[0].content if messages and messages[0].role == "system" else ""
    if "retrieved excerpts can answer" in system:
        return "grade"
    if "rewrite search queries" in system:
        return "rewrite"
    return "answer"


def test_parse_grade_accepts_json_and_falls_back_to_keywords():
    assert _parse_grade('{"verdict": "good", "reason": "covers it"}') == ("good", "covers it")
    assert _parse_grade('Sure: {"verdict": "weak", "reason": "off topic"}')[0] == "weak"
    assert _parse_grade("weak, nothing relevant")[0] == "weak"
    assert _parse_grade("good enough")[0] == "good"


def test_good_grade_goes_straight_to_answer_with_sources():
    def script(messages, tools):
        kind = role_of(messages)
        if kind == "grade":
            assert "Knowledge base excerpts" in messages[1].content
            return text_response(json.dumps({"verdict": "good", "reason": "ok"}))
        return text_response("A He-Net oferece fibra de 500 mega.")

    service = AgentService(FakeProvider(script), build_index(), settings())

    result = service.ask("qual a velocidade da fibra?", thread_id="t1")

    assert result.answer == "A He-Net oferece fibra de 500 mega."
    assert result.grade == "good"
    assert result.rewrites == 0
    assert result.sources[0]["url"] == "https://henet.com.br/internet"
    assert result.input_tokens == 20
    assert result.cost_usd == 0.002
    assert result.thread_id == "t1"


def test_weak_grade_rewrites_at_most_twice_then_answers():
    def script(messages, tools):
        kind = role_of(messages)
        if kind == "grade":
            return text_response('{"verdict": "weak", "reason": "off topic"}')
        if kind == "rewrite":
            return text_response("planos de fibra He-Net")
        return text_response("Nao encontrei essa informacao.")

    provider = FakeProvider(script)
    service = AgentService(provider, build_index(), settings())

    result = service.ask("xyz", thread_id="t2")

    kinds = [role_of(messages) for messages, _ in provider.calls]
    assert kinds == ["grade", "rewrite", "grade", "rewrite", "grade", "answer"]
    assert result.rewrites == 2
    assert result.query == "planos de fibra He-Net"
    assert result.answer == "Nao encontrei essa informacao."


def test_answer_node_can_call_the_search_tool_and_adds_its_sources():
    def script(messages, tools):
        kind = role_of(messages)
        if kind == "grade":
            return text_response('{"verdict": "good", "reason": "ok"}')
        if not any(m.role == "tool" for m in messages):
            return tool_response(
                ToolCall("c1", "search_knowledge_base", {"query": "WhatsApp", "top_k": 1})
            )
        return text_response("Atendimento pelo WhatsApp.")

    service = AgentService(FakeProvider(script), build_index(), settings())

    result = service.ask("qual a velocidade da fibra e como falo com voces?", thread_id="t3")

    urls = {source["url"] for source in result.sources}
    assert "https://henet.com.br/fale-conosco" in urls
    assert result.tool_rounds == 2


def test_stream_emits_events_in_order():
    def script(messages, tools):
        if role_of(messages) == "grade":
            return text_response('{"verdict": "good", "reason": "ok"}')
        return text_response("Fibra de 500 mega.")

    service = AgentService(FakeProvider(script), build_index(), settings())

    events = list(service.stream("fibra?", thread_id="t4"))

    types = [event["type"] for event in events]
    assert types[0] == "start"
    assert types[-1] == "done"
    stages = [event["stage"] for event in events if event["type"] == "status"]
    assert stages == ["searching", "grading", "answering"]
    assert (
        "".join(e["text"] for e in events if e["type"] == "delta").strip() == "Fibra de 500 mega."
    )
    assert types.index("sources") < types.index("done")
    done = events[-1]
    assert done["answer"] == "Fibra de 500 mega."
    assert done["input_tokens"] == 20
    assert done["sources"][0]["url"] == "https://henet.com.br/internet"


def test_stream_reports_provider_failures_as_error_event():
    def script(messages, tools):
        raise RuntimeError("provider down")

    service = AgentService(FakeProvider(script), build_index(), settings())

    events = list(service.stream("fibra?", thread_id="t5"))

    assert events[-1]["type"] == "error"
    assert "provider down" in events[-1]["message"]


def test_sqlite_checkpointer_persists_thread_state(tmp_path):
    def script(messages, tools):
        if role_of(messages) == "grade":
            return text_response('{"verdict": "good", "reason": "ok"}')
        return text_response("ok")

    service = AgentService(
        FakeProvider(script), build_index(), settings(), checkpoint_path=str(tmp_path / "cp.sqlite")
    )
    service.ask("fibra?", thread_id="persisted")

    snapshot = service.graph.get_state({"configurable": {"thread_id": "persisted"}})
    assert snapshot.values["answer"] == "ok"
    assert (tmp_path / "cp.sqlite").exists()
