import pytest
from fastapi.testclient import TestClient

from henet_kb.agent import AgentService
from henet_kb.api import SSEParser, create_app, encode_event
from henet_kb.config import Settings
from henet_kb.ingest.base import Document, Section
from tests.fakes import FakeProvider, text_response
from tests.test_graph import role_of
from tests.test_hybrid import build_index


def script(messages, tools):
    if role_of(messages) == "grade":
        return text_response('{"verdict": "good", "reason": "ok"}')
    return text_response("Fibra de 500 mega.")


@pytest.fixture
def client() -> TestClient:
    settings = Settings(
        openai_api_key="x",
        ingest_token="secret",
        cors_origins="http://localhost:3000",
        top_k=3,
        _env_file=None,
    )
    service = AgentService(FakeProvider(script), build_index(), settings)

    class FakeSource:
        name = "fake"

        def documents(self):
            yield Document(
                url="https://henet.com.br/novo",
                title="Novo",
                sections=[Section("", "Pagina nova com combo de internet e TV.")],
            )

    app = create_app(settings=settings, service=service, source_factory=lambda *_: FakeSource())
    return TestClient(app)


def test_sse_parser_handles_split_chunks_and_multiline_data():
    parser = SSEParser()
    raw = encode_event("delta", {"text": "olá"}) + 'event: done\ndata: {"a":\ndata: 1}\n\n'
    first, second = raw[:10], raw[10:]

    assert parser.feed(first) == []
    events = parser.feed(second)

    assert events == [("delta", {"text": "olá"}), ("done", {"a": 1})]


def test_health_reports_index_and_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-model"
    assert body["chunks"] == 3


def test_ask_returns_answer_sources_and_cost(client):
    response = client.post("/ask", json={"question": "qual a velocidade da fibra?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fibra de 500 mega."
    assert body["sources"][0]["url"] == "https://henet.com.br/internet"
    assert body["input_tokens"] == 20
    assert body["cost_usd"] == 0.002
    assert body["thread_id"]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_stream_emits_sse_events(client):
    with client.stream("POST", "/ask/stream", json={"question": "fibra?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        parser = SSEParser()
        events = []
        for chunk in response.iter_text():
            events.extend(parser.feed(chunk))

    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    assert [data["stage"] for name, data in events if name == "status"] == [
        "searching",
        "grading",
        "answering",
    ]
    assert (
        "".join(data["text"] for name, data in events if name == "delta").strip()
        == "Fibra de 500 mega."
    )
    assert events[-1][1]["cost_usd"] == 0.002
    assert any(name == "sources" for name in names)


def test_cors_allows_the_frontend_origin(client):
    response = client.options(
        "/ask",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_ingest_requires_the_token(client):
    assert client.post("/ingest", json={}).status_code == 401
    assert (
        client.post("/ingest", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
    )


def test_ingest_runs_the_source_and_reports_counts(client):
    response = client.post(
        "/ingest", json={"source": "sitemap"}, headers={"Authorization": "Bearer secret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 1
    assert body["chunks"] == 1
    assert body["total_chunks"] == 4


def test_ingest_is_hidden_when_no_token_is_configured():
    settings = Settings(openai_api_key="x", _env_file=None)
    service = AgentService(FakeProvider(script), build_index(), settings)
    client = TestClient(create_app(settings=settings, service=service))

    assert client.post("/ingest", json={}, headers={"Authorization": "Bearer x"}).status_code == 404
