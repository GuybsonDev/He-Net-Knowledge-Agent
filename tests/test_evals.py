from pathlib import Path

from evals.run import evaluate, load_questions, normalize, render_table
from henet_kb.agent import AgentService
from henet_kb.config import Settings
from tests.fakes import FakeProvider
from tests.test_api import script
from tests.test_hybrid import build_index


def test_questions_file_has_ten_complete_rows():
    questions = load_questions(Path("evals/questions.jsonl"))

    assert len(questions) == 10
    assert all({"id", "question", "expected_answer", "expected_url"} <= set(q) for q in questions)
    assert len({q["id"] for q in questions}) == 10


def test_evaluate_checks_expected_url_among_sources():
    settings = Settings(openai_api_key="x", top_k=3, _env_file=None)
    service = AgentService(FakeProvider(script), build_index(), settings)
    questions = [
        {"id": "a", "question": "fibra?", "expected_url": "https://henet.com.br/internet/"},
        {"id": "b", "question": "fibra?", "expected_url": "https://henet.com.br/nao-existe"},
    ]

    rows = evaluate(service, questions)
    table = render_table(rows)

    assert [row.source_hit for row in rows] == [True, False]
    assert "Source accuracy: 1/2 (50%)" in table
    assert normalize("https://X.com/a/") == "https://x.com/a"
