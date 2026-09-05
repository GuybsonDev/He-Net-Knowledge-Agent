"""Source accuracy eval: is the expected URL among the cited sources?

Usage: python -m evals.run [--questions evals/questions.jsonl] [--output evals/results.json]
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from henet_kb.agent.service import AgentService


@dataclass
class EvalRow:
    id: str
    question: str
    expected_url: str
    cited_urls: list[str]
    source_hit: bool
    rewrites: int
    cost_usd: float
    seconds: float
    answer: str


def normalize(url: str) -> str:
    return url.rstrip("/").lower()


def load_questions(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def evaluate(service: AgentService, questions: list[dict[str, str]]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for item in questions:
        started = time.perf_counter()
        result = service.ask(item["question"], thread_id=f"eval-{item['id']}")
        cited = [source["url"] for source in result.sources]
        rows.append(
            EvalRow(
                id=item["id"],
                question=item["question"],
                expected_url=item["expected_url"],
                cited_urls=cited,
                source_hit=normalize(item["expected_url"]) in {normalize(url) for url in cited},
                rewrites=result.rewrites,
                cost_usd=result.cost_usd,
                seconds=round(time.perf_counter() - started, 1),
                answer=result.answer,
            )
        )
    return rows


def render_table(rows: list[EvalRow]) -> str:
    lines = [
        "| id | source hit | rewrites | cost (USD) | seconds | question |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = "yes" if row.source_hit else "no"
        lines.append(
            f"| {row.id} | {mark} | {row.rewrites} | {row.cost_usd:.4f} | {row.seconds} | "
            f"{row.question} |"
        )
    hits = sum(row.source_hit for row in rows)
    total_cost = sum(row.cost_usd for row in rows)
    lines.append("")
    lines.append(
        f"Source accuracy: {hits}/{len(rows)} ({100 * hits / max(len(rows), 1):.0f}%), "
        f"total cost USD {total_cost:.4f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--output", default="evals/results.json")
    args = parser.parse_args(argv)

    from henet_kb.api.app import default_service
    from henet_kb.config import get_settings

    service = default_service(get_settings())
    rows = evaluate(service, load_questions(Path(args.questions)))
    Path(args.output).write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
