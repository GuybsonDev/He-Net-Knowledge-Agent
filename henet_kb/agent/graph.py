import json
import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from henet_kb.agent.prompts import ANSWER_SYSTEM, GRADE_SYSTEM, REWRITE_SYSTEM, format_context
from henet_kb.llm.base import LLMProvider, Message, Usage
from henet_kb.tools.loop import run_tool_loop
from henet_kb.tools.search import SEARCH_TOOL, SearchTool


class AgentState(TypedDict, total=False):
    question: str
    query: str
    hits: list[dict[str, Any]]
    grade: Literal["good", "weak"]
    grade_reason: str
    rewrites: int
    answer: str
    sources: list[dict[str, str]]
    tool_rounds: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]


def _usage_update(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def _emit(payload: dict[str, Any]) -> None:
    get_stream_writer()(payload)


def _parse_grade(text: str) -> tuple[Literal["good", "weak"], str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = "good" if str(data.get("verdict", "")).lower() == "good" else "weak"
            return verdict, str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    return ("good" if "good" in text.lower() else "weak"), text.strip()[:200]


def _collect_sources(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for hit in hits:
        sources.setdefault(hit["url"], {"url": hit["url"], "title": hit["title"]})
    return list(sources.values())


def build_graph(
    provider: LLMProvider,
    search_tool: SearchTool,
    top_k: int = 6,
    max_rewrites: int = 2,
    checkpointer: BaseCheckpointSaver | None = None,
    max_tool_rounds: int = 3,
) -> CompiledStateGraph:
    def retrieve(state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        _emit({"type": "status", "stage": "searching", "query": query})
        result = search_tool({"query": query, "top_k": top_k})
        return {"query": query, "hits": result["results"]}

    def grade(state: AgentState) -> dict[str, Any]:
        _emit({"type": "status", "stage": "grading"})
        prompt = (
            f"Question: {state['question']}\n\n{format_context(state['hits'])}\n\n"
            "Can these excerpts answer the question?"
        )
        response = provider.complete(
            [Message.system(GRADE_SYSTEM), Message.user(prompt)], max_tokens=200
        )
        verdict, reason = _parse_grade(response.text)
        return {"grade": verdict, "grade_reason": reason, **_usage_update(response.usage)}

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        _emit({"type": "status", "stage": "rewriting"})
        prompt = (
            f"Question: {state['question']}\nFailed query: {state['query']}\n"
            f"Why it failed: {state.get('grade_reason', '')}"
        )
        response = provider.complete(
            [Message.system(REWRITE_SYSTEM), Message.user(prompt)], max_tokens=100
        )
        new_query = response.text.strip().strip('"') or state["question"]
        return {
            "query": new_query,
            "rewrites": state.get("rewrites", 0) + 1,
            **_usage_update(response.usage),
        }

    def answer(state: AgentState) -> dict[str, Any]:
        _emit({"type": "status", "stage": "answering"})
        seen_before = len(search_tool.hits)
        messages = [
            Message.system(f"{ANSWER_SYSTEM}\n\n{format_context(state['hits'])}"),
            Message.user(state["question"]),
        ]
        result = run_tool_loop(
            provider,
            messages,
            tools=[SEARCH_TOOL],
            handlers={SEARCH_TOOL.name: search_tool},
            max_rounds=max_tool_rounds,
            max_tokens=1024,
            on_text=lambda text: _emit({"type": "delta", "text": text}),
        )
        extra_hits = [hit.to_dict() for hit in search_tool.hits[seen_before:]]
        sources = _collect_sources(state["hits"] + extra_hits)
        _emit({"type": "sources", "sources": sources})
        return {
            "answer": result.text,
            "sources": sources,
            "tool_rounds": result.rounds,
            **_usage_update(result.usage),
        }

    def after_grade(state: AgentState) -> Literal["answer", "rewrite_query"]:
        if state.get("grade") == "good" or state.get("rewrites", 0) >= max_rewrites:
            return "answer"
        return "rewrite_query"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)
