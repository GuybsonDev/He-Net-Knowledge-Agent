import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from henet_kb.agent.graph import AgentState, build_graph
from henet_kb.config import Settings
from henet_kb.index.hybrid import HybridIndex
from henet_kb.llm.base import LLMProvider
from henet_kb.tools.search import SearchTool


@dataclass
class AskResult:
    answer: str
    sources: list[dict[str, str]]
    query: str
    grade: str
    rewrites: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    thread_id: str
    model: str
    tool_rounds: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("extra")
        return data


class AgentService:
    """Holds the compiled graph and flattens its stream into simple events."""

    def __init__(
        self,
        provider: LLMProvider,
        index: HybridIndex,
        settings: Settings,
        checkpoint_path: str | None = None,
    ) -> None:
        self.provider = provider
        self.index = index
        self.settings = settings
        self.search_tool = SearchTool(index, default_top_k=settings.top_k)
        checkpointer = None
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            checkpointer = SqliteSaver(connection)
        self.graph = build_graph(
            provider,
            self.search_tool,
            top_k=settings.top_k,
            max_rewrites=settings.max_rewrites,
            checkpointer=checkpointer,
        )

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.recursion_limit,
        }

    @staticmethod
    def _initial_state(question: str) -> AgentState:
        return {
            "question": question,
            "query": "",
            "hits": [],
            "rewrites": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    def _result(self, state: dict[str, Any], thread_id: str) -> AskResult:
        return AskResult(
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            query=state.get("query", ""),
            grade=state.get("grade", ""),
            rewrites=state.get("rewrites", 0),
            input_tokens=state.get("input_tokens", 0),
            output_tokens=state.get("output_tokens", 0),
            cost_usd=round(state.get("cost_usd", 0.0), 6),
            thread_id=thread_id,
            model=self.provider.model,
            tool_rounds=state.get("tool_rounds", 0),
        )

    def ask(self, question: str, thread_id: str | None = None) -> AskResult:
        thread_id = thread_id or uuid.uuid4().hex
        final_state = self.graph.invoke(self._initial_state(question), self._config(thread_id))
        return self._result(final_state, thread_id)

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yields dicts with a "type" of start, status, delta, sources, done or error."""
        thread_id = thread_id or uuid.uuid4().hex
        yield {"type": "start", "thread_id": thread_id, "question": question}
        state: dict[str, Any] = dict(self._initial_state(question))
        try:
            for mode, payload in self.graph.stream(
                self._initial_state(question),
                self._config(thread_id),
                stream_mode=["custom", "updates"],
            ):
                if mode == "custom":
                    yield payload
                elif mode == "updates":
                    for node_update in payload.values():
                        for key, value in node_update.items():
                            if key in ("input_tokens", "output_tokens", "cost_usd"):
                                state[key] = state.get(key, 0) + value
                            else:
                                state[key] = value
        except Exception as exc:
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return
        yield {"type": "done", **self._result(state, thread_id).to_dict()}
