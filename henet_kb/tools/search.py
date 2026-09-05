from typing import Any

from henet_kb.index.hybrid import HybridIndex, SearchHit
from henet_kb.llm.base import ToolSpec

MAX_TOP_K = 20

SEARCH_TOOL = ToolSpec(
    name="search_knowledge_base",
    description=(
        "Search the He-Net public knowledge base (plans, coverage, support, blog posts). "
        "Returns text excerpts with their source URL. Use it whenever you need facts "
        "you do not already have in the conversation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms, in Portuguese when possible.",
            },
            "top_k": {
                "type": "integer",
                "description": f"How many excerpts to return, between 1 and {MAX_TOP_K}.",
            },
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
)


class SearchTool:
    """Callable behind search_knowledge_base. Validates input and returns plain dicts."""

    name = SEARCH_TOOL.name

    def __init__(self, index: HybridIndex, default_top_k: int = 6) -> None:
        self.index = index
        self.default_top_k = default_top_k
        self.hits: list[SearchHit] = []

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = arguments.get("top_k", self.default_top_k)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        top_k = max(1, min(top_k, MAX_TOP_K))
        hits = self.index.search(query, top_k=top_k)
        self.hits.extend(hits)
        return {
            "query": query,
            "results": [
                {
                    "url": hit.url,
                    "title": hit.title,
                    "section": hit.section,
                    "text": hit.text,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }
