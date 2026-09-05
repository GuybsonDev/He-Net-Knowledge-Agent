import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from henet_kb.agent.service import AgentService

INSTRUCTIONS = (
    "Answers questions about He-Net, an internet, TV and mobile provider in Bahia, Brazil, "
    "from its public website. Use search_knowledge_base for raw excerpts and ask for a "
    "complete answer with sources."
)


def build_mcp(service: AgentService, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    mcp = FastMCP("henet-kb", instructions=INSTRUCTIONS, host=host, port=port)

    @mcp.tool()
    def search_knowledge_base(query: str, top_k: int = 6) -> dict[str, Any]:
        """Hybrid search over the He-Net knowledge base. Returns excerpts with source URLs."""
        return service.search_tool({"query": query, "top_k": top_k})

    @mcp.tool()
    def ask(question: str, thread_id: str | None = None) -> dict[str, Any]:
        """Answer a question about He-Net with the retrieval agent. Includes sources and cost."""
        return service.ask(question, thread_id).to_dict()

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="henet-kb-mcp")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    # stdio is the protocol channel, so logs must never reach stdout.
    logging.basicConfig(level=logging.WARNING)

    from henet_kb.api.app import default_service
    from henet_kb.config import get_settings

    mcp = build_mcp(default_service(get_settings()), host=args.host, port=args.port)
    mcp.run(transport="stdio" if args.transport == "stdio" else "streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
