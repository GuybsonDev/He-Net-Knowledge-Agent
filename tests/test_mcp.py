import json

import pytest

from henet_kb.agent import AgentService
from henet_kb.config import Settings
from henet_kb.mcp_server import build_mcp
from tests.fakes import FakeProvider
from tests.test_api import script
from tests.test_hybrid import build_index


@pytest.fixture
def mcp():
    settings = Settings(openai_api_key="x", top_k=3, _env_file=None)
    return build_mcp(AgentService(FakeProvider(script), build_index(), settings))


async def test_tools_are_listed_with_schemas(mcp):
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert set(tools) == {"search_knowledge_base", "ask"}
    assert set(tools["search_knowledge_base"].inputSchema["properties"]) == {"query", "top_k"}
    assert tools["ask"].inputSchema["required"] == ["question"]


async def test_search_tool_returns_excerpts(mcp):
    content, structured = await mcp.call_tool(
        "search_knowledge_base", {"query": "esporte", "top_k": 1}
    )

    assert structured["results"][0]["url"] == "https://henet.com.br/tv"
    assert json.loads(content[0].text)["results"][0]["url"] == "https://henet.com.br/tv"


async def test_ask_tool_returns_answer_and_sources(mcp):
    _, structured = await mcp.call_tool("ask", {"question": "fibra?"})

    assert structured["answer"] == "Fibra de 500 mega."
    assert structured["sources"][0]["url"] == "https://henet.com.br/internet"
    assert structured["cost_usd"] == 0.002
