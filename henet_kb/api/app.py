import logging
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from henet_kb import __version__
from henet_kb.agent.service import AgentService
from henet_kb.api.sse import encode_event
from henet_kb.config import Settings, get_settings
from henet_kb.index import HybridIndex, OpenAIEmbedder
from henet_kb.ingest.base import Source
from henet_kb.llm import make_provider

log = logging.getLogger(__name__)

SourceFactory = Callable[[str, str, int | None], Source]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=128)


class IngestRequest(BaseModel):
    source: str = Field(default="sitemap", pattern="^(sitemap|wordpress)$")
    site_url: str | None = None
    max_pages: int | None = Field(default=None, ge=1)
    reset: bool = False


def default_service(settings: Settings) -> AgentService:
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    index = HybridIndex.from_settings(settings, embedder)
    return AgentService(make_provider(settings), index, settings, settings.checkpoint_db)


def default_source_factory(name: str, site_url: str, max_pages: int | None) -> Source:
    from henet_kb.cli import make_source

    return make_source(name, site_url, max_pages)


def create_app(
    settings: Settings | None = None,
    service: AgentService | None = None,
    source_factory: SourceFactory = default_source_factory,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="He-Net knowledge base", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @lru_cache
    def get_service() -> AgentService:
        return service or default_service(settings)

    def require_ingest_token(authorization: str | None = Header(default=None)) -> None:
        if not settings.ingest_token:
            raise HTTPException(status_code=404, detail="ingest endpoint is disabled")
        if authorization != f"Bearer {settings.ingest_token}":
            raise HTTPException(status_code=401, detail="invalid token")

    Agent = Annotated[AgentService, Depends(get_service)]

    @app.get("/health")
    def health(agent: Agent) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.llm_provider,
            "model": agent.provider.model,
            "chunks": agent.index.count(),
        }

    @app.post("/ask")
    def ask(body: AskRequest, agent: Agent) -> dict[str, Any]:
        return agent.ask(body.question, body.thread_id).to_dict()

    @app.post("/ask/stream")
    async def ask_stream(body: AskRequest, request: Request, agent: Agent) -> StreamingResponse:
        def events() -> Iterator[str]:
            for event in agent.stream(body.question, body.thread_id):
                payload = dict(event)
                name = str(payload.pop("type"))
                yield encode_event(name, payload)

        async def body_iterator() -> Any:
            async for chunk in iterate_in_threadpool(events()):
                if await request.is_disconnected():
                    break
                yield chunk

        return StreamingResponse(
            body_iterator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/ingest", dependencies=[Depends(require_ingest_token)])
    def ingest(body: IngestRequest, agent: Agent) -> dict[str, Any]:
        from henet_kb.cli import ingest as run_ingest

        source = source_factory(body.source, body.site_url or settings.site_url, body.max_pages)
        summary = run_ingest(source, agent.index, reset=body.reset)
        return {"source": body.source, **summary}

    return app


app = create_app()
