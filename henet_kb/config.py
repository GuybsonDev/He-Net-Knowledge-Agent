from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-opus-5"
    embedding_model: str = "text-embedding-3-small"

    site_url: str = "https://henet.com.br"
    chroma_path: str = "data/chroma"
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_collection: str = "henet"
    checkpoint_db: str = "data/checkpoints.sqlite"

    ingest_token: str = ""
    cors_origins: str = "http://localhost:3000"

    top_k: int = 6
    max_rewrites: int = 2
    recursion_limit: int = 25

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
