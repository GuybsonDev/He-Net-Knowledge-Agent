from henet_kb.config import Settings
from henet_kb.llm.base import LLMProvider


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        from henet_kb.llm.anthropic_provider import AnthropicProvider

        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)

    from henet_kb.llm.openai_provider import OpenAIProvider

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAIProvider(settings.openai_api_key, settings.openai_model)
