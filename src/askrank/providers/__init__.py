"""Model adapters. `get_provider` builds one from a config entry.

Each adapter imports its SDK lazily, so a missing key only matters for the
providers you actually use.
"""

from __future__ import annotations

from askrank.config import ProviderConfig
from askrank.providers.base import AnswerResult, Provider
from askrank.providers.fake import FakeProvider

__all__ = ["AnswerResult", "Provider", "FakeProvider", "get_provider", "PROVIDERS"]

PROVIDERS = ("openai", "anthropic", "gemini", "perplexity", "fake")


def get_provider(cfg: ProviderConfig) -> Provider:
    if cfg.name == "fake":
        return FakeProvider(model=cfg.model, seed=sum(map(ord, cfg.model)))
    if cfg.name == "openai":
        from askrank.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=cfg.model, web_search=cfg.web_search)
    if cfg.name == "anthropic":
        from askrank.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=cfg.model, web_search=cfg.web_search)
    if cfg.name == "gemini":
        from askrank.providers.gemini_provider import GeminiProvider

        return GeminiProvider(model=cfg.model, web_search=cfg.web_search)
    if cfg.name == "perplexity":
        from askrank.providers.perplexity_provider import PerplexityProvider

        return PerplexityProvider(model=cfg.model, web_search=cfg.web_search)
    raise ValueError(f"Unknown provider {cfg.name!r}. Available: {', '.join(PROVIDERS)}")
