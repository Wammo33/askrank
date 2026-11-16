"""Model adapters. `get_provider` builds one from a config entry."""

from __future__ import annotations

from askrank.config import ProviderConfig
from askrank.providers.base import AnswerResult, Provider
from askrank.providers.fake import FakeProvider

__all__ = ["AnswerResult", "Provider", "FakeProvider", "get_provider"]


def get_provider(cfg: ProviderConfig) -> Provider:
    if cfg.name == "fake":
        return FakeProvider(model=cfg.model)
    if cfg.name == "openai":
        from askrank.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=cfg.model, web_search=cfg.web_search)
    raise ValueError(f"Unknown provider {cfg.name!r}. Available: openai, fake")
