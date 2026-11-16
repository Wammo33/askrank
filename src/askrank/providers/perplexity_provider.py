"""Perplexity Sonar adapter.

Sonar models always search the web. Perplexity's API is OpenAI-compatible, so we use
the `openai` client pointed at api.perplexity.ai; sources come back in extra response
fields (`search_results`, or the older `citations` list of URLs).
"""

from __future__ import annotations

import os

from askrank.config import Location
from askrank.providers.base import AnswerResult
from askrank.schemas import Citation

BASE_URL = "https://api.perplexity.ai"


class PerplexityProvider:
    name = "perplexity"

    def __init__(self, model: str, web_search: bool = True, client=None):
        self.model = model
        self.web_search = web_search  # Sonar always searches; kept for a uniform interface
        if client is None:
            key = os.environ.get("PERPLEXITY_API_KEY")
            if not key:
                raise RuntimeError("PERPLEXITY_API_KEY is not set (export it or put it in .env).")
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=BASE_URL)
        self.client = client

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult:
        extra: dict = {}
        if location is not None and location.country:
            extra["web_search_options"] = {"user_location": {"country": location.country}}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra or None,
        )
        text = resp.choices[0].message.content or ""
        return AnswerResult(text=text, citations=_citations(resp))


def _field(obj, name):
    val = getattr(obj, name, None)
    if val is None:
        val = (getattr(obj, "model_extra", None) or {}).get(name)
    return val


def _citations(resp) -> list[Citation]:
    seen: dict[str, Citation] = {}
    for r in _field(resp, "search_results") or []:
        url = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
        title = r.get("title") if isinstance(r, dict) else getattr(r, "title", None)
        if url and url not in seen:
            seen[url] = Citation(url=url, title=title)
    for url in _field(resp, "citations") or []:
        if isinstance(url, str) and url not in seen:
            seen[url] = Citation(url=url)
    return list(seen.values())
