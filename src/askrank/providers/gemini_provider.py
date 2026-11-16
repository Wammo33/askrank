"""Google Gemini adapter with Google Search grounding.

Gemini's grounding sources come back as redirect links
(vertexaisearch.cloud.google.com/grounding-api-redirect/...), with the real site in
the `title` field. We rewrite those to https://<title>/ so the "top cited sources"
table shows medpages.info rather than a Google redirect host.

The Gemini API doesn't take a user location for search, so the place name must be in
the question wording.
"""

from __future__ import annotations

import os
import re

from askrank.config import Location
from askrank.providers.base import AnswerResult
from askrank.schemas import Citation

_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$", re.IGNORECASE)


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, web_search: bool = True, client=None):
        self.model = model
        self.web_search = web_search
        if client is None:
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise RuntimeError("GEMINI_API_KEY is not set (export it or put it in .env).")
            from google import genai

            client = genai.Client()
        self.client = client

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult:
        from google.genai import types

        tools = [types.Tool(google_search=types.GoogleSearch())] if self.web_search else []
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(tools=tools),
        )
        return AnswerResult(text=resp.text or "", citations=_citations(resp))


def _citations(resp) -> list[Citation]:
    seen: dict[str, Citation] = {}
    for cand in getattr(resp, "candidates", None) or []:
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None or not web.uri:
                continue
            url, title = web.uri, web.title
            if _REDIRECT_HOST in url and title and _DOMAIN_RE.match(title):
                url = f"https://{title.lower()}/"
            if url not in seen:
                seen[url] = Citation(url=url, title=title)
    return list(seen.values())
