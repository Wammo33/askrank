"""OpenAI adapter using the Responses API with the built-in web search tool.

Web search matters: the consumer ChatGPT app searches the web, and the plain API
model does not unless you give it this tool. Without it you'd be measuring a
different thing from what real users see.
"""

from __future__ import annotations

import os

from askrank.config import Location
from askrank.providers.base import AnswerResult
from askrank.schemas import Citation


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, web_search: bool = True, client=None):
        self.model = model
        self.web_search = web_search
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it in your shell or put it in .env; "
                    "never commit it."
                )
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def _tools(self, location: Location | None) -> list[dict]:
        if not self.web_search:
            return []
        tool: dict = {"type": "web_search"}
        if location is not None:
            loc = {k: v for k, v in location.model_dump().items() if v}
            if loc:
                tool["user_location"] = {"type": "approximate", **loc}
        return [tool]

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            tools=self._tools(location),
        )
        return AnswerResult(text=resp.output_text, citations=_citations(resp))


def _citations(resp) -> list[Citation]:
    """Pull url_citation annotations out of the response, de-duplicated, in order."""
    seen: dict[str, Citation] = {}
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation" and ann.url not in seen:
                    seen[ann.url] = Citation(url=ann.url, title=getattr(ann, "title", None))
    return list(seen.values())
