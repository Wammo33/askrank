"""Anthropic (Claude) adapter using the Messages API with the server-side web search tool.

Claude runs the searches itself on Anthropic's side, so there's no tool loop to write.
Two details matter:

* `pause_turn`: a long search turn can pause; we send it back so the server continues.
* `refusal`: a safety classifier can decline (HTTP 200, stop_reason "refusal"). We
  enable server-side fallbacks so a decline is retried on another model in the same
  call, and raise if the whole chain still refuses, so the runner logs it.
"""

from __future__ import annotations

import os

from askrank.config import Location
from askrank.providers.base import AnswerResult
from askrank.schemas import Citation

WEB_SEARCH_TOOL = "web_search_20260209"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_CONTINUATIONS = 3


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, web_search: bool = True, max_searches: int = 5, client=None):
        self.model = model
        self.web_search = web_search
        self.max_searches = max_searches
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (export it or put it in .env).")
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def _tools(self, location: Location | None) -> list[dict]:
        if not self.web_search:
            return []
        tool: dict = {"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": self.max_searches}
        if location is not None:
            loc = {k: v for k, v in location.model_dump().items() if v}
            if loc:
                tool["user_location"] = {"type": "approximate", **loc}
        return [tool]

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tools = self._tools(location)
        resp = None
        for _ in range(MAX_CONTINUATIONS + 1):
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=messages,
                tools=tools,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            if resp.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined to answer: {resp.stop_details}")
        return _to_result(resp.content)


def _to_result(content) -> AnswerResult:
    texts: list[str] = []
    seen: dict[str, Citation] = {}
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        texts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            url = getattr(c, "url", None)
            if url and url not in seen:
                seen[url] = Citation(url=url, title=getattr(c, "title", None))
    return AnswerResult(text="".join(texts).strip(), citations=list(seen.values()))
