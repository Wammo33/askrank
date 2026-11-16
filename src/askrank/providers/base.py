"""The provider interface. Every model adapter implements `ask`.

Adding Claude, Gemini or Perplexity in Phase 2 means writing one new class with
this method; nothing else in the pipeline has to change.
"""

from __future__ import annotations

from typing import Protocol

from askrank.config import Location
from askrank.schemas import Citation


class AnswerResult:
    def __init__(self, text: str, citations: list[Citation] | None = None):
        self.text = text
        self.citations = citations or []


class Provider(Protocol):
    name: str
    model: str

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult: ...
