"""Data shapes shared across the pipeline.

`Extraction` is the fixed JSON schema the extractor LLM must fill in. Keeping it
small and explicit is what makes the extractor testable with an eval set.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Sentiment(StrEnum):
    positive = "positive"
    neutral = "neutral"
    hedged = "hedged"  # e.g. "some reviews mention long waits"
    negative = "negative"


class Citation(BaseModel):
    url: str
    title: str | None = None


class Answer(BaseModel):
    """One raw answer from one model to one query."""

    provider: str
    model: str
    query_id: str
    query_text: str
    run_index: int
    text: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Mention(BaseModel):
    name: str = Field(description="Business name exactly as written in the answer")
    position: int = Field(ge=1, description="1 = first business named in the answer")
    sentiment: Sentiment
    # Filled in after extraction by matching against config names/aliases.
    matched: str | None = Field(
        default=None, description="Canonical config name, or null if not a tracked business"
    )


class Extraction(BaseModel):
    mentions: list[Mention] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)
