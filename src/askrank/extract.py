"""Turn a free-text answer into structured data (who was mentioned, where, how).

Two extractors share one interface:

* `RuleExtractor`  - matches config names/aliases with regex. Free, fast,
  deterministic, but blind to sentiment and to businesses you didn't list.
* `LLMExtractor`   - asks a small model to fill the fixed `Extraction` schema
  (structured outputs), then maps names back onto the config.

The eval set in `evals/` (Phase 2) measures how accurate each one is.
"""

from __future__ import annotations

import re
from typing import Protocol

from askrank.config import Business, Config
from askrank.schemas import Answer, Extraction, Mention, Sentiment

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


class Extractor(Protocol):
    def extract(self, answer: Answer) -> Extraction: ...


def normalise(name: str) -> str:
    name = name.lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def match_business(name: str, businesses: list[Business]) -> str | None:
    """Map a name as written in an answer to a canonical config name, if any.

    Exact (normalised) match wins; otherwise accept containment either way, so
    "Example Family Practice Brackenfell" still matches "Example Family Practice".
    """
    n = normalise(name)
    if not n:
        return None
    for b in businesses:
        if any(normalise(a) == n for a in b.all_names):
            return b.name
    for b in businesses:
        for a in b.all_names:
            na = normalise(a)
            if len(na) >= 4 and len(n) >= 4 and (na in n or n in na):
                return b.name
    return None


def _all_urls(answer: Answer, extra: list[str] | None = None) -> list[str]:
    urls = [c.url for c in answer.citations] + _URL_RE.findall(answer.text) + (extra or [])
    return list(dict.fromkeys(u.rstrip(".,;") for u in urls))


class RuleExtractor:
    def __init__(self, config: Config):
        self.businesses = config.businesses

    def extract(self, answer: Answer) -> Extraction:
        hits: list[tuple[int, Business]] = []
        for b in self.businesses:
            pattern = "|".join(re.escape(a) for a in sorted(b.all_names, key=len, reverse=True))
            m = re.search(rf"(?<!\w)(?:{pattern})(?!\w)", answer.text, flags=re.IGNORECASE)
            if m:
                hits.append((m.start(), b))
        hits.sort(key=lambda h: h[0])
        mentions = [
            Mention(name=b.name, position=i, sentiment=Sentiment.neutral, matched=b.name)
            for i, (_, b) in enumerate(hits, start=1)
        ]
        return Extraction(mentions=mentions, cited_urls=_all_urls(answer))


EXTRACTION_PROMPT = """You extract structured data from an AI assistant's answer.
List EVERY specific business (practice, clinic, company) the answer names, in the order
they first appear. Do not list generic things ("a GP", "Google Maps", "Medpages").
position: 1 for the first business named, 2 for the next, etc.
sentiment toward each business:
  positive = recommended or praised; neutral = just listed;
  hedged = recommended with a caveat (e.g. "some reviews mention long waits");
  negative = advised against.
cited_urls: every URL that appears in the answer text.
Leave `matched` null. Return only what is in the answer; never invent businesses."""


class LLMExtractor:
    def __init__(self, config: Config, model: str, client=None):
        self.businesses = config.businesses
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def extract(self, answer: Answer) -> Extraction:
        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": answer.text},
            ],
            text_format=Extraction,
        )
        ex: Extraction = resp.output_parsed or Extraction()
        return postprocess(ex, answer, self.businesses)


def postprocess(ex: Extraction, answer: Answer, businesses: list[Business]) -> Extraction:
    """Re-number positions, match to config, merge duplicates, add API citations.

    Never trust the model's bookkeeping: we fix positions and matching ourselves.
    """
    out: list[Mention] = []
    seen: set[str] = set()
    for m in sorted(ex.mentions, key=lambda m: m.position):
        matched = match_business(m.name, businesses)
        key = matched or normalise(m.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(m.model_copy(update={"matched": matched, "position": len(out) + 1}))
    return Extraction(mentions=out, cited_urls=_all_urls(answer, ex.cited_urls))


def get_extractor(config: Config) -> Extractor:
    if config.extractor.provider == "rules":
        return RuleExtractor(config)
    if config.extractor.provider == "openai":
        return LLMExtractor(config, model=config.extractor.model)
    raise ValueError(f"Unknown extractor {config.extractor.provider!r}. Use openai or rules")
