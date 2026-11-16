"""A deterministic offline provider for tests and demos. Costs nothing.

It cycles through canned answers so you can try the whole pipeline
(`askrank run ... --provider fake`) before spending a cent on API calls.
"""

from __future__ import annotations

import random

from askrank.config import Location
from askrank.providers.base import AnswerResult
from askrank.schemas import Citation


class FakeProvider:
    name = "fake"

    def __init__(self, model: str = "fake-1", answers: list[AnswerResult] | None = None,
                 seed: int = 0):
        self.model = model
        self._answers = answers
        self._rng = random.Random(seed)

    def ask(self, prompt: str, location: Location | None = None) -> AnswerResult:
        if self._answers:
            return self._rng.choice(self._answers)
        return self._rng.choice(_DEMO)


# Placeholders: {t} = your business, {a} {b} {c} = competitors (made up if you have none).
_TEMPLATES = [
    ("Here are some well-reviewed GPs in the area:\n"
     "1. **{a}** - popular, open Saturdays.\n"
     "2. **{t}** - friendly doctors, does medicals.\n"
     "3. {c} - some reviews mention long waits.",
     ["https://www.medpages.info/sf/index.php?page=listing", "https://www.google.com/maps/place/x"]),
    ("{t} is a highly rated option nearby. {b} is another choice.",
     ["{site}"]),
    ("Try {a} or {b}; both are close by and well reviewed.",
     ["https://www.healthbridge.co.za/find"]),
]

_PLACEHOLDER_COMPETITORS = ["Sample Medical Centre", "Sample Family Doctors", "Sample Clinic"]


def demo_answers(target: str, competitors: list[str], website: str | None = None
                 ) -> list[AnswerResult]:
    """Canned answers that name YOUR businesses, so `--fake` output looks like your report."""
    comps = (competitors + _PLACEHOLDER_COMPETITORS)[:3]
    site = f"https://{website}/" if website else "https://example-practice.co.za/"
    fill = {"t": target, "a": comps[0], "b": comps[1], "c": comps[2], "site": site}
    return [AnswerResult(text.format(**fill), [Citation(url=u.format(**fill)) for u in urls])
            for text, urls in _TEMPLATES]


_DEMO = demo_answers("Example Family Practice",
                     ["Competitor Medical Centre A", "Competitor Family Doctors B",
                      "Competitor Clinic C (Kuils River)"], "example-practice.co.za")
