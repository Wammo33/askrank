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


_DEMO = [
    AnswerResult(
        "Here are some well-reviewed GPs in Brackenfell:\n"
        "1. **Competitor Medical Centre A** - popular, open Saturdays.\n"
        "2. **Example Family Practice** - friendly doctors, does medicals.\n"
        "3. Competitor Clinic C (Kuils River) - some reviews mention long waits.",
        [Citation(url="https://www.medpages.info/sf/index.php?page=listing"),
         Citation(url="https://www.google.com/maps/place/x")],
    ),
    AnswerResult(
        "Example Family Practice is a highly rated option in Brackenfell and is open on "
        "Saturday mornings. Competitor Family Doctors B is another choice.",
        [Citation(url="https://example-practice.co.za/hours")],
    ),
    AnswerResult(
        "Try Competitor Medical Centre A or Competitor Family Doctors B; both are close to "
        "Brackenfell and well reviewed.",
        [Citation(url="https://www.healthbridge.co.za/find")],
    ),
]
