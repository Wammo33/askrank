"""Statistics for noisy LLM answers.

A model gives a different answer each time you ask, so "mentioned in 4 of 10
runs" is an estimate, not a fact. These functions put error bars on it.

Pure Python (only `math`), so it's easy to read and test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _z_for(confidence: float) -> float:
    """Two-sided z critical value, e.g. 0.95 -> 1.96."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return _norm_ppf(1 - (1 - confidence) / 2)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF by bisection: slow but simple and accurate enough."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Why Wilson and not the textbook p ± z*sqrt(p(1-p)/n)? With small n (we often
    have n=5 or 10) and rates near 0% or 100%, the textbook interval gives
    nonsense like [-5%, 25%] or zero width. Wilson stays inside [0, 1] and has
    much better coverage.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = _z_for(confidence)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ZTest:
    p1: float
    p2: float
    diff: float  # p2 - p1
    z: float
    p_value: float  # two-sided

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> ZTest:
    """Is the change from k1/n1 (before) to k2/n2 (after) real or just noise?

    Uses the pooled standard error under H0: p1 == p2. Rule of thumb: needs
    roughly 5+ successes and failures in each group to be trustworthy; with
    fewer, prefer Fisher's exact test (Phase 2).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both samples need n > 0")
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return ZTest(p1, p2, p2 - p1, 0.0, 1.0)
    z = (p2 - p1) / se
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return ZTest(p1, p2, p2 - p1, z, p_value)


def share_of_voice(mentions: dict[str, int]) -> dict[str, float]:
    """Each business's mentions divided by all tracked-business mentions."""
    total = sum(mentions.values())
    return {k: (v / total if total else 0.0) for k, v in mentions.items()}
