"""Summarise a run: mention rate with confidence intervals, share of voice,
average position, sentiment and the sources the models relied on."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from askrank.config import Config
from askrank.stats import share_of_voice, wilson_interval
from askrank.store import Store


@dataclass
class BusinessStats:
    name: str
    is_target: bool
    answers_mentioning: int
    total_answers: int
    ci: tuple[float, float]
    share_of_voice: float
    avg_position: float | None
    sentiment: Counter = field(default_factory=Counter)

    @property
    def mention_rate(self) -> float:
        return self.answers_mentioning / self.total_answers if self.total_answers else 0.0


@dataclass
class Summary:
    run_id: int
    total_answers: int
    businesses: list[BusinessStats]
    untracked: Counter  # businesses the models named that aren't in your config
    domains: list[tuple[str, int]]


def summarise(store: Store, run_id: int, config: Config, confidence: float = 0.95) -> Summary:
    answers = [a for a in store.answers(run_id) if a["extracted"]]
    n = len(answers)
    mentions = store.mentions(run_id)

    answer_sets: dict[str, set[int]] = defaultdict(set)
    positions: dict[str, list[int]] = defaultdict(list)
    sentiments: dict[str, Counter] = defaultdict(Counter)
    untracked: Counter = Counter()
    for m in mentions:
        if m["matched"] is None:
            untracked[m["name"]] += 1
            continue
        answer_sets[m["matched"]].add(m["answer_id"])
        positions[m["matched"]].append(m["position"])
        sentiments[m["matched"]][m["sentiment"]] += 1

    counts = {b.name: len(answer_sets[b.name]) for b in config.businesses}
    sov = share_of_voice(counts)
    stats = [
        BusinessStats(
            name=b.name,
            is_target=(b.name == config.target.name),
            answers_mentioning=counts[b.name],
            total_answers=n,
            ci=wilson_interval(counts[b.name], n, confidence) if n else (0.0, 1.0),
            share_of_voice=sov[b.name],
            avg_position=(sum(positions[b.name]) / len(positions[b.name])
                          if positions[b.name] else None),
            sentiment=sentiments[b.name],
        )
        for b in config.businesses
    ]
    stats.sort(key=lambda s: (-s.answers_mentioning, s.name))
    return Summary(run_id, n, stats, untracked, store.cited_domains(run_id))


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def render(summary: Summary, console: Console | None = None, top_n: int = 10) -> None:
    console = console or Console()
    t = Table(title=f"Share of voice: run #{summary.run_id} ({summary.total_answers} answers)")
    t.add_column("Business")
    t.add_column("Mention rate", justify="right")
    t.add_column("95% CI", justify="right")
    t.add_column("Share of voice", justify="right")
    t.add_column("Avg position", justify="right")
    t.add_column("Sentiment +/=/~/-", justify="right")
    for s in summary.businesses:
        name = f"[bold green]{s.name} (you)[/bold green]" if s.is_target else s.name
        sent = s.sentiment
        t.add_row(
            name,
            f"{_pct(s.mention_rate)} ({s.answers_mentioning}/{s.total_answers})",
            f"{_pct(s.ci[0])}-{_pct(s.ci[1])}",
            _pct(s.share_of_voice),
            f"{s.avg_position:.1f}" if s.avg_position is not None else "-",
            f"{sent['positive']}/{sent['neutral']}/{sent['hedged']}/{sent['negative']}",
        )
    console.print(t)

    if summary.untracked:
        u = Table(title="Other businesses the models named (consider adding as competitors)")
        u.add_column("Name")
        u.add_column("Mentions", justify="right")
        for name, c in summary.untracked.most_common(top_n):
            u.add_row(name, str(c))
        console.print(u)

    if summary.domains:
        d = Table(title="Top cited sources (what the models relied on)")
        d.add_column("Domain")
        d.add_column("Answers citing", justify="right")
        for dom, c in summary.domains[:top_n]:
            d.add_row(dom, str(c))
        console.print(d)
