"""Measure extractor accuracy against hand-labelled answers.

Workflow:
1. `askrank label-export` writes answers from a run to a JSONL file, pre-filled with
   the rule extractor's guess so labelling is mostly checking, not typing.
2. You open the file and fix each `gold` list: every business the answer names, in
   order, with sentiment. Set `"labelled": true` on each line you've checked.
3. `askrank eval` runs an extractor over the labelled answers and reports precision,
   recall, F1, position accuracy and sentiment accuracy.

Why this matters: every number in a report depends on the extractor. If it misses
20% of mentions, your mention rates are 20% too low. Publishing this accuracy is
what makes the rest of the numbers credible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from askrank.config import Business, Config
from askrank.extract import Extractor, RuleExtractor, match_business, normalise
from askrank.schemas import Answer, Citation, Sentiment
from askrank.store import Store


class GoldMention(BaseModel):
    name: str
    sentiment: Sentiment = Sentiment.neutral


class LabelledAnswer(BaseModel):
    id: str
    provider: str
    model: str
    query_id: str
    query_text: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    gold: list[GoldMention] = Field(default_factory=list)  # in order of appearance
    labelled: bool = False

    def to_answer(self) -> Answer:
        return Answer(provider=self.provider, model=self.model, query_id=self.query_id,
                      query_text=self.query_text, run_index=0, text=self.text,
                      citations=self.citations)


def export_for_labelling(store: Store, run_id: int, config: Config, out: Path,
                         limit: int | None = None) -> int:
    rules = RuleExtractor(config)
    rows = store.answers(run_id)[:limit] if limit else store.answers(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            item = LabelledAnswer(
                id=f"run{run_id}-a{r['id']}", provider=r["provider"], model=r["model"],
                query_id=r["query_id"], query_text=r["query_text"], text=r["text"],
                citations=[Citation(**c) for c in json.loads(r["citations_json"])],
            )
            guess = rules.extract(item.to_answer())
            item.gold = [GoldMention(name=m.name, sentiment=m.sentiment) for m in guess.mentions]
            f.write(item.model_dump_json() + "\n")
    return len(rows)


def load_labelled(path: Path) -> list[LabelledAnswer]:
    items = [LabelledAnswer.model_validate_json(line)
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [i for i in items if i.labelled]


def _key(name: str, businesses: list[Business]) -> str:
    return match_business(name, businesses) or normalise(name)


@dataclass
class EvalResult:
    n_answers: int
    tp: int
    fp: int
    fn: int
    position_correct: int
    sentiment_correct: int
    errors: list[str]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def position_accuracy(self) -> float:
        return self.position_correct / self.tp if self.tp else 1.0

    @property
    def sentiment_accuracy(self) -> float:
        return self.sentiment_correct / self.tp if self.tp else 1.0


def evaluate(extractor: Extractor, items: list[LabelledAnswer],
             businesses: list[Business]) -> EvalResult:
    tp = fp = fn = pos_ok = sent_ok = 0
    errors: list[str] = []
    for item in items:
        gold = {_key(g.name, businesses): (i, g) for i, g in enumerate(item.gold, start=1)}
        try:
            pred_list = extractor.extract(item.to_answer()).mentions
        except Exception as e:
            errors.append(f"{item.id}: {e}")
            fn += len(gold)
            continue
        pred = {}
        for m in pred_list:
            pred.setdefault(_key(m.name, businesses), m)
        for k, m in pred.items():
            if k in gold:
                tp += 1
                pos, g = gold[k]
                pos_ok += m.position == pos
                sent_ok += m.sentiment == g.sentiment
            else:
                fp += 1
                errors.append(f"{item.id}: extra '{m.name}'")
        for k, (_, g) in gold.items():
            if k not in pred:
                fn += 1
                errors.append(f"{item.id}: missed '{g.name}'")
    return EvalResult(len(items), tp, fp, fn, pos_ok, sent_ok, errors)
