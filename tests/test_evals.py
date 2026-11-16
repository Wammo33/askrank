import json

from askrank.evals import GoldMention, LabelledAnswer, evaluate, export_for_labelling, load_labelled
from askrank.extract import RuleExtractor
from askrank.providers import AnswerResult, FakeProvider
from askrank.runner import run
from askrank.schemas import Sentiment
from askrank.store import Store


def _item(text, gold):
    return LabelledAnswer(id="x", provider="p", model="m", query_id="q", query_text="?",
                          text=text, gold=[GoldMention(name=n) for n in gold], labelled=True)


def test_perfect_score(config):
    items = [_item("Example FP then Competitor A.",
                   ["Example Family Practice", "Competitor Medical Centre A"])]
    r = evaluate(RuleExtractor(config), items, config.businesses)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.precision == r.recall == r.f1 == 1.0
    assert r.position_accuracy == 1.0


def test_rules_miss_untracked_businesses(config):
    # The rule extractor can't see businesses that aren't in the config: recall drops.
    items = [_item("Dr Unknown first, then Example FP.", ["Dr Unknown", "Example FP"])]
    r = evaluate(RuleExtractor(config), items, config.businesses)
    assert (r.tp, r.fp, r.fn) == (1, 0, 1)
    assert r.recall == 0.5
    assert r.position_accuracy == 0.0  # rules say position 1, truth is 2
    assert any("missed 'Dr Unknown'" in e for e in r.errors)


def test_sentiment_accuracy(config):
    item = _item("Example FP.", [])
    item.gold = [GoldMention(name="Example FP", sentiment=Sentiment.positive)]
    r = evaluate(RuleExtractor(config), [item], config.businesses)
    assert r.sentiment_accuracy == 0.0  # rules always say neutral


def test_export_then_load_roundtrip(config, tmp_path):
    store = Store(tmp_path / "t.db")
    fp = FakeProvider(answers=[AnswerResult("Example FP and Competitor B.")])
    rid = run(config, store, [fp], RuleExtractor(config), runs_per_query=1)
    out = tmp_path / "labels.jsonl"
    assert export_for_labelling(store, rid, config, out) == len(config.queries)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [g["name"] for g in lines[0]["gold"]] == ["Example Family Practice",
                                                     "Competitor Family Doctors B"]
    assert load_labelled(out) == []  # nothing marked labelled yet
    lines[0]["labelled"] = True
    out.write_text("\n".join(json.dumps(x) for x in lines))
    assert len(load_labelled(out)) == 1
