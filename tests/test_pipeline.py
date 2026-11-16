from typer.testing import CliRunner

from askrank.cli import app
from askrank.extract import RuleExtractor
from askrank.providers import AnswerResult, FakeProvider
from askrank.report import summarise
from askrank.runner import run
from askrank.store import Store

from .conftest import EXAMPLE


class Flaky:
    name, model = "flaky", "f"

    def __init__(self):
        self.calls = 0

    def ask(self, prompt, location=None):
        self.calls += 1
        if self.calls % 2 == 0:
            raise TimeoutError("boom")
        return AnswerResult("Example Family Practice, then Competitor B.")


def test_end_to_end_with_fixed_answers(config, tmp_path):
    store = Store(tmp_path / "t.db")
    provider = FakeProvider(answers=[AnswerResult("Example Family Practice and Competitor A.")])
    rid = run(config, store, [provider], RuleExtractor(config), runs_per_query=2)
    s = summarise(store, rid, config)
    assert s.total_answers == len(config.queries) * 2
    by = {b.name: b for b in s.businesses}
    assert by["Example Family Practice"].mention_rate == 1.0
    assert by["Example Family Practice"].avg_position == 1.0
    assert by["Competitor Medical Centre A"].avg_position == 2.0
    assert by["Example Family Practice"].share_of_voice == 0.5
    assert by["Competitor Family Doctors B"].answers_mentioning == 0


def test_failed_calls_do_not_abort_run(config, tmp_path):
    store = Store(tmp_path / "t.db")
    rid = run(config, store, [Flaky()], RuleExtractor(config), runs_per_query=2)
    assert len(store.answers(rid)) == len(config.queries)  # every second call failed


def test_cli_fake_run_and_report(tmp_path):
    db = str(tmp_path / "cli.db")
    r = CliRunner().invoke(app, ["run", str(EXAMPLE), "--fake", "--db", db, "-n", "2"])
    assert r.exit_code == 0, r.output
    assert "Share of voice" in r.output
    r = CliRunner().invoke(app, ["report", str(EXAMPLE), "--db", db])
    assert r.exit_code == 0, r.output
    assert "(you)" in r.output


def test_cli_validate():
    r = CliRunner().invoke(app, ["validate", str(EXAMPLE)])
    assert r.exit_code == 0
    assert "15 answer calls" in r.output
