import pytest
from pydantic import ValidationError

from askrank.config import Config


def test_example_loads(config):
    assert config.target.name == "Example Family Practice"
    assert config.businesses[0] is config.target
    assert config.location.country == "ZA"


def _base(**over):
    d = {
        "project": "p",
        "target": {"name": "A"},
        "queries": [{"id": "q", "text": "best?"}],
        "providers": [{"name": "fake", "model": "m"}],
    }
    d.update(over)
    return d


def test_duplicate_query_ids_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_base(queries=[{"id": "q", "text": "a"}, {"id": "q", "text": "b"}]))


def test_duplicate_business_names_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_base(competitors=[{"name": "A"}]))


def test_bad_country_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_base(location={"country": "South Africa"}))


def test_runs_per_query_bounds():
    with pytest.raises(ValidationError):
        Config.model_validate(_base(runs_per_query=0))


def test_pilot_config_loads():
    from pathlib import Path

    from askrank.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "examples" / "brackenfell-gp.yaml")
    assert cfg.target.name == "Dr Wayne Kretschmer"
    assert {p.name for p in cfg.providers} == {"openai", "anthropic", "gemini", "perplexity"}


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_base(providers=[{"name": "bard", "model": "m"}]))
