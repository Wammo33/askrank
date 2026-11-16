from pathlib import Path

import pytest

from askrank.config import load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "brackenfell-gp.yaml"


@pytest.fixture
def config():
    return load_config(EXAMPLE)
