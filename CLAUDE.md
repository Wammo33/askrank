# CLAUDE.md

Context for Claude Code (and humans) working on askrank.

## What this is
An open-source AI-visibility (GEO) tracker. Given a business, its competitors and
realistic customer questions, it asks AI assistants each question n times, extracts which
businesses were named (position, sentiment, cited URLs), stores everything in SQLite and
reports mention rate with confidence intervals and share of voice.

## Layout
- `src/askrank/config.py`  Pydantic models + YAML loader. All config validation lives here.
- `src/askrank/schemas.py` Shared data shapes. `Extraction` is the extractor's fixed JSON schema.
- `src/askrank/providers/` One adapter per model. Each has `ask(prompt, location) -> AnswerResult`.
  `fake.py` is offline and deterministic; use it in tests.
- `src/askrank/extract.py` `RuleExtractor` (alias regex) and `LLMExtractor` (structured output).
  `postprocess` re-numbers positions and matches names to config; never trust model bookkeeping.
- `src/askrank/stats.py`   Pure-Python stats: Wilson interval, two-proportion z-test, share of voice.
- `src/askrank/store.py`   SQLite. Raw answers are saved before extraction.
- `src/askrank/runner.py`  Main loop. One failed call must not abort the batch.
- `src/askrank/report.py`  `summarise` (data) and `render` (rich tables) are separate on purpose.
- `src/askrank/evals.py`   Label export, labelled-answer loading, precision/recall scoring.
- `src/askrank/cli.py`     Typer CLI: `validate`, `run`, `report`, `answers`, `compare`, `label-export`, `eval`.
- `evals/`                 Hand-labelled answers for measuring extractor accuracy.
- `examples/demo.yaml`     Made-up names; used by tests and the `--fake` demo.
- `examples/dr-kretschmer.yaml` Real pilot config (Dr Kretschmer Associates, Kraaifontein, drwayne.co.za).

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest` (must pass with no network and no API keys)
- Lint: `ruff check .`
- Demo: `askrank run examples/demo.yaml --fake --db /tmp/demo.db`

## Conventions
- Tests never hit real APIs. Inject a fake client or use `FakeProvider`.
- API keys only via environment variables / `.env` (git-ignored). Never log or commit keys.
- Use the web-search-enabled mode of each provider so results reflect what real users see.
- Keep stats code dependency-free and explain the "why" in docstrings.
- Adding a provider: new file in `providers/`, register it in `providers/__init__.py:get_provider`,
  add a test with a fake client.
- Prefer small models for extraction to keep costs down.
- Provider quirks live in the adapter: Claude `pause_turn`/`refusal`, Gemini redirect
  citation URLs, Perplexity sources in extra response fields.
