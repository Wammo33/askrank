# askrank

**How often do AI assistants recommend your business, compared with your competitors?**

People now ask ChatGPT, Claude, Gemini and Perplexity things like *"best GP in Brackenfell"*
instead of Googling. askrank asks those questions for you, many times over, records who
gets named, in what position and with what sources, and reports your share of voice
**with confidence intervals**, because a single AI answer tells you almost nothing.

> Status: Phase 1 MVP. One provider (OpenAI with web search), rule-based and LLM
> extractors, SQLite storage, terminal report. Multi-model, evals and a dashboard are next.

## Quick start (30 seconds, no API key)

```bash
git clone https://github.com/Wammo33/askrank && cd askrank
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
askrank run examples/brackenfell-gp.yaml --fake
```

`--fake` uses canned answers so you can see the whole pipeline for free.

## Real run

```bash
cp .env.example .env          # then put your key in .env (it's git-ignored)
askrank validate examples/brackenfell-gp.yaml   # shows how many API calls a run makes
askrank run examples/brackenfell-gp.yaml --runs 3
askrank report examples/brackenfell-gp.yaml     # re-print the latest run
```

Set a monthly spending limit on your API account before your first real run.

## What it measures

| Metric | Meaning |
|---|---|
| Mention rate | % of answers that name the business, with a 95% Wilson interval |
| Share of voice | Your mentioning answers ÷ all tracked-business mentioning answers |
| Average position | 1 = named first |
| Sentiment | positive / neutral / hedged / negative (LLM extractor only) |
| Cited sources | Domains the model relied on: this is what to fix |
| Other businesses | Names the models gave that aren't in your config |

### Why confidence intervals?

Ask the same model the same question ten times and you'll get several different answers.
"Mentioned in 4 of 10 runs" is an estimate: the 95% interval is roughly 17% to 69%.
askrank uses the **Wilson score interval** (it behaves well at small n and near 0% or 100%)
and a **two-proportion z-test** to check whether a change between two runs is real.

## How it works

```
config.yaml ─► runner ─► provider adapter (OpenAI Responses API + web_search)
                               │
                               ▼
                       raw answer saved (SQLite)
                               │
                               ▼
              extractor: rules (alias regex) or LLM (fixed JSON schema)
                               │
                               ▼
                   mentions, positions, sentiment, URLs
                               │
                               ▼
                    stats (Wilson CI, z-test) ─► report
```

Raw answers are stored before extraction, so you can re-extract later without paying again.

## Honest limitations

- **API ≠ app.** The ChatGPT app has memory, personalisation and its own search pipeline.
  askrank uses the official API with the web-search tool turned on, which is the closest
  you can get legally, but results can still differ from what a given user sees.
- **Location.** Apps personalise by where you are; APIs mostly don't. Put the place name
  in the question wording, and askrank also passes an approximate location to the search tool.
- **Official APIs only.** askrank never scrapes consumer apps.
- **Measurement, not magic.** It tells you where you stand and what sources matter. It
  can't guarantee a better ranking.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

See [CLAUDE.md](CLAUDE.md) for architecture notes and conventions.

## Roadmap

- [x] Phase 1: YAML config, one provider, n runs, extraction, share-of-voice table
- [ ] Phase 2: Claude, Gemini, Perplexity; model breakdown; hand-labelled eval set
- [ ] Phase 3: weekly tracking and Streamlit dashboard
- [ ] Phase 4: AWS Lambda schedule and monthly PDF report

## License

MIT
