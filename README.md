# askrank

**How often do AI assistants recommend your business, compared with your competitors?**

People now ask ChatGPT, Claude, Gemini and Perplexity things like *"best GP in Brackenfell"*
instead of Googling. askrank asks those questions for you, many times over, records who
gets named, in what position and with what sources, and reports your share of voice
**with confidence intervals**, because a single AI answer tells you almost nothing.

> Status: Phase 2. Four providers (ChatGPT, Claude, Gemini, Perplexity, all with web
> search), model breakdown with confidence intervals, before/after significance test,
> and an extractor eval workflow. Weekly tracking and a dashboard are next.

## Quick start (30 seconds, no API key)

```bash
git clone https://github.com/Wammo33/askrank && cd askrank
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
askrank run examples/demo.yaml --fake
```

`--fake` uses canned answers so you can see the whole pipeline for free.

## Real run

```bash
cp .env.example .env          # then put your keys in .env (it's git-ignored)
askrank validate examples/dr-kretschmer.yaml   # shows how many API calls a run makes
askrank run examples/dr-kretschmer.yaml --runs 3
askrank report examples/dr-kretschmer.yaml     # re-print the latest run
askrank compare examples/dr-kretschmer.yaml 1 2  # did visibility really change?
```

Set a monthly spending limit on every API account before your first real run.

Only have some keys? Pick providers with `--only`. With just the free Gemini key:

```bash
askrank run examples/dr-kretschmer.yaml --only gemini --extractor rules --runs 1
```

| Provider | Key | Search mode used |
|---|---|---|
| OpenAI (ChatGPT) | `OPENAI_API_KEY` | Responses API `web_search` tool, approximate user location |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Server-side `web_search` tool, approximate user location |
| Google (Gemini) | `GEMINI_API_KEY` | Google Search grounding (no location parameter) |
| Perplexity | `PERPLEXITY_API_KEY` | Sonar (always searches), country hint |

## Extractor accuracy (evals)

Every number above depends on the extractor finding mentions correctly, so askrank
measures it against hand-labelled answers:

```bash
askrank label-export examples/dr-kretschmer.yaml --limit 50   # writes evals/labels.jsonl
# fix each "gold" list by hand and set "labelled": true
askrank eval examples/dr-kretschmer.yaml evals/labels.jsonl --extractor rules
askrank eval examples/dr-kretschmer.yaml evals/labels.jsonl --extractor openai
```

It reports precision, recall, F1, position accuracy and sentiment accuracy.

| Extractor | Precision | Recall | F1 | Sentiment |
|---|---|---|---|---|
| rules | _tbd_ | _tbd_ | _tbd_ | n/a |
| LLM (gpt-5.4-nano) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## What it measures

| Metric | Meaning |
|---|---|
| Mention rate | % of answers that name the business, with a 95% Wilson interval |
| Share of voice | Your mentioning answers ÷ all tracked-business mentioning answers |
| Average position | 1 = named first |
| Sentiment | positive / neutral / hedged / negative (LLM extractor only) |
| Cited sources | Domains the model relied on: this is what to fix |
| Model breakdown | Mention rate per model, each with its own interval |
| Other businesses | Names the models gave that aren't in your config |

### Why confidence intervals?

Ask the same model the same question ten times and you'll get several different answers.
"Mentioned in 4 of 10 runs" is an estimate: the 95% interval is roughly 17% to 69%.
askrank uses the **Wilson score interval** (it behaves well at small n and near 0% or 100%)
and a **two-proportion z-test** to check whether a change between two runs is real.

## How it works

```
config.yaml ─► runner ─► provider adapters (OpenAI, Claude, Gemini, Perplexity; web search on)
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
- [x] Phase 2: Claude, Gemini, Perplexity; model breakdown; compare runs; eval tooling
- [ ] Phase 2b: publish extractor accuracy on ~50 hand-labelled answers
- [ ] Phase 3: weekly tracking and Streamlit dashboard
- [ ] Phase 4: AWS Lambda schedule and monthly PDF report

## License

MIT
