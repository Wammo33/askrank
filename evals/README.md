# Extractor evals

Measures how accurately the extractor finds business mentions in AI answers.

1. After a real run: `askrank label-export examples/dr-kretschmer.yaml --limit 50`
   writes `evals/labels.jsonl`, one answer per line, pre-filled by the rule extractor.
2. For each line, read `text` and fix `gold`: every business the answer names, in the
   order it first appears, with `sentiment` (positive / neutral / hedged / negative).
   Include businesses that aren't in your config. Then set `"labelled": true`.
3. Score: `askrank eval examples/dr-kretschmer.yaml evals/labels.jsonl --extractor openai`
   (and `--extractor rules` to compare).

Tips: label answers from all four models, and include some where the target is not
mentioned. About 50 answers takes an hour or so.

Note: the rule extractor only sees businesses listed in the config, so its positions
count tracked businesses only. The LLM extractor counts every business. That difference
shows up as lower position accuracy for rules.
