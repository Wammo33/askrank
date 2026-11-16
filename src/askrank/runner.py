"""The main loop: every query x every provider x n runs -> answer -> extraction."""

from __future__ import annotations

from collections.abc import Callable

from askrank.config import Config
from askrank.extract import Extractor
from askrank.providers import Provider
from askrank.schemas import Answer
from askrank.store import Store


def run(
    config: Config,
    store: Store,
    providers: list[Provider],
    extractor: Extractor,
    runs_per_query: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> int:
    """Returns the run id. Progress goes to `on_progress` (transient); failures go to
    `on_error` (shown permanently), falling back to `on_progress`."""
    on_error = on_error or on_progress
    n = runs_per_query or config.runs_per_query
    run_id = store.start_run(config.project, config.model_dump(mode="json"))
    errors = 0
    for provider in providers:
        for q in config.queries:
            for i in range(n):
                label = f"{provider.name}/{provider.model} | {q.id} | run {i + 1}/{n}"
                try:
                    result = provider.ask(q.text, location=config.location)
                except Exception as e:  # one failed call shouldn't kill the batch
                    errors += 1
                    _say(on_error, f"[red]ask failed[/red] {label}: {_short(e)}")
                    continue
                answer = Answer(
                    provider=provider.name, model=provider.model, query_id=q.id,
                    query_text=q.text, run_index=i, text=result.text,
                    citations=result.citations,
                )
                answer_id = store.save_answer(run_id, answer)  # save raw first
                try:
                    store.save_extraction(answer_id, extractor.extract(answer))
                except Exception as e:
                    errors += 1
                    _say(on_error, f"[yellow]extract failed[/yellow] {label}: {_short(e)}")
                    continue
                _say(on_progress, label)
    if errors:
        _say(on_error, f"[yellow]{errors} call(s) failed; see messages above[/yellow]")
    return run_id


def _say(cb: Callable[[str], None] | None, msg: str) -> None:
    if cb:
        cb(msg)


def _short(e: Exception, limit: int = 300) -> str:
    msg = f"{type(e).__name__}: {e}".replace("\n", " ")
    return msg if len(msg) <= limit else msg[:limit] + "..."
