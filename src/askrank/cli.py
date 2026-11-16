"""Command-line interface.

    askrank validate examples/dr-kretschmer.yaml
    askrank run examples/demo.yaml --fake               # free offline demo
    askrank run examples/dr-kretschmer.yaml --runs 3    # real API calls
    askrank report examples/dr-kretschmer.yaml
    askrank compare examples/dr-kretschmer.yaml 1 2     # did visibility really change?
    askrank label-export examples/dr-kretschmer.yaml    # start an eval set
    askrank eval examples/dr-kretschmer.yaml evals/labels.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from askrank.config import Config, ExtractorConfig, ProviderConfig, load_config
from askrank.evals import evaluate, export_for_labelling, load_labelled
from askrank.extract import get_extractor
from askrank.providers import FakeProvider, get_provider
from askrank.providers.fake import demo_answers
from askrank.report import render, summarise, target_counts
from askrank.runner import run as run_pipeline
from askrank.stats import two_proportion_z_test, wilson_interval
from askrank.store import Store

app = typer.Typer(help="Measure how often AI assistants recommend your business.",
                  no_args_is_help=True)
console = Console()

ConfigArg = typer.Argument(..., exists=True, dir_okay=False, help="Path to the YAML config")
DbOpt = typer.Option(Path("askrank.db"), "--db", help="SQLite database file")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _load(path: Path) -> Config:
    try:
        return load_config(path)
    except Exception as e:
        console.print(f"[red]Invalid config {path}:[/red]\n{e}")
        raise typer.Exit(1) from e


@app.command()
def validate(config: Path = ConfigArg) -> None:
    """Check a config file and show what a run would cost in API calls."""
    cfg = _load(config)
    calls = len(cfg.queries) * len(cfg.providers) * cfg.runs_per_query
    console.print(f"[green]OK[/green] {cfg.project}: {len(cfg.businesses)} businesses, "
                  f"{len(cfg.queries)} queries, {len(cfg.providers)} provider(s), "
                  f"{cfg.runs_per_query} runs each")
    extra = f" + {calls} extraction calls" if cfg.extractor.provider != "rules" else ""
    console.print(f"A run makes {calls} answer calls{extra}.")


@app.command()
def run(
    config: Path = ConfigArg,
    db: Path = DbOpt,
    runs: int | None = typer.Option(None, "--runs", "-n", min=1, help="Override runs_per_query"),
    fake: bool = typer.Option(False, "--fake", help="Offline demo: fake provider + rule extractor"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation"),
) -> None:
    """Ask every query to every provider n times, extract mentions, store results."""
    _load_dotenv()
    cfg = _load(config)
    if fake:
        cfg = cfg.model_copy(update={
            # one simulated model per configured provider, so the model breakdown shows
            "providers": [ProviderConfig(name="fake", model=f"{p.name}-sim")
                          for p in cfg.providers],
            "extractor": ExtractorConfig(provider="rules"),
        })
    n = runs or cfg.runs_per_query
    calls = len(cfg.queries) * len(cfg.providers) * n
    if not fake and not yes:
        typer.confirm(f"This makes about {calls} paid API answer calls. Continue?", abort=True)

    try:
        if fake:
            answers = demo_answers(cfg.target.name, [c.name for c in cfg.competitors],
                                   cfg.target.website)
            providers = [FakeProvider(model=p.model, answers=answers, seed=i)
                         for i, p in enumerate(cfg.providers)]
        else:
            providers = [get_provider(p) for p in cfg.providers]
        extractor = get_extractor(cfg)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    store = Store(db)
    try:
        with console.status("Asking the models...") as status:
            run_id = run_pipeline(cfg, store, providers, extractor, runs_per_query=n,
                                  on_progress=lambda m: status.update(m))
        render(summarise(store, run_id, cfg), console)
        console.print(f"Saved run #{run_id} to {db}")
    finally:
        store.close()


@app.command()
def report(
    config: Path = ConfigArg,
    db: Path = DbOpt,
    run_id: int | None = typer.Option(None, "--run", help="Run id (default: latest)"),
) -> None:
    """Print the share-of-voice report for a stored run."""
    cfg = _load(config)
    store = _open_store(db)
    try:
        rid = run_id or store.latest_run_id(cfg.project)
        if rid is None:
            console.print(f"[red]No runs for project {cfg.project!r} in {db}.[/red]")
            raise typer.Exit(1)
        render(summarise(store, rid, cfg), console)
    finally:
        store.close()


def _open_store(db: Path) -> Store:
    if not db.exists():
        console.print(f"[red]No database at {db}. Run `askrank run` first.[/red]")
        raise typer.Exit(1)
    return Store(db)


@app.command()
def compare(
    config: Path = ConfigArg,
    before: int = typer.Argument(..., help="Earlier run id"),
    after: int = typer.Argument(..., help="Later run id"),
    db: Path = DbOpt,
) -> None:
    """Test whether the target's mention rate really changed between two runs."""
    cfg = _load(config)
    store = _open_store(db)
    try:
        k1, n1 = target_counts(store, before, cfg)
        k2, n2 = target_counts(store, after, cfg)
    finally:
        store.close()
    if not n1 or not n2:
        console.print("[red]One of the runs has no extracted answers.[/red]")
        raise typer.Exit(1)
    r = two_proportion_z_test(k1, n1, k2, n2)
    for label, k, n in (("before", k1, n1), ("after", k2, n2)):
        lo, hi = wilson_interval(k, n)
        console.print(f"{label:>6}: {k}/{n} = {k / n:.0%}  (95% CI {lo:.0%}-{hi:.0%})")
    verdict = ("[green]a real change[/green]" if r.significant()
               else "[yellow]could be noise[/yellow]")
    console.print(f"change {r.diff:+.0%}, z = {r.z:.2f}, p = {r.p_value:.3f}: {verdict}")
    if min(k1, n1 - k1, k2, n2 - k2) < 5:
        console.print("[dim]Small counts: the p-value is rough. Do more runs.[/dim]")


@app.command("label-export")
def label_export(
    config: Path = ConfigArg,
    db: Path = DbOpt,
    out: Path = typer.Option(Path("evals/labels.jsonl"), "--out", help="JSONL file to write"),
    run_id: int | None = typer.Option(None, "--run", help="Run id (default: latest)"),
    limit: int | None = typer.Option(None, "--limit", help="Only export the first N answers"),
) -> None:
    """Write answers to a JSONL file for hand-labelling (pre-filled with a rule-based guess)."""
    cfg = _load(config)
    store = _open_store(db)
    try:
        rid = run_id or store.latest_run_id(cfg.project)
        if rid is None:
            console.print(f"[red]No runs for project {cfg.project!r}.[/red]")
            raise typer.Exit(1)
        n = export_for_labelling(store, rid, cfg, out, limit)
    finally:
        store.close()
    console.print(f"Wrote {n} answers to {out}. Fix each `gold` list, set \"labelled\": true, "
                  "then run `askrank eval`.")


@app.command("eval")
def eval_cmd(
    config: Path = ConfigArg,
    labels: Path = typer.Argument(..., exists=True, dir_okay=False, help="Labelled JSONL"),
    extractor: str | None = typer.Option(None, "--extractor",
                                         help="rules or openai (default: from config)"),
    show_errors: int = typer.Option(10, "--show-errors", help="How many mistakes to list"),
) -> None:
    """Score an extractor against hand-labelled answers."""
    _load_dotenv()
    cfg = _load(config)
    if extractor:
        cfg = cfg.model_copy(update={"extractor": cfg.extractor.model_copy(
            update={"provider": extractor})})
    items = load_labelled(labels)
    if not items:
        console.print('[red]No lines marked "labelled": true yet.[/red]')
        raise typer.Exit(1)
    try:
        ex = get_extractor(cfg)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    r = evaluate(ex, items, cfg.businesses)
    t = Table(title=f"Extractor eval: {cfg.extractor.provider} on {r.n_answers} labelled answers")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for name, v in (("Precision", r.precision), ("Recall", r.recall), ("F1", r.f1),
                    ("Position accuracy", r.position_accuracy),
                    ("Sentiment accuracy", r.sentiment_accuracy)):
        t.add_row(name, f"{v:.1%}")
    t.add_row("Matched / extra / missed", f"{r.tp} / {r.fp} / {r.fn}")
    console.print(t)
    for e in r.errors[:show_errors]:
        console.print(f"[dim]- {e}[/dim]")


if __name__ == "__main__":
    app()
