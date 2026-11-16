"""Command-line interface.

    askrank validate examples/brackenfell-gp.yaml
    askrank run examples/brackenfell-gp.yaml --fake      # free offline demo
    askrank run examples/brackenfell-gp.yaml --runs 3    # real API calls
    askrank report examples/brackenfell-gp.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from askrank.config import Config, ExtractorConfig, ProviderConfig, load_config
from askrank.extract import get_extractor
from askrank.providers import get_provider
from askrank.report import render, summarise
from askrank.runner import run as run_pipeline
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
            "providers": [ProviderConfig(name="fake", model="fake-1")],
            "extractor": ExtractorConfig(provider="rules"),
        })
    n = runs or cfg.runs_per_query
    calls = len(cfg.queries) * len(cfg.providers) * n
    if not fake and not yes:
        typer.confirm(f"This makes about {calls} paid API answer calls. Continue?", abort=True)

    try:
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
    if not db.exists():
        console.print(f"[red]No database at {db}. Run `askrank run` first.[/red]")
        raise typer.Exit(1)
    store = Store(db)
    try:
        rid = run_id or store.latest_run_id(cfg.project)
        if rid is None:
            console.print(f"[red]No runs for project {cfg.project!r} in {db}.[/red]")
            raise typer.Exit(1)
        render(summarise(store, rid, cfg), console)
    finally:
        store.close()


if __name__ == "__main__":
    app()
