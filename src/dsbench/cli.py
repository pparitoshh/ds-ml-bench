"""Command line interface for ds-ml-bench."""

from pathlib import Path
from typing import Optional

import typer

from dsbench.factory import default_solver_factory, load_config
from dsbench.leaderboard import build_leaderboard, render_html, render_markdown, write_report_markdown
from dsbench.runner import run_benchmark
from dsbench.tasks import load_tasks

app = typer.Typer(
    help="ds-ml-bench: benchmark LLMs on data-science and ML tasks.",
    no_args_is_help=True,
)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a table with rich if available, else plain text."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
        print(line)  # noqa: T201
        print("  ".join("-" * w for w in widths))  # noqa: T201
        for row in rows:
            print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))  # noqa: T201


@app.command("list-tasks")
def list_tasks(
    tasks_dir: Path = typer.Option(Path("tasks"), "--tasks-dir", help="Task root directory."),
) -> None:
    """List all available benchmark tasks."""
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print(f"No tasks found in {tasks_dir}")  # noqa: T201
        raise typer.Exit(code=1)
    _print_table(
        ["task_id", "category", "difficulty", "scorer", "weight", "answer_type"],
        [
            [
                task.id,
                task.category,
                task.difficulty,
                task.scorer,
                str(task.weight),
                task.answer_type,
            ]
            for task in tasks
        ],
    )


@app.command()
def run(
    config: Path = typer.Option(..., "--config", help="Path to config.yaml."),
    model: str = typer.Option(..., "--model", help="Model label from the config."),
    task: Optional[str] = typer.Option(None, "--task", help="Run a single task id."),
    runs: Optional[int] = typer.Option(None, "--runs", help="Override runs_per_task."),
    tasks_dir: Path = typer.Option(Path("tasks"), "--tasks-dir", help="Task root directory."),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="Results output directory."),
) -> None:
    """Run the benchmark for one model from the config."""
    cfg = load_config(config)

    entries = [
        entry
        for entry in cfg.get("models", [])
        if (entry.get("label") or entry.get("name")) == model
    ]
    if not entries:
        known = [entry.get("label") or entry.get("name") for entry in cfg.get("models", [])]
        print(f"Unknown model label {model!r}. Known labels: {known}")  # noqa: T201
        raise typer.Exit(code=1)
    cfg["models"] = entries

    tasks = load_tasks(tasks_dir)
    if task is not None:
        selected = [candidate for candidate in tasks if candidate.id == task]
        if not selected:
            print(f"Unknown task id {task!r}. Known ids: {[t.id for t in tasks]}")  # noqa: T201
            raise typer.Exit(code=1)
        tasks = selected

    runs_per_task = runs if runs is not None else int(cfg.get("runs_per_task", 1))
    if runs_per_task < 1:
        print(f"runs must be >= 1, got {runs_per_task}")  # noqa: T201
        raise typer.Exit(code=1)

    results_paths = run_benchmark(
        config=cfg,
        solver_factory=default_solver_factory,
        tasks=tasks,
        runs_per_task=runs_per_task,
        out_dir=out_dir,
    )

    for results_path in results_paths:
        print(f"Results written to {results_path}")  # noqa: T201


@app.command()
def leaderboard(
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Results directory."),
) -> None:
    """Print the leaderboard and write leaderboard.md and leaderboard.html."""
    entries = build_leaderboard(results_dir)
    if not entries:
        print(f"No results found under {results_dir}")  # noqa: T201
        raise typer.Exit(code=1)

    _print_table(
        ["model", "mode", "rating", "cost_usd", "date"],
        [
            [
                entry.get("model_label", "unknown"),
                entry.get("mode", "api"),
                f"{float(entry.get('rating', 0.0)):.1f}",
                f"{entry.get('total_cost_usd'):.2f}"
                if isinstance(entry.get("total_cost_usd"), (int, float))
                else "-",
                entry.get("date", "-"),
            ]
            for entry in entries
        ],
    )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / "leaderboard.md"
    html_path = results_dir / "leaderboard.html"
    md_path.write_text(render_markdown(entries))
    html_path.write_text(render_html(entries))
    report_path = write_report_markdown(entries, results_dir)
    print(
        f"Leaderboard written to {md_path} and {html_path}; report written to {report_path}"
    )  # noqa: T201


if __name__ == "__main__":
    app()
