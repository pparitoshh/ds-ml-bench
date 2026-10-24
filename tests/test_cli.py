"""CLI tests using typer's CliRunner."""

import json
from pathlib import Path

from typer.testing import CliRunner

from dsbench.cli import app

from .conftest import build_numeric_task

runner = CliRunner()


def _write_results(results_dir: Path, label: str, date: str, rating: float) -> None:
    data = {
        "model_label": label,
        "mode": "api",
        "date": date,
        "per_category": {"stats": rating / 100.0},
        "rating": rating,
        "total_cost_usd": 0.0,
    }
    out_dir = results_dir / label / date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(data))


def test_list_tasks(tmp_path):
    build_numeric_task(tmp_path, task_id="mean_price", category="stats")
    build_numeric_task(tmp_path, task_id="clean_stuff", category="cleaning")

    result = runner.invoke(app, ["list-tasks", "--tasks-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "mean_price" in result.stdout
    assert "clean_stuff" in result.stdout
    assert "numeric" in result.stdout


def test_list_tasks_empty_dir_fails(tmp_path):
    result = runner.invoke(app, ["list-tasks", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_run_with_fake_model(tmp_path):
    task_root = tmp_path / "tasks"
    build_numeric_task(task_root, task_id="task_one", category="stats")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "models:\n  - label: fake-model\n    mode: api\n    adapter: fake\n"
        "runs_per_task: 1\n"
    )
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_file),
            "--model",
            "fake-model",
            "--tasks-dir",
            str(task_root),
            "--out-dir",
            str(out_dir),
            "--runs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Results written to" in result.stdout
    results_files = list(out_dir.glob("*/*/results.json"))
    assert len(results_files) == 1
    data = json.loads(results_files[0].read_text())
    assert data["model_label"] == "fake-model"
    assert data["rating"] == 0.0


def test_run_unknown_model_fails(tmp_path):
    task_root = tmp_path / "tasks"
    build_numeric_task(task_root, task_id="task_one", category="stats")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "models:\n  - label: known\n    mode: api\n    adapter: fake\n"
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_file),
            "--model",
            "unknown",
            "--tasks-dir",
            str(task_root),
        ],
    )

    assert result.exit_code == 1


def test_run_single_task_filter(tmp_path):
    task_root = tmp_path / "tasks"
    build_numeric_task(task_root, task_id="task_one", category="stats")
    build_numeric_task(task_root, task_id="task_two", category="cleaning")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "models:\n  - label: known\n    mode: api\n    adapter: fake\n"
    )
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_file),
            "--model",
            "known",
            "--task",
            "task_two",
            "--tasks-dir",
            str(task_root),
            "--out-dir",
            str(out_dir),
            "--runs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "task_one" not in result.stdout
    run_dirs = list(out_dir.glob("known/*/task_two/*"))
    assert [p.name for p in run_dirs] == ["run_0"]
    assert run_dirs[0].parent.name == "task_two"


def test_leaderboard_writes_files(tmp_path):
    results_dir = tmp_path / "results"
    _write_results(results_dir, "model-a", "2026-01-01T00-00-00", 40.0)
    _write_results(results_dir, "model-b", "2026-01-01T00-00-00", 95.0)

    result = runner.invoke(
        app, ["leaderboard", "--results-dir", str(results_dir)]
    )

    assert result.exit_code == 0, result.stdout
    md_path = results_dir / "leaderboard.md"
    html_path = results_dir / "leaderboard.html"
    assert md_path.is_file()
    assert html_path.is_file()
    md_text = md_path.read_text()
    assert "model-b" in md_text
    assert "95.0" in md_text
    assert "<table>" in html_path.read_text()


def test_leaderboard_empty_results(tmp_path):
    result = runner.invoke(
        app, ["leaderboard", "--results-dir", str(tmp_path / "none")]
    )
    assert result.exit_code == 1
