"""End-to-end runner tests with fake solvers."""

import json
from pathlib import Path

import pytest

from dsbench.factory import default_solver_factory
from dsbench.runner import run_benchmark
from dsbench.sandbox import run_code
from dsbench.solvers.base import SolveResult, Solver
from dsbench.tasks import load_tasks

from .conftest import build_numeric_task


class PerfectSolver(Solver):
    """Fake solver that runs the task's hidden reference solution."""

    name = "perfect"

    def solve(self, task, workdir: Path) -> SolveResult:
        code = (task.hidden_dir / "reference_solution.py").read_text()
        code_result = run_code(code, task.data_dir, workdir, task.timeout_s)
        return SolveResult(
            answer_text=code,
            returncode=code_result.returncode if code_result.returncode is not None else -1,
            timed_out=code_result.timed_out,
            duration_s=0.01,
            cost_usd=0.001,
            tokens_in=100,
            tokens_out=50,
            transcript=code,
            meta={"stdout": code_result.stdout, "stderr": code_result.stderr},
        )


class NothingSolver(Solver):
    """Fake solver that writes nothing to workdir."""

    name = "nothing"

    def solve(self, task, workdir: Path) -> SolveResult:
        return SolveResult(answer_text="I did nothing useful.")


def _config(label: str, runs: int = 2) -> dict:
    return {
        "models": [{"label": label, "mode": "api", "adapter": "fake", "model": "m"}],
        "runs_per_task": runs,
    }


def test_perfect_solver_rates_100(task_root, tmp_path):
    tasks = load_tasks(task_root)
    out_dir = tmp_path / "out"

    paths = run_benchmark(
        config=_config("perfect-model"),
        solver_factory=lambda entry: PerfectSolver(),
        tasks=tasks,
        runs_per_task=2,
        out_dir=out_dir,
    )

    assert len(paths) == 1
    results = json.loads(paths[0].read_text())
    assert results["model_label"] == "perfect-model"
    assert results["mode"] == "api"
    assert results["model"] == "m"
    assert results["rating"] == pytest.approx(100.0)
    assert results["per_task"]["task_one"]["mean"] == pytest.approx(1.0)
    assert results["per_task"]["task_one"]["std"] == pytest.approx(0.0)
    assert results["per_task"]["task_two"]["mean"] == pytest.approx(1.0)
    assert results["per_category"] == {"cleaning": 1.0, "stats": 1.0}
    assert results["total_cost_usd"] == pytest.approx(2 * 2 * 0.001)
    assert results["total_duration_s"] > 0


def test_zero_solver_rates_0(tmp_path):
    build_numeric_task(tmp_path, task_id="task_one", category="stats")
    task = load_tasks(tmp_path)[0]
    out_dir = tmp_path / "out"

    paths = run_benchmark(
        config=_config("zero-model"),
        solver_factory=lambda entry: NothingSolver(),
        tasks=[task],
        runs_per_task=1,
        out_dir=out_dir,
    )

    results = json.loads(paths[0].read_text())
    assert results["rating"] == pytest.approx(0.0)
    assert results["per_task"]["task_one"]["mean"] == pytest.approx(0.0)


def test_run_dir_structure_and_no_hidden_copy(task_root, tmp_path):
    tasks = load_tasks(task_root)
    out_dir = tmp_path / "out"

    run_benchmark(
        config=_config("perfect-model"),
        solver_factory=lambda entry: PerfectSolver(),
        tasks=tasks,
        runs_per_task=2,
        out_dir=out_dir,
    )

    model_root = out_dir / "perfect-model"
    (timestamp,) = [p for p in model_root.iterdir() if p.is_dir()]
    run_dir = timestamp / "task_one" / "run_0"

    # Data and prompt are copied; hidden files are never copied.
    assert (run_dir / "prompt.md").is_file()
    assert (run_dir / "values.csv").is_file()
    assert not (run_dir / "hidden").exists()
    assert not (run_dir / "answer.yaml").exists()
    assert not (run_dir / "reference_solution.py").exists()

    # Per-run json and transcript exist.
    record = json.loads((run_dir / "result.json").read_text())
    assert record["task_id"] == "task_one"
    assert record["score"] == pytest.approx(1.0)
    assert record["passed"] is True
    assert record["solve"]["tokens_in"] == 100
    assert record["transcript_file"] == "transcript.md"
    assert (run_dir / "transcript.md").is_file()


def test_factory_supports_fake_api_model(task_root, tmp_path):
    tasks = load_tasks(task_root)
    out_dir = tmp_path / "out"
    config = _config("fake-model", runs=1)

    paths = run_benchmark(
        config=config,
        solver_factory=default_solver_factory,
        tasks=tasks,
        runs_per_task=1,
        out_dir=out_dir,
    )

    results = json.loads(paths[0].read_text())
    # The fake adapter returns text, not valid code, so scores are 0.
    assert results["rating"] == pytest.approx(0.0)
    task_one_runs = json.loads(
        (out_dir / "fake-model" / paths[0].parent.name / "task_one" / "run_0" / "result.json")
        .read_text()
    )
    assert task_one_runs["solve"]["returncode"] != 0


def test_scoring_error_scores_zero_and_continues(task_root, tmp_path):
    tasks = load_tasks(task_root)
    for task in tasks:
        if task.id == "task_one":
            (task.hidden_dir / "answer.yaml").unlink()
    out_dir = tmp_path / "out"

    paths = run_benchmark(
        config=_config("perfect-model"),
        solver_factory=lambda entry: PerfectSolver(),
        tasks=tasks,
        runs_per_task=1,
        out_dir=out_dir,
    )

    results = json.loads(paths[0].read_text())
    assert results["per_task"]["task_one"]["mean"] == pytest.approx(0.0)
    assert results["per_task"]["task_two"]["mean"] == pytest.approx(1.0)
    record = json.loads(
        (out_dir / "perfect-model" / paths[0].parent.name / "task_one" / "run_0" / "result.json")
        .read_text()
    )
    assert "scoring failed" in record["details"]["error"]


def test_run_benchmark_requires_models(tmp_path):
    with pytest.raises(ValueError, match="at least one model"):
        run_benchmark(
            config={"models": []},
            solver_factory=lambda entry: None,
            tasks=[],
            runs_per_task=1,
            out_dir=tmp_path,
        )
