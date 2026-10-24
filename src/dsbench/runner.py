"""Benchmark runner."""

import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from dsbench.scorers import get_scorer
from dsbench.scorers.base import ScoreResult
from dsbench.solvers.base import SolveResult, Solver


def _make_judge(config: dict):
    """Build a judge instance from the config, or return None."""
    judge_cfg = config.get("judge")
    if not judge_cfg:
        return None
    from dsbench.judge.claude_judge import ClaudeJudge

    return ClaudeJudge(model=judge_cfg.get("model"))


def _copy_task_inputs(task, workdir: Path) -> None:
    """Copy task data files and prompt.md into workdir (never hidden/)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if task.data_dir.exists():
        for item in task.data_dir.iterdir():
            dest = workdir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    prompt_dest = workdir / "prompt.md"
    if task.prompt is not None and not prompt_dest.exists():
        prompt_dest.write_text(task.prompt)


class _ScorerRunView:
    """View of a SolveResult for scorers that expect a CodeResult."""

    def __init__(self, solve_result: SolveResult):
        self.returncode = solve_result.returncode
        self.timed_out = solve_result.timed_out
        meta = solve_result.meta or {}
        self.stdout = meta.get("stdout", "")
        self.stderr = meta.get("stderr", "")


def _mean(scores: list[float]) -> float:
    return statistics.fmean(scores) if scores else 0.0


def _std(scores: list[float]) -> float:
    return statistics.pstdev(scores) if len(scores) > 1 else 0.0


def run_benchmark(
    config: dict,
    solver_factory,
    tasks: list,
    runs_per_task: int,
    out_dir: Path,
) -> list[Path]:
    """Run the benchmark for every model in the config.

    Args:
        config: Parsed benchmark configuration dict with a 'models' list.
        solver_factory: Callable that maps a model entry to a Solver.
        tasks: List of Task objects to run.
        runs_per_task: Number of runs per task.
        out_dir: Root output directory for results.

    Returns:
        List of paths to the written results.json files, one per model.
    """
    models = config.get("models") or []
    if not models:
        raise ValueError("config must define at least one model under 'models'")

    judge = _make_judge(config)
    out_dir = Path(out_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    results_paths: list[Path] = []
    for model_entry in models:
        label = model_entry.get("label") or model_entry.get("name")
        if not label:
            raise ValueError("each model entry needs a 'label' (or 'name')")
        solver: Solver = solver_factory(model_entry)
        results_path = _run_one_model(
            solver=solver,
            label=label,
            model_entry=model_entry,
            tasks=tasks,
            runs_per_task=runs_per_task,
            out_dir=out_dir,
            timestamp=timestamp,
            judge=judge,
        )
        results_paths.append(results_path)

    return results_paths


def _run_one_model(
    solver: Solver,
    label: str,
    model_entry: dict,
    tasks: list,
    runs_per_task: int,
    out_dir: Path,
    timestamp: str,
    judge,
) -> Path:
    """Run one model over all tasks and write its results.json."""
    model_root = Path(out_dir) / label / timestamp
    task_meta: dict[str, dict] = {}
    total_cost = 0.0
    total_duration = 0.0

    for task in tasks:
        task_dir = model_root / task.id
        scores: list[float] = []
        run_records: list[dict] = []

        for run_index in range(runs_per_task):
            run_dir = task_dir / f"run_{run_index}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            _copy_task_inputs(task, run_dir)
            try:
                solve_result = solver.solve(task, run_dir)
                scoring_error = None
            except Exception as exc:  # noqa: BLE001
                solve_result = SolveResult(
                    answer_text="",
                    returncode=-1,
                    meta={"error": f"solver failed: {exc}"},
                )
                scoring_error = exc

            try:
                scorer = get_scorer(task.scorer)
                score_result = scorer.score(
                    task=task,
                    run_result=_ScorerRunView(solve_result),
                    workdir=run_dir,
                    answer_text=solve_result.answer_text,
                    judge=judge,
                )
            except Exception as exc:  # noqa: BLE001
                score_result = ScoreResult(
                    score=0.0,
                    passed=False,
                    details={"error": f"scoring failed: {exc}"},
                )
                if scoring_error is not None:
                    score_result.details["solver_error"] = str(scoring_error)

            transcript_file = run_dir / "transcript.md"
            transcript_file.write_text(solve_result.transcript or "")

            record = {
                "task_id": task.id,
                "run": run_index,
                "score": score_result.score,
                "passed": score_result.passed,
                "details": _jsonable(score_result.details),
                "solve": {
                    "answer_text": solve_result.answer_text,
                    "returncode": solve_result.returncode,
                    "timed_out": solve_result.timed_out,
                    "duration_s": solve_result.duration_s,
                    "cost_usd": solve_result.cost_usd,
                    "tokens_in": solve_result.tokens_in,
                    "tokens_out": solve_result.tokens_out,
                    "meta": _jsonable(solve_result.meta),
                },
                "transcript_file": transcript_file.name,
            }
            (run_dir / "result.json").write_text(json.dumps(record, indent=2))
            run_records.append(record)

            if solve_result.cost_usd:
                total_cost += solve_result.cost_usd
            total_duration += solve_result.duration_s

        scores = [record["score"] for record in run_records]
        task_meta[task.id] = {
            "category": task.category,
            "difficulty": task.difficulty,
            "weight": task.weight,
            "scorer": task.scorer,
            "mean": _mean(scores),
            "std": _std(scores),
            "runs": run_records,
        }

    results = _build_summary(
        label=label,
        model_entry=model_entry,
        task_meta=task_meta,
        total_cost=total_cost,
        total_duration=total_duration,
        runs_per_task=runs_per_task,
    )

    model_root.mkdir(parents=True, exist_ok=True)
    results_path = model_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    return results_path


def _build_summary(
    label: str,
    model_entry: dict,
    task_meta: dict[str, dict],
    total_cost: float,
    total_duration: float,
    runs_per_task: int,
) -> dict:
    """Build the summary results dict for one model."""
    category_scores: dict[str, list[float]] = {}
    for meta in task_meta.values():
        category_scores.setdefault(meta["category"], []).append(meta["mean"])

    per_category = {
        category: _mean(scores) for category, scores in sorted(category_scores.items())
    }

    total_weight = sum(meta["weight"] for meta in task_meta.values())
    if total_weight > 0:
        weighted = sum(
            meta["mean"] * meta["weight"] for meta in task_meta.values()
        ) / total_weight
    else:
        weighted = 0.0

    return {
        "model_label": label,
        "mode": model_entry.get("mode", "api"),
        "harness": model_entry.get("harness"),
        "model": model_entry.get("model"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S"),
        "runs_per_task": runs_per_task,
        "per_task": {
            task_id: {
                "category": meta["category"],
                "difficulty": meta["difficulty"],
                "weight": meta["weight"],
                "scorer": meta["scorer"],
                "mean": meta["mean"],
                "std": meta["std"],
            }
            for task_id, meta in task_meta.items()
        },
        "per_category": per_category,
        "rating": weighted * 100.0,
        "total_cost_usd": total_cost,
        "total_duration_s": total_duration,
    }


def _jsonable(value):
    """Convert a value into JSON-serializable data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)
