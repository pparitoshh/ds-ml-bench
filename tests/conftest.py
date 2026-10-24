"""Shared fixtures for building tiny benchmark tasks."""

from pathlib import Path

import pytest


def build_numeric_task(
    root: Path,
    task_id: str = "mean_price",
    category: str = "stats",
    weight: float = 1.0,
) -> Path:
    """Create a numeric-scorer task whose reference solution scores 1.0."""
    task_dir = root / category / task_id
    (task_dir / "data").mkdir(parents=True, exist_ok=True)
    (task_dir / "hidden").mkdir(parents=True, exist_ok=True)

    (task_dir / "task.yaml").write_text(
        f"id: {task_id}\n"
        f"category: {category}\n"
        "difficulty: easy\n"
        "scorer: numeric\n"
        f"weight: {weight}\n"
        "timeout_s: 30\n"
        "answer_type: code\n"
    )
    (task_dir / "prompt.md").write_text(
        f"# {task_id}\n\nCompute the mean of values.csv and write answer.json.\n"
    )
    (task_dir / "data" / "values.csv").write_text("10\n20\n30\n40\n50\n60\n")
    (task_dir / "hidden" / "answer.yaml").write_text(
        "answers:\n  mean: 35.0\nrel_tol: 1e-5\n"
    )
    (task_dir / "hidden" / "reference_solution.py").write_text(
        "import json\n"
        "values = [float(line) for line in open('values.csv')]\n"
        "json.dump({'mean': sum(values) / len(values)}, open('answer.json', 'w'))\n"
    )
    return task_dir


@pytest.fixture
def task_root(tmp_path):
    """A root with two perfect tasks in different categories."""
    build_numeric_task(tmp_path, task_id="task_one", category="stats", weight=1.0)
    build_numeric_task(
        tmp_path, task_id="task_two", category="cleaning", weight=3.0
    )
    return tmp_path
