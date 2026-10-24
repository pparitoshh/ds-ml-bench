"""Task loader and Task dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_SCORERS = {"pytest", "metric", "numeric", "judge"}
VALID_ANSWER_TYPES = {"code", "text"}


@dataclass
class Task:
    """A single benchmark task."""

    id: str
    category: str
    difficulty: str
    scorer: str
    weight: float
    timeout_s: int
    answer_type: str
    prompt: str
    data_dir: Path
    hidden_dir: Path


def _require(meta: dict[str, Any], key: str, task_file: Path) -> Any:
    if key not in meta:
        raise ValueError(f"{task_file}: missing required field '{key}'")
    return meta[key]


def _validate_enum(
    value: str,
    key: str,
    allowed: set[str],
    task_file: Path,
) -> None:
    if value not in allowed:
        raise ValueError(
            f"{task_file}: '{key}' must be one of {sorted(allowed)}, got {value!r}"
        )


def load_tasks(root: Path | str) -> list[Task]:
    """Load and validate all tasks under root."""
    root = Path(root)
    tasks: list[Task] = []

    for task_file in sorted(root.rglob("task.yaml")):
        task_dir = task_file.parent
        meta = yaml.safe_load(task_file.read_text()) or {}

        task_id = _require(meta, "id", task_file)
        category = _require(meta, "category", task_file)
        difficulty = _require(meta, "difficulty", task_file)
        scorer = _require(meta, "scorer", task_file)
        weight = _require(meta, "weight", task_file)
        timeout_s = _require(meta, "timeout_s", task_file)
        answer_type = _require(meta, "answer_type", task_file)

        _validate_enum(scorer, "scorer", VALID_SCORERS, task_file)
        _validate_enum(answer_type, "answer_type", VALID_ANSWER_TYPES, task_file)

        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"{task_file}: 'weight' must be numeric, got {type(weight).__name__}"
            )
        if not isinstance(timeout_s, int):
            raise ValueError(
                f"{task_file}: 'timeout_s' must be an integer, got {type(timeout_s).__name__}"
            )

        prompt_file = task_dir / "prompt.md"
        if not prompt_file.is_file():
            raise FileNotFoundError(f"{task_file}: missing prompt file {prompt_file}")

        data_dir = task_dir / "data"
        hidden_dir = task_dir / "hidden"

        tasks.append(
            Task(
                id=task_id,
                category=category,
                difficulty=difficulty,
                scorer=scorer,
                weight=float(weight),
                timeout_s=int(timeout_s),
                answer_type=answer_type,
                prompt=prompt_file.read_text(),
                data_dir=data_dir,
                hidden_dir=hidden_dir,
            )
        )

    return tasks
