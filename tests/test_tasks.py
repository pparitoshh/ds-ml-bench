"""Tests for task loading and validation."""

import pytest

from dsbench.tasks import Task, load_tasks


@pytest.fixture
def task_root(tmp_path):
    root = tmp_path / "tasks"
    task_dir = root / "stats" / "mean_price"
    task_dir.mkdir(parents=True)

    (task_dir / "task.yaml").write_text(
        "id: mean_price\n"
        "category: stats\n"
        "difficulty: easy\n"
        "scorer: numeric\n"
        "weight: 1.0\n"
        "timeout_s: 30\n"
        "answer_type: code\n"
    )
    (task_dir / "prompt.md").write_text("Compute the mean price.")
    (task_dir / "data").mkdir()
    (task_dir / "data" / "prices.csv").write_text("10,20,30\n")
    (task_dir / "hidden").mkdir()
    (task_dir / "hidden" / "answer.json").write_text('{"mean": 20}')

    return root


def test_load_tasks_returns_one_task(task_root):
    tasks = load_tasks(task_root)
    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, Task)
    assert task.id == "mean_price"
    assert task.category == "stats"
    assert task.difficulty == "easy"
    assert task.scorer == "numeric"
    assert task.weight == 1.0
    assert task.timeout_s == 30
    assert task.answer_type == "code"
    assert task.prompt == "Compute the mean price."
    assert task.data_dir == task_root / "stats" / "mean_price" / "data"
    assert task.hidden_dir == task_root / "stats" / "mean_price" / "hidden"


def test_load_tasks_validates_missing_field(task_root):
    task_file = task_root / "stats" / "mean_price" / "task.yaml"
    content = task_file.read_text()
    task_file.write_text(content.replace("weight: 1.0\n", ""))

    with pytest.raises(ValueError, match="missing required field 'weight'"):
        load_tasks(task_root)


def test_load_tasks_validates_bad_scorer(task_root):
    task_file = task_root / "stats" / "mean_price" / "task.yaml"
    content = task_file.read_text()
    task_file.write_text(content.replace("scorer: numeric", "scorer: human"))

    with pytest.raises(ValueError, match="'scorer' must be one of"):
        load_tasks(task_root)


def test_load_tasks_validates_bad_answer_type(task_root):
    task_file = task_root / "stats" / "mean_price" / "task.yaml"
    content = task_file.read_text()
    task_file.write_text(content.replace("answer_type: code", "answer_type: json"))

    with pytest.raises(ValueError, match="'answer_type' must be one of"):
        load_tasks(task_root)


def test_load_tasks_missing_prompt(task_root):
    (task_root / "stats" / "mean_price" / "prompt.md").unlink()

    with pytest.raises(FileNotFoundError, match="missing prompt file"):
        load_tasks(task_root)
