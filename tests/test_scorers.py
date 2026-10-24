"""Tests for scorer registry and scorer implementations."""

from pathlib import Path

import pandas as pd
import pytest

from dsbench.sandbox import CodeResult
from dsbench.scorers import get_scorer
from dsbench.scorers.base import ScoreResult
from dsbench.scorers.judge_scorer import JudgeScorer
from dsbench.scorers.metric_scorer import MetricScorer
from dsbench.scorers.numeric_scorer import NumericScorer
from dsbench.scorers.pytest_scorer import PytestScorer
from dsbench.tasks import Task


def _make_task(tmp_path: Path, scorer: str, hidden_files: dict[str, str]) -> Task:
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    for name, content in hidden_files.items():
        (hidden_dir / name).write_text(content)

    return Task(
        id="test_task",
        category="test",
        difficulty="easy",
        scorer=scorer,
        weight=1.0,
        timeout_s=30,
        answer_type="code" if scorer == "pytest" else "text",
        prompt="Test prompt.",
        data_dir=tmp_path / "data",
        hidden_dir=hidden_dir,
    )


def test_get_scorer_returns_instances():
    for name in ["pytest", "metric", "numeric", "judge"]:
        scorer = get_scorer(name)
        assert scorer is not None


def test_get_scorer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown scorer: 'unknown'"):
        get_scorer("unknown")


def test_pytest_scorer_all_pass(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "solution.py").write_text("answer = 42\n")

    test_code = (
        "import os\n"
        "workdir = os.environ['DSBENCH_WORKDIR']\n"
        "def test_solution_exists():\n"
        "    with open(os.path.join(workdir, 'solution.py')) as f:\n"
        "        assert 'answer' in f.read()\n"
    )
    task = _make_task(tmp_path, "pytest", {"test_solution.py": test_code})
    run_result = CodeResult(stdout="", stderr="", returncode=0, timed_out=False)

    scorer = PytestScorer()
    result = scorer.score(task, run_result, workdir, answer_text="")

    assert isinstance(result, ScoreResult)
    assert result.score == 1.0
    assert result.passed is True
    assert result.details["passed"] == 1
    assert result.details["total"] == 1


def test_pytest_scorer_failure(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "solution.py").write_text("answer = 42\n")

    task = _make_task(tmp_path, "pytest", {"test_fail.py": "def test_fail():\n    assert False\n"})
    run_result = CodeResult(stdout="", stderr="", returncode=0, timed_out=False)

    result = PytestScorer().score(task, run_result, workdir, answer_text="")

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["failures"] == 1


def test_pytest_scorer_model_code_failed(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    task = _make_task(tmp_path, "pytest", {"test_dummy.py": "def test_dummy(): pass\n"})
    run_result = CodeResult(stdout="", stderr="error", returncode=1, timed_out=False)

    result = PytestScorer().score(task, run_result, workdir, answer_text="")

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["reason"] == "model_code_failed"


def test_pytest_scorer_model_code_timed_out(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    task = _make_task(tmp_path, "pytest", {"test_dummy.py": "def test_dummy(): pass\n"})
    run_result = CodeResult(stdout="", stderr="", returncode=None, timed_out=True)

    result = PytestScorer().score(task, run_result, workdir, answer_text="")

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["reason"] == "model_code_timed_out"


def test_metric_scorer_accuracy(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    pd.DataFrame({"prediction": [1, 0, 1, 1]}).to_csv(
        workdir / "preds.csv", index=False
    )

    metric_yaml = (
        "predictions_file: preds.csv\n"
        "labels_file: labels.csv\n"
        "metric: accuracy\n"
        "threshold: 0.9\n"
        "higher_is_better: true\n"
    )
    labels = pd.DataFrame({"target": [1, 0, 1, 0]})
    task = _make_task(tmp_path, "metric", {"metric.yaml": metric_yaml, "labels.csv": labels.to_csv(index=False)})

    result = MetricScorer().score(task, None, workdir, answer_text="")

    assert result.score == pytest.approx(0.75 / 0.9)
    assert result.passed is False
    assert result.details["metric"] == "accuracy"
    assert result.details["value"] == pytest.approx(0.75)


def test_metric_scorer_recall_at_k(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    preds = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "item_id": [10, 20, 30, 10, 20, 30],
        "rank": [1, 2, 3, 1, 2, 3],
    })
    labels = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 20, 30],
    })
    preds.to_csv(workdir / "preds.csv", index=False)

    metric_yaml = (
        "predictions_file: preds.csv\n"
        "labels_file: labels.csv\n"
        "metric: recall_at_k\n"
        "k: 3\n"
        "threshold: 1.0\n"
        "higher_is_better: true\n"
    )
    task = _make_task(tmp_path, "metric", {"metric.yaml": metric_yaml, "labels.csv": labels.to_csv(index=False)})

    result = MetricScorer().score(task, None, workdir, answer_text="")

    assert result.score == pytest.approx(1.0)
    assert result.passed is True
    assert result.details["value"] == pytest.approx(1.0)


def test_metric_scorer_rmse_lower_is_better(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    pd.DataFrame({"prediction": [2.0, 4.0, 6.0]}).to_csv(workdir / "preds.csv", index=False)
    # targets [1,2,3] -> errors 1,2,3 -> mse=14/3 -> rmse ~2.16
    metric_yaml = (
        "predictions_file: preds.csv\n"
        "labels_file: labels.csv\n"
        "metric: rmse\n"
        "threshold: 2.0\n"
        "higher_is_better: false\n"
    )
    labels = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    task = _make_task(tmp_path, "metric", {"metric.yaml": metric_yaml, "labels.csv": labels.to_csv(index=False)})

    result = MetricScorer().score(task, None, workdir, answer_text="")

    assert result.passed is False
    assert result.score < 1.0
    assert result.details["metric"] == "rmse"


def test_metric_scorer_missing_predictions(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    metric_yaml = (
        "predictions_file: preds.csv\n"
        "labels_file: labels.csv\n"
        "metric: accuracy\n"
        "threshold: 1.0\n"
        "higher_is_better: true\n"
    )
    labels = pd.DataFrame({"target": [1, 0]})
    task = _make_task(tmp_path, "metric", {"metric.yaml": metric_yaml, "labels.csv": labels.to_csv(index=False)})

    result = MetricScorer().score(task, None, workdir, answer_text="")

    assert result.score == 0.0
    assert result.passed is False
    assert "error" in result.details


def test_numeric_scorer_all_correct(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "answer.json").write_text('{"mean": 5.0, "std": 2.0}')

    answer_yaml = "answers:\n  mean: 5.0\n  std: 2.0\nrel_tol: 1e-5\n"
    task = _make_task(tmp_path, "numeric", {"answer.yaml": answer_yaml})

    result = NumericScorer().score(task, None, workdir, answer_text="")

    assert result.score == 1.0
    assert result.passed is True
    assert result.details["per_key"]["mean"]["correct"] is True


def test_numeric_scorer_partial(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "answer.json").write_text('{"mean": 5.0, "std": 9.0}')

    answer_yaml = "answers:\n  mean: 5.0\n  std: 2.0\nrel_tol: 1e-5\n"
    task = _make_task(tmp_path, "numeric", {"answer.yaml": answer_yaml})

    result = NumericScorer().score(task, None, workdir, answer_text="")

    assert result.score == 0.5
    assert result.passed is False
    assert result.details["per_key"]["mean"]["correct"] is True
    assert result.details["per_key"]["std"]["correct"] is False


def test_numeric_scorer_missing_answer_file(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()

    answer_yaml = "answers:\n  mean: 5.0\nrel_tol: 1e-5\n"
    task = _make_task(tmp_path, "numeric", {"answer.yaml": answer_yaml})

    result = NumericScorer().score(task, None, workdir, answer_text="")

    assert result.score == 0.0
    assert result.passed is False
    assert "error" in result.details


def test_judge_scorer_uses_judge(tmp_path):
    class FakeJudge:
        def score(self, prompt, answer_text, rubric):
            return {"score": 0.75, "reasoning": "Good."}

    rubric = "Rate correctness."
    task = _make_task(tmp_path, "judge", {"rubric.md": rubric})

    result = JudgeScorer().score(task, None, tmp_path / "work", answer_text="42", judge=FakeJudge())

    assert result.score == pytest.approx(0.75)
    assert result.passed is True
    assert result.details["reasoning"] == "Good."


def test_judge_scorer_requires_judge(tmp_path):
    task = _make_task(tmp_path, "judge", {"rubric.md": "Rate correctness."})

    with pytest.raises(ValueError, match="Judge scorer requires a judge instance"):
        JudgeScorer().score(task, None, tmp_path / "work", answer_text="42")
