"""End-to-end tests that run the reference solution for every MVP task."""

from pathlib import Path

import pytest

from dsbench.sandbox import run_code
from dsbench.scorers import get_scorer
from dsbench.tasks import load_tasks


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"

MVP_TASK_IDS = [
    "clean_orders",
    "fraud_features",
    "churn_classifier",
    "ab_test_simpson",
    "movielens_recall_at_10",
]


def test_load_tasks_finds_all_mvp_tasks():
    tasks = load_tasks(TASKS_ROOT)
    found = {task.id for task in tasks}
    assert found >= set(MVP_TASK_IDS)


@pytest.fixture(scope="module")
def task_by_id():
    return {task.id: task for task in load_tasks(TASKS_ROOT)}


def _run_solution(task, solution_code, tmp_path: Path):
    workdir = tmp_path / task.id
    result = run_code(solution_code, task.data_dir, workdir, task.timeout_s)
    assert result.returncode == 0, f"{task.id} reference solution failed: {result.stderr}"
    return result


def _score(task, run_result, tmp_path: Path):
    workdir = tmp_path / task.id
    scorer = get_scorer(task.scorer)
    return scorer.score(task, run_result, workdir, answer_text="")


@pytest.mark.parametrize("task_id", MVP_TASK_IDS)
def test_reference_solution_passes(task_id, task_by_id, tmp_path):
    task = task_by_id[task_id]

    if task_id == "movielens_recall_at_10" and not (task.data_dir / "train.csv").exists():
        pytest.skip("MovieLens data not fetched; run scripts/fetch_movielens.py")

    solution_code = (task.hidden_dir / "reference_solution.py").read_text()
    run_result = _run_solution(task, solution_code, tmp_path)
    score_result = _score(task, run_result, tmp_path)

    assert score_result.score >= 0.99, (
        f"{task_id} score {score_result.score} below 0.99: {score_result.details}"
    )
    assert score_result.passed is True, (
        f"{task_id} did not pass: {score_result.details}"
    )


def test_churn_random_baseline_scores_zero(task_by_id, tmp_path):
    task = task_by_id["churn_classifier"]

    random_code = """
import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")
np.random.seed(0)
proba = np.full(len(test), 0.5)
pd.DataFrame({"id": test["id"], "proba": proba}).to_csv("predictions.csv", index=False)
"""
    run_result = _run_solution(task, random_code, tmp_path)
    score_result = _score(task, run_result, tmp_path)

    assert score_result.score == pytest.approx(0.0)
    assert score_result.passed is False


def test_churn_leaky_baseline_scores_zero(task_by_id, tmp_path):
    task = task_by_id["churn_classifier"]

    leaky_code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["days_since_cancellation"] = train["days_since_cancellation"].fillna(0)
test["days_since_cancellation"] = test["days_since_cancellation"].fillna(0)

model = LogisticRegression(max_iter=1000)
model.fit(train[["days_since_cancellation"]], train["churn"])
proba = model.predict_proba(test[["days_since_cancellation"]])[:, 1]

pd.DataFrame({"id": test["id"], "proba": proba}).to_csv("predictions.csv", index=False)
"""
    run_result = _run_solution(task, leaky_code, tmp_path)
    score_result = _score(task, run_result, tmp_path)

    assert score_result.score == pytest.approx(0.0)
    assert score_result.passed is False
