"""Deterministically generate small CSV datasets for the benchmark tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"


def _task_path(category: str, task_id: str, *parts: str) -> Path:
    return TASKS_DIR / category / task_id / Path(*parts)


def make_churn() -> None:
    rng = np.random.default_rng(42)
    n_train = 800
    n_test = 200

    def _make_features(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "x1": np.round(rng.normal(0, 1, size=n), 4),
            "x2": np.round(rng.normal(0, 1, size=n), 4),
            "x3": np.round(rng.normal(0, 1, size=n), 4),
        })

    train_x = _make_features(n_train)
    test_x = _make_features(n_test)

    # True churn driven by x1, x2, x3.
    logits = (
        1.2 * train_x["x1"].values
        - 1.0 * train_x["x2"].values
        + 0.8 * train_x["x3"].values
        + 0.1
    )
    train_y = (rng.random(n_train) < (1 / (1 + np.exp(-logits)))).astype(int)

    test_logits = (
        1.2 * test_x["x1"].values
        - 1.0 * test_x["x2"].values
        + 0.8 * test_x["x3"].values
        + 0.1
    )
    test_y = (rng.random(n_test) < (1 / (1 + np.exp(-test_logits)))).astype(int)

    train = train_x.copy()
    train["id"] = np.arange(1, n_train + 1)
    train["days_since_cancellation"] = np.nan
    churner_mask = train_y == 1
    train.loc[churner_mask, "days_since_cancellation"] = rng.integers(
        1, 365, size=churner_mask.sum()
    )
    train["churn"] = train_y

    test = test_x.copy()
    test["id"] = np.arange(n_train + 1, n_train + n_test + 1)
    test["days_since_cancellation"] = np.nan
    test["churn"] = test_y

    data_dir = _task_path("modeling", "churn_classifier", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    train[["id", "x1", "x2", "x3", "days_since_cancellation", "churn"]].to_csv(
        data_dir / "train.csv", index=False
    )
    test[["id", "x1", "x2", "x3", "days_since_cancellation"]].to_csv(
        data_dir / "test.csv", index=False
    )

    hidden_dir = _task_path("modeling", "churn_classifier", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    test[["id", "churn"]].to_csv(hidden_dir / "labels.csv", index=False)

    # Sound model AUC used for verification; threshold is fixed.
    features = ["x1", "x2", "x3"]
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["churn"])
    proba = model.predict_proba(test[features])[:, 1]
    sound_auc = roc_auc_score(test["churn"], proba)

    threshold = 0.75
    baseline = 0.5
    metric_yaml = (
        f"predictions_file: predictions.csv\n"
        f"labels_file: labels.csv\n"
        f"metric: auc\n"
        f"threshold: {threshold}\n"
        f"baseline: {baseline}\n"
        f"higher_is_better: true\n"
    )
    (hidden_dir / "metric.yaml").write_text(metric_yaml)
    print(f"churn_classifier sound AUC: {sound_auc:.4f}")


def make_ab_test() -> None:
    rows = [
        {"segment": "A", "variant": "control", "users": 1000, "conversions": 100},
        {"segment": "A", "variant": "treatment", "users": 500, "conversions": 40},
        {"segment": "B", "variant": "control", "users": 500, "conversions": 25},
        {"segment": "B", "variant": "treatment", "users": 1000, "conversions": 80},
    ]
    df = pd.DataFrame(rows)

    data_dir = _task_path("statistics", "ab_test_simpson", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "ab_data.csv", index=False)

    def _lift(segment_df: pd.DataFrame) -> float:
        ctrl = segment_df[segment_df["variant"] == "control"]
        treat = segment_df[segment_df["variant"] == "treatment"]
        ctrl_rate = ctrl["conversions"].sum() / ctrl["users"].sum()
        treat_rate = treat["conversions"].sum() / treat["users"].sum()
        return float((treat_rate - ctrl_rate) / ctrl_rate)

    overall = _lift(df)
    seg_a = _lift(df[df["segment"] == "A"])
    seg_b = _lift(df[df["segment"] == "B"])

    hidden_dir = _task_path("statistics", "ab_test_simpson", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    answer_yaml = (
        f"answers:\n"
        f"  overall_lift: {overall}\n"
        f"  segment_a_lift: {seg_a}\n"
        f"  segment_b_lift: {seg_b}\n"
        f"rel_tol: 0.01\n"
    )
    (hidden_dir / "answer.yaml").write_text(answer_yaml)


def main() -> None:
    make_churn()
    make_ab_test()
    print("Generated task data.")


if __name__ == "__main__":
    main()
