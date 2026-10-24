"""Pure metric functions for the metric scorer."""

import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)


def compute_auc(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Area under the ROC curve; preds must contain a 'score' column."""
    return float(roc_auc_score(labels["target"], preds["score"]))


def compute_accuracy(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Classification accuracy; preds must contain a 'prediction' column."""
    return float(accuracy_score(labels["target"], preds["prediction"]))


def compute_rmse(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Root mean squared error; preds must contain a 'prediction' column."""
    mse = mean_squared_error(labels["target"], preds["prediction"])
    return float(math.sqrt(mse))


def compute_f1(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    """F1 score; preds must contain a 'prediction' column."""
    return float(f1_score(labels["target"], preds["prediction"], zero_division=0))


def compute_recall_at_k(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    k: int,
) -> float:
    """Mean per-user Recall@k.

    preds must contain columns: user_id, item_id, rank.
    labels must contain columns: user_id, item_id.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = preds[preds["rank"] <= k]

    relevant_by_user = labels.groupby("user_id")["item_id"].apply(set).to_dict()
    predicted_by_user = top_k.groupby("user_id")["item_id"].apply(set).to_dict()

    scores: list[float] = []
    for user_id, relevant_items in relevant_by_user.items():
        denominator = min(k, len(relevant_items))
        if denominator == 0:
            continue
        predicted_items = predicted_by_user.get(user_id, set())
        hits = len(predicted_items & relevant_items)
        scores.append(hits / denominator)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
