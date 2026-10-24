"""Tests for pure metric functions."""

import math

import pandas as pd
import pytest

from dsbench.scorers.metrics import (
    compute_accuracy,
    compute_auc,
    compute_f1,
    compute_recall_at_k,
    compute_rmse,
)


def test_compute_auc_perfect():
    preds = pd.DataFrame({"score": [0.1, 0.2, 0.8, 0.9]})
    labels = pd.DataFrame({"target": [0, 0, 1, 1]})
    assert compute_auc(preds, labels) == pytest.approx(1.0)


def test_compute_accuracy():
    preds = pd.DataFrame({"prediction": [1, 0, 1, 1, 0]})
    labels = pd.DataFrame({"target": [1, 0, 0, 1, 0]})
    assert compute_accuracy(preds, labels) == pytest.approx(0.8)


def test_compute_rmse():
    preds = pd.DataFrame({"prediction": [3.0, -0.5, 2.0, 7.0]})
    labels = pd.DataFrame({"target": [2.5, 0.0, 2.0, 8.0]})
    expected = math.sqrt(0.375)
    assert compute_rmse(preds, labels) == pytest.approx(expected)


def test_compute_f1():
    preds = pd.DataFrame({"prediction": [1, 0, 1, 0, 1]})
    labels = pd.DataFrame({"target": [1, 0, 0, 1, 1]})
    # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3 -> f1=2/3
    assert compute_f1(preds, labels) == pytest.approx(2 / 3)


def test_compute_recall_at_k_basic():
    preds = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "item_id": [10, 20, 30, 10, 20, 30],
        "rank": [1, 2, 3, 1, 2, 3],
    })
    labels = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 20, 30],
    })
    # user 1: top3 {10,20,30}, relevant {10,20}, denom 2 -> 2/2
    # user 2: top3 {10,20,30}, relevant {20,30}, denom 2 -> 2/2
    assert compute_recall_at_k(preds, labels, k=3) == pytest.approx(1.0)


def test_compute_recall_at_k_fewer_relevant_than_k():
    preds = pd.DataFrame({
        "user_id": [1, 1, 1],
        "item_id": [10, 30, 40],
        "rank": [1, 2, 3],
    })
    labels = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
    })
    # relevant size 2, k=3, denom=2, hits=1 -> 0.5
    assert compute_recall_at_k(preds, labels, k=3) == pytest.approx(0.5)


def test_compute_recall_at_k_user_with_no_predictions():
    preds = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "rank": [1],
    })
    labels = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 99],
    })
    # user 1: 1/1 = 1.0; user 2: no predictions -> 0/1 = 0
    assert compute_recall_at_k(preds, labels, k=3) == pytest.approx(0.5)


def test_compute_recall_at_k_only_ranked_items_count():
    preds = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_id": [10, 20, 30, 40],
        "rank": [1, 2, 3, 4],
    })
    labels = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_id": [10, 20, 30, 40],
    })
    # k=2, denom=2, hits=2 -> 1.0
    assert compute_recall_at_k(preds, labels, k=2) == pytest.approx(1.0)


def test_compute_recall_at_k_no_relevant_items():
    preds = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "rank": [1],
    })
    labels = pd.DataFrame({
        "user_id": [],
        "item_id": [],
    })
    assert compute_recall_at_k(preds, labels, k=3) == 0.0


def test_compute_recall_at_k_invalid_k():
    preds = pd.DataFrame({"user_id": [], "item_id": [], "rank": []})
    labels = pd.DataFrame({"user_id": [], "item_id": []})
    with pytest.raises(ValueError, match="k must be positive"):
        compute_recall_at_k(preds, labels, k=0)
