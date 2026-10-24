"""Hidden tests for the rolling_features_no_leak task."""

import os
from pathlib import Path

import pandas as pd


WORKDIR = Path(os.environ["DSBENCH_WORKDIR"])


def test_rolling_features_match_expected():
    actual_path = WORKDIR / "features.csv"
    expected_path = Path(__file__).parent / "expected_features.csv"

    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)

    actual["date"] = pd.to_datetime(actual["date"])
    expected["date"] = pd.to_datetime(expected["date"])

    actual = actual.sort_values(["customer_id", "date", "amount"]).reset_index(drop=True)
    expected = expected.sort_values(["customer_id", "date", "amount"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_like=True,
    )


def test_rolling_features_no_leakage_for_first_transaction():
    df = pd.read_csv(WORKDIR / "features.csv")
    first = df.groupby("customer_id").head(1)
    assert (first["rolling_count_7d"] == 0).all()
    assert (first["rolling_spend_7d"] == 0.0).all()
