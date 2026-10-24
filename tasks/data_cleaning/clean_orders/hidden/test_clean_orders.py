"""Hidden tests for the clean_orders task."""

import os
from pathlib import Path

import pandas as pd


WORKDIR = Path(os.environ["DSBENCH_WORKDIR"])


def test_clean_orders_matches_expected():
    actual_path = WORKDIR / "clean_orders.csv"
    expected_path = Path(__file__).parent / "expected_clean_orders.csv"

    actual = pd.read_csv(actual_path, parse_dates=["order_date"])
    expected = pd.read_csv(expected_path, parse_dates=["order_date"])

    actual = actual.sort_values("order_id").reset_index(drop=True)
    expected = expected.sort_values("order_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_clean_orders_columns_and_types():
    df = pd.read_csv(WORKDIR / "clean_orders.csv")
    assert list(df.columns) == ["order_id", "order_date", "amount", "country"]
    assert df["order_id"].dtype.kind == "i"
    assert df["amount"].dtype.kind == "f"
    assert df["country"].str.match(r"^[A-Z]+$").all()
