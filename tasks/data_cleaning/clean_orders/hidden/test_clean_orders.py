"""Hidden tests for the clean_orders task. One test per trap or rule."""

import json
import os
import re
from pathlib import Path

import pandas as pd

WORKDIR = Path(os.environ["DSBENCH_WORKDIR"])
HERE = Path(__file__).parent
MIN_SHARE = 0.97


def _expected():
    df = pd.read_csv(HERE / "expected_clean_orders.csv", keep_default_na=False)
    return df.rename(columns=lambda c: c.lstrip("_"))


def _actual():
    df = pd.read_csv(WORKDIR / "clean_orders.csv", keep_default_na=False, dtype=str)
    df["amount_num"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
    return df


def _ids():
    return json.loads((HERE / "trap_ids.json").read_text())


def _share_correct(subset, column, check):
    """Share of expected rows in subset that have a matching actual row."""
    assert len(subset) >= 10, "test subset too small"
    actual = _actual().drop_duplicates("order_id").set_index("order_id")
    ok = 0
    for row in subset.itertuples():
        if row.order_id in actual.index and check(actual.loc[row.order_id], row):
            ok += 1
    share = ok / len(subset)
    assert share >= MIN_SHARE, f"{column}: only {share:.1%} of {len(subset)} rows correct"


def _amount_ok(act, row):
    return abs(act["amount_num"] - row.amount_usd) <= 0.011


def test_schema():
    df = _actual()
    assert list(df.columns[:5]) == ["order_id", "order_date", "amount_usd", "country", "is_refund"]
    assert df["order_id"].notna().all()
    assert df["order_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert df["amount_num"].notna().all()
    assert (df["amount_num"].round(2) == df["amount_num"]).all()
    assert df["country"].str.fullmatch(r"[A-Z]{2}|UNKNOWN").all()
    assert df["is_refund"].str.lower().isin(["true", "false"]).all()
    assert len(df) > 0


def test_dates_us_slash_format():
    exp = _expected()
    sub = exp[exp["date_fmt"] == "us_slash"]
    _share_correct(sub, "date", lambda a, r: a["order_date"] == r.order_date)


def test_dates_eu_dotted_format():
    exp = _expected()
    sub = exp[exp["date_fmt"] == "eu_dot"]
    _share_correct(sub, "date", lambda a, r: a["order_date"] == r.order_date)


def test_symbols_and_codes_in_usd_amounts():
    exp = _expected()
    sub = exp[exp["amt_fmt"] == "usd_decorated"]
    _share_correct(sub, "amount", _amount_ok)


def test_european_number_format():
    exp = _expected()
    sub = exp[(exp["amt_fmt"] == "eur") & (exp["local"] > 0)]
    # Parsing only: the result must be within a plausible EUR/USD factor of the local amount.
    _share_correct(
        sub, "amount", lambda a, r: 0.8 <= a["amount_num"] / r.local <= 1.4
    )


def test_cents_currency_divided_by_100():
    exp = _expected()
    sub = exp[exp["amt_fmt"] == "cents"]
    _share_correct(sub, "amount", _amount_ok)


def test_fx_conversion_on_order_date():
    exp = _expected()
    sub = exp[exp["cur"].isin(["EUR", "GBP"]) & (exp["amt_fmt"] != "refund") & ~exp["fx_filled"]]
    _share_correct(sub, "fx", _amount_ok)


def test_fx_weekend_uses_last_available_rate():
    exp = _expected()
    sub = exp[exp["cur"].isin(["EUR", "GBP"]) & (exp["amt_fmt"] != "refund") & exp["fx_filled"]]
    _share_correct(sub, "fx weekend", _amount_ok)


def test_refunds_negative_and_flagged():
    exp, act = _expected(), _actual()
    flagged = act["is_refund"].str.lower() == "true"
    assert flagged.sum() == exp["is_refund"].sum()
    assert (act.loc[flagged, "amount_num"] < 0).all()
    assert (act.loc[~flagged, "amount_num"] > 0).all()


def test_refunds_not_double_counted():
    exp, act = _expected(), _actual()
    flagged = act["is_refund"].str.lower() == "true"
    expected_sum = exp.loc[exp["is_refund"], "amount_usd"].sum()
    assert abs(act.loc[flagged, "amount_num"].sum() - expected_sum) <= 0.01 * abs(expected_sum)


def test_test_orders_removed():
    act = _actual()
    assert len(_ids()["test"]) >= 10
    assert not act["order_id"].isin(_ids()["test"]).any()


def test_exact_duplicates_removed():
    act = _actual()
    ids = _ids()["exact_dup"]
    counts = act[act["order_id"].isin(ids)]["order_id"].value_counts()
    assert len(counts) >= 0.97 * len(ids)
    assert (counts == 1).all()


def test_whitespace_case_duplicates_removed():
    act = _actual()
    ids = _ids()["fuzzy_dup"]
    counts = act[act["order_id"].isin(ids)]["order_id"].value_counts()
    assert len(counts) >= 0.97 * len(ids)
    assert (counts == 1).all()


def test_latest_version_kept():
    exp = _expected()
    sub = exp[exp["order_id"].isin(_ids()["version"])]
    _share_correct(sub, "version", _amount_ok)


def test_country_aliases_mapped_to_iso2():
    exp = _expected()
    sub = exp[exp["alias"] & ~exp["null_country"]]
    _share_correct(sub, "country", lambda a, r: a["country"] == r.country)


def test_cents_bug_outlier_fixed():
    exp = _expected()
    sub = exp[exp["amt_fmt"] == "usd_bug"]
    _share_correct(sub, "outlier", _amount_ok)


def test_null_amounts_dropped():
    act = _actual()
    assert not act["order_id"].isin(_ids()["null_amount"]).any()
    assert len(act) > 0.5 * len(_expected())


def test_null_country_is_unknown():
    exp = _expected()
    sub = exp[exp["null_country"]]
    _share_correct(sub, "country", lambda a, r: a["country"] == "UNKNOWN")


def test_monthly_net_revenue_by_country():
    exp, act = _expected(), _actual()
    exp["month"] = exp["order_date"].str[:7]
    act["month"] = act["order_date"].str[:7]
    e = exp.groupby(["month", "country"])["amount_usd"].sum()
    a = act.groupby(["month", "country"])["amount_num"].sum()
    for key, value in e.items():
        assert key in a.index, f"missing {key}"
        assert abs(a[key] - value) <= max(0.005 * abs(value), 1.0), f"{key}: {a[key]} vs {value}"


NOTE_GROUPS = [
    ["date", "format"],
    ["currency", "fx", "exchange", "rate"],
    ["refund"],
    ["duplicate", "dedup"],
    ["test order", "example.com", "test"],
    ["country", "iso"],
    ["outlier", "cents bug", "100x", "100 times"],
    ["null", "missing", "empty"],
    ["cents"],
    ["latest", "updated_at", "version"],
]


def test_notes_md_mentions_issues():
    path = WORKDIR / "notes.md"
    assert path.exists()
    text = path.read_text().lower()
    hits = sum(any(k in text for k in group) for group in NOTE_GROUPS)
    assert hits >= 5, f"only {hits} issue keywords found"
