"""Hidden tests for the fraud_features task. One test per trap or rule."""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path(os.environ["DSBENCH_WORKDIR"])
HERE = Path(__file__).parent
MIN_SHARE = 0.97
FEATURES = ["txn_count_24h", "amount_sum_7d", "amount_mean_30d", "distinct_merchants_7d",
            "amount_zscore_30d", "days_since_signup", "prior_chargebacks",
            "is_first_txn_at_merchant"]
DAY = 86400


def _expected():
    return pd.read_csv(HERE / "expected_features.csv").set_index("transaction_id")


def _actual():
    df = pd.read_csv(WORKDIR / "features.csv", dtype=str, keep_default_na=False)
    df = df.drop_duplicates("transaction_id").set_index("transaction_id")
    for c in FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"True": "1", "False": "0", "true": "1", "false": "0"}),
                                  errors="coerce")
    return df


def _ids(key):
    return json.loads((HERE / "trap_ids.json").read_text())[key]


def _close(a, b, tol=0.011, rel=1e-6):
    """Elementwise closeness; NaN equals NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    same_nan = np.isnan(a) & np.isnan(b)
    return same_nan | (np.abs(a - b) <= np.maximum(tol, rel * np.abs(b)))


def _share(ids, col, alts=(), tol=0.011, rel=1e-6, nan_ok_zero=False, min_rows=10):
    """Share of ids whose actual value matches the expected value or any alternative column."""
    exp, act = _expected(), _actual()
    assert col in act.columns, f"missing column {col}"
    ids = [i for i in ids if i in exp.index]
    assert len(ids) >= min_rows, "test subset too small"
    a = act[col].reindex(ids).to_numpy(float)
    ok = _close(a, exp.loc[ids, col].to_numpy(), tol, rel)
    for alt in alts:
        ok |= _close(a, exp.loc[ids, alt].to_numpy(), tol, rel)
    if nan_ok_zero:
        ok |= np.isnan(a) | (a == 0)
    share = ok.mean()
    assert share >= MIN_SHARE, f"{col}: only {share:.1%} of {len(ids)} rows correct"


def _alts(col):
    return ["_alt_" + col]


def test_schema():
    df = pd.read_csv(WORKDIR / "features.csv")
    assert list(df.columns[: len(FEATURES) + 1]) == ["transaction_id"] + FEATURES
    assert df["transaction_id"].is_unique
    for c in FEATURES:
        assert not np.isinf(pd.to_numeric(df[c], errors="coerce")).any(), f"{c} has inf"


def test_one_row_per_transaction():
    act = pd.read_csv(WORKDIR / "features.csv")
    assert len(act) == len(_expected())
    assert set(act["transaction_id"]) == set(_expected().index)


def test_count_24h_mixed_time_zones():
    _share(_ids("tz"), "txn_count_24h", _alts("txn_count_24h"))


def test_sum_7d_mixed_time_zones():
    _share(_ids("tz"), "amount_sum_7d", _alts("amount_sum_7d"))


def test_distinct_merchants_mixed_time_zones():
    _share(_ids("tz"), "distinct_merchants_7d", _alts("distinct_merchants_7d"))


def test_same_timestamp_not_counted():
    _share(_ids("same_ts"), "txn_count_24h", _alts("txn_count_24h"))


def test_same_timestamp_sum_and_first_at_merchant():
    ids = _ids("same_ts")
    _share(ids, "amount_sum_7d", _alts("amount_sum_7d"), min_rows=5)
    _share(ids, "is_first_txn_at_merchant", _alts("is_first_txn_at_merchant"), min_rows=5)


def test_unsorted_input_count_and_sum():
    ids = _ids("unsorted")
    _share(ids, "txn_count_24h", _alts("txn_count_24h"))
    _share(ids, "amount_sum_7d", _alts("amount_sum_7d"))


def test_unsorted_input_mean_30d():
    _share(_ids("unsorted"), "amount_mean_30d", _alts("amount_mean_30d"), nan_ok_zero=True)


def test_duplicates_dropped_before_counting():
    ids = _ids("dups")
    _share(ids, "txn_count_24h", _alts("txn_count_24h"))
    _share(ids, "amount_sum_7d", _alts("amount_sum_7d"))


def test_duplicates_in_mean_30d():
    _share(_ids("dups"), "amount_mean_30d", _alts("amount_mean_30d"), nan_ok_zero=True)


def test_future_chargebacks_not_counted():
    _share(_ids("chargeback"), "prior_chargebacks")


def test_prior_chargebacks_overall():
    exp = _expected()
    _share(list(exp.index), "prior_chargebacks")


def test_signup_row_valid_at_scoring_time():
    # Accept the original signup_date too (the other documented choice).
    _share(_ids("signup"), "days_since_signup", ["_signup_orig"], tol=1.0)


def test_reversed_not_counted_as_spend():
    # Tolerant on the convention: spend excluding or including reversed rows.
    _share(_ids("reversed"), "amount_sum_7d", _alts("amount_sum_7d"))


def test_zscore_values():
    exp = _expected()
    ids = exp.index[exp["_n_spend_30d"] >= 2].tolist()
    ids = [i for i in ids if np.isfinite(exp.loc[i, "amount_zscore_30d"])]
    _share(ids, "amount_zscore_30d", ["_zscore_pop", "_alt_amount_zscore_30d", "_alt_zscore_pop"],
           tol=1e-3, rel=1e-3)


def test_zscore_few_prior_is_nan_or_zero():
    act = _actual()
    ids = [i for i in _ids("zscore_few") if i in act.index]
    vals = act.loc[ids, "amount_zscore_30d"].to_numpy(float)
    assert len(ids) >= 10
    assert len(act) > 0 and not np.isinf(vals).any()
    assert (np.isnan(vals) | (vals == 0)).mean() >= MIN_SHARE


def test_first_txn_at_merchant_overall():
    exp = _expected()
    _share(list(exp.index), "is_first_txn_at_merchant", _alts("is_first_txn_at_merchant"))


def test_no_feature_uses_future_data():
    """Recompute a sample from the clean UTC events using a strict cutoff."""
    ev = pd.read_csv(HERE / "events_utc.csv")
    act = _actual()
    sample = ev.sample(300, random_state=3)
    ok = {c: 0 for c in ["txn_count_24h", "amount_sum_7d", "distinct_merchants_7d", "prior_chargebacks"]}
    for r in sample.itertuples():
        h = ev[(ev["customer_id"] == r.customer_id) & (ev["ts"] < r.ts)]
        w24, w7 = h[h["ts"] >= r.ts - DAY], h[h["ts"] >= r.ts - 7 * DAY]
        got = act.loc[r.transaction_id]
        ok["txn_count_24h"] += got["txn_count_24h"] in (len(w24), (w24["status"] != "reversed").sum())
        sums = {round(w7["amount"].sum(), 2), round(w7.loc[w7["status"] != "reversed", "amount"].sum(), 2)}
        ok["amount_sum_7d"] += any(abs(got["amount_sum_7d"] - s) <= 0.011 for s in sums)
        d = {w7["merchant_id"].nunique(), w7.loc[w7["status"] != "reversed", "merchant_id"].nunique()}
        ok["distinct_merchants_7d"] += got["distinct_merchants_7d"] in d
        own = ev[ev["customer_id"] == r.customer_id]
        ok["prior_chargebacks"] += got["prior_chargebacks"] == int((own["cb_ts"] < r.ts).sum())
    for col, n in ok.items():
        assert n / len(sample) >= MIN_SHARE, f"{col}: only {n / len(sample):.1%} match a strict cutoff"


def test_notes_cover_leakage_risks():
    path = WORKDIR / "notes.md"
    assert path.exists(), "notes.md missing"
    text = path.read_text().lower()
    groups = [
        r"time ?zone|utc|\btz\b|offset",
        r"chargeback|label leak",
        r"same timestamp|identical timestamp|tie|strictly (earlier|before)|simultaneous",
        r"duplicate|dedup",
        r"signup|updated_at|valid at|as of|point-in-time",
        r"reversed|cancel",
        r"sort|order|look-?ahead|future",
    ]
    hits = sum(bool(re.search(g, text)) for g in groups)
    assert hits >= 4, f"notes cover only {hits} of {len(groups)} leakage groups"
    assert len(text) > 300
