"""Generate the data for the hard fraud_features task.

Run with: uv run python scripts/gen_fraud_features.py

Outputs:
  data/transactions.csv, customers.csv, chargebacks.csv   visible to the model
  hidden/expected_features.csv    expected answer plus helper columns
  hidden/events_utc.csv           clean UTC events, used by the no-future-data test
  hidden/trap_ids.json            transaction ids per trap, used by the tests

Definitions (the reference solution and the tests use the same ones):
  time       every timestamp is converted to UTC. Windows are time based.
  window     for a transaction at time t, the history is every other
             transaction of the same customer with t - W <= ts < t. Strictly
             earlier only, so transactions with the same timestamp do not
             see each other.
  dedupe     rows with the same transaction_id are dropped before counting.
  reversed   status "reversed" is an event, but it is not spend. It counts in
             txn_count_24h, distinct_merchants_7d and is_first_txn_at_merchant.
             It is excluded from amount_sum_7d, amount_mean_30d and zscore.
             The alt_ columns hold the other convention; tests accept either.
  zscore     (amount - mean) / std of the prior 30 days of spend. NaN when
             there are fewer than 2 prior spend rows or the std is 0.
  signup     days_since_signup uses the signup_date row with the latest
             updated_at that is not later than the transaction.
  chargeback prior_chargebacks counts the customer's chargebacks with
             chargeback_date (UTC) strictly before the transaction time.

Planted traps (each is checked with a variant that ignores the rule):
  tz        mixed zones: naive local time plus tz column, ISO offsets, ISO Z
  same_ts   several transactions of one customer at the same instant
  unsorted  the file is shuffled
  dups      repeated transaction_id rows
  chargeback chargebacks reported days after the transaction
  signup    customers with a later corrected signup_date row
  reversed  status "reversed" rows
  zscore_few  fewer than 2 prior spend rows
"""

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "tasks" / "feature_engineering" / "fraud_features"
SEED = 11
N_CUST = 1000
TARGET_TXNS = 20000
START = pd.Timestamp("2024-01-01", tz="UTC")
SPAN_DAYS = 182
DAY = 86400
ZONES = [
    "UTC", "America/New_York", "Europe/Berlin", "Asia/Kolkata", "Asia/Tokyo",
    "America/Sao_Paulo", "Asia/Dubai", "Africa/Lagos", "Asia/Singapore",
    "Europe/London", "America/Los_Angeles",
]


# ---- feature computation ----------------------------------------------------

def compute(ev, cb, cust, *, strict=True, file_order=False, reversed_mode="ref",
            chargeback="before", signup="valid", ddof=1):
    """Features for every row of ev. Options turn individual rules off.

    ev columns: transaction_id, customer_id, merchant_id, amount, status, ts,
    pos. ts is integer UTC seconds. cb has transaction_id, cb_ts.
    reversed_mode: "ref" (not spend, still counted) or "alt" (spend, not counted).
    """
    cb_ts = cb.merge(ev[["transaction_id", "customer_id"]].drop_duplicates("transaction_id"),
                     on="transaction_id").groupby("customer_id")["cb_ts"].apply(
        lambda s: np.sort(s.to_numpy()))
    out = pd.DataFrame(index=ev.index, columns=[
        "txn_count_24h", "amount_sum_7d", "amount_mean_30d", "distinct_merchants_7d",
        "amount_zscore_30d", "days_since_signup", "prior_chargebacks",
        "is_first_txn_at_merchant", "n_spend_30d"], dtype=float)
    for cid, g in ev.groupby("customer_id"):
        ts = g["ts"].to_numpy()
        amt = g["amount"].to_numpy(dtype=float)
        rev = (g["status"] == "reversed").to_numpy()
        mer = g["merchant_id"].to_numpy()
        pos = g["pos"].to_numpy()
        if reversed_mode == "ref":
            spend, counted = ~rev, np.ones(len(g), bool)
        else:
            spend, counted = np.ones(len(g), bool), ~rev
        cbs = cb_ts.get(cid, np.array([], dtype=np.int64))
        crow = cust[cust["customer_id"] == cid].sort_values("updated_at")
        rows = []
        for i in range(len(g)):
            t = ts[i]
            before = (ts < t) if strict else ((ts <= t) & (np.arange(len(g)) != i))
            if file_order:
                before &= pos < pos[i]
            def win(sec, mask):
                return before & mask & (ts >= t - sec)
            m24 = win(DAY, counted)
            m7 = win(7 * DAY, spend)
            m30 = win(30 * DAY, spend)
            mm7 = win(7 * DAY, counted)
            n30 = int(m30.sum())
            a30 = amt[m30]
            mean = a30.mean() if n30 else np.nan
            sd = a30.std(ddof=ddof) if n30 >= 2 else np.nan
            z = (amt[i] - mean) / sd if n30 >= 2 and sd > 0 else np.nan
            if chargeback == "before":
                ncb = int((cbs < t).sum())
            else:
                ncb = len(cbs)
            valid = crow[crow["updated_at"] <= t]
            if signup == "latest":
                row = crow.iloc[-1]
            elif signup == "orig" or valid.empty:
                row = crow.iloc[0]
            else:
                row = valid.iloc[-1]
            days = (t - row["signup_ts"]) / DAY
            first = not bool((before & counted & (mer == mer[i])).any())
            rows.append((m24.sum(), amt[m7].sum(), mean, len(set(mer[mm7])), z, days,
                         ncb, float(first), n30))
        out.loc[g.index] = rows
    return out


# ---- data generation --------------------------------------------------------

def make_customers(rng):
    cust = []
    for i in range(N_CUST):
        zones = [rng.choice(ZONES)]
        if rng.random() < 0.35:
            zones.append(rng.choice(ZONES))
        cust.append(dict(customer_id=f"C{i:04d}", zones=zones,
                         n=int(np.clip(rng.lognormal(2.7, 0.8), 4, 140)),
                         mu=rng.normal(3.6, 0.7), cb_rate=rng.beta(0.4, 12)))
    scale = TARGET_TXNS / sum(c["n"] for c in cust)
    for c in cust:
        c["n"] = max(4, int(round(c["n"] * scale)))
    return cust


def gen_events(rng, cust):
    merchants = [f"M{i:03d}" for i in range(300)]
    ev, k = [], 0
    for c in cust:
        favs = rng.choice(merchants, 8, replace=False)
        n, times = c["n"], []
        while len(times) < n:
            t0 = int(rng.integers(0, SPAN_DAYS * DAY))
            for _ in range(int(rng.integers(1, 5))):
                times.append(t0)
                t0 += int(rng.exponential(2500)) + 1
        times = [min(t, SPAN_DAYS * DAY - 1) for t in times[:n]]
        for t in times:
            mer = rng.choice(favs) if rng.random() < 0.7 else rng.choice(merchants)
            ev.append(dict(customer_id=c["customer_id"], ts=int(START.timestamp()) + t,
                           merchant_id=mer, amount=round(float(rng.lognormal(c["mu"], 0.8)), 2),
                           status="reversed" if rng.random() < 0.05 else "completed",
                           zone=c["zones"][int(rng.integers(len(c["zones"])))]))
    # Same-instant twins: another transaction of the same customer at the same second.
    base = list(ev)
    for j in rng.choice(len(base), 260, replace=False):
        e = dict(base[j])
        e["merchant_id"] = rng.choice(merchants) if rng.random() < 0.5 else e["merchant_id"]
        e["amount"] = round(float(rng.lognormal(3.6, 0.8)), 2)
        e["status"] = "completed"
        if rng.random() < 0.5:
            e["zone"] = rng.choice(ZONES)
        ev.append(e)
    ids = rng.permutation(len(ev))
    for e, i in zip(ev, ids):
        e["transaction_id"] = f"T{100000 + i}"
    return pd.DataFrame(ev)


def format_time(rng, ts, zone):
    """Return (timestamp string, tz string, wall-clock epoch)."""
    inst = pd.Timestamp(ts, unit="s", tz="UTC")
    zi = ZoneInfo(zone)
    loc = inst.tz_convert(zi)
    naive = loc.tz_localize(None)
    # Skip ambiguous wall-clock times (DST fall back): shift to a safe instant.
    a = naive.to_pydatetime().replace(tzinfo=zi, fold=0).utcoffset()
    b = naive.to_pydatetime().replace(tzinfo=zi, fold=1).utcoffset()
    if a != b:
        return None
    r = rng.random()
    wall = int(naive.tz_localize("UTC").timestamp())
    if r < 0.5:
        return naive.strftime("%Y-%m-%d %H:%M:%S"), zone, wall
    if r < 0.9:
        return loc.isoformat(), "", wall
    return inst.strftime("%Y-%m-%dT%H:%M:%SZ"), "", int(inst.timestamp())


def epoch(col):
    return (pd.to_datetime(col) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")


def main():
    rng = np.random.default_rng(SEED)
    cust = make_customers(rng)
    ev = gen_events(rng, cust)

    ts_str, tz_str, wall = [], [], []
    for r in ev.itertuples():
        res = format_time(rng, r.ts, r.zone)
        while res is None:
            res = format_time(rng, r.ts, "UTC")
        ts_str.append(res[0]); tz_str.append(res[1]); wall.append(res[2])
    ev["timestamp"], ev["tz"], ev["wall"] = ts_str, tz_str, wall

    # Customers: original row plus a corrected, earlier signup_date row for some.
    ev_first = ev.groupby("customer_id")["ts"].min()
    ev_last = ev.groupby("customer_id")["ts"].max()
    crow = []
    for c in cust:
        cid = c["customer_id"]
        signup = pd.Timestamp(int(ev_first[cid]) - int(rng.integers(20, 400)) * DAY, unit="s")
        signup = signup.normalize()
        crow.append(dict(customer_id=cid, signup_date=signup.strftime("%Y-%m-%d"),
                         updated_at=signup.strftime("%Y-%m-%d %H:%M:%S"),
                         country=str(rng.choice(["US", "DE", "IN", "JP", "BR", "AE", "NG", "SG", "GB"]))))
        if rng.random() < 0.25:
            lo, hi = int(ev_first[cid]), int(ev_last[cid])
            upd = int(rng.integers(lo + (hi - lo) // 4, lo + 3 * (hi - lo) // 4 + 1))
            new = (signup - pd.Timedelta(days=int(rng.integers(30, 150)))).strftime("%Y-%m-%d")
            crow.append(dict(customer_id=cid, signup_date=new,
                             updated_at=pd.Timestamp(upd, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                             country=crow[-1]["country"]))
    cust_df = pd.DataFrame(crow).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Chargebacks: reported 3 to 60 days after the transaction (UTC timestamp).
    rate = {c["customer_id"]: c["cb_rate"] for c in cust}
    cb_rows = []
    for r in ev.itertuples():
        if r.status == "completed" and rng.random() < rate[r.customer_id]:
            delay = int(rng.integers(3 * DAY, 60 * DAY))
            cb_rows.append(dict(transaction_id=r.transaction_id,
                                chargeback_date=pd.Timestamp(r.ts + delay, unit="s").strftime("%Y-%m-%d %H:%M:%S")))
    cb_df = pd.DataFrame(cb_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Visible file: shuffled, with duplicated rows.
    vis = ev.copy()
    dups = vis.sample(450, random_state=SEED)
    vis = pd.concat([vis, dups]).sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    cols = ["transaction_id", "customer_id", "timestamp", "tz", "merchant_id", "amount", "status"]

    # Expected output on the clean (deduped) frame.
    clean = vis.drop_duplicates("transaction_id").copy()
    clean["pos"] = np.arange(len(clean))
    cust_df["signup_ts"] = epoch(cust_df["signup_date"])
    cust_df["updated_at"] = epoch(cust_df["updated_at"])
    cb = cb_df.copy()
    cb["cb_ts"] = epoch(cb["chargeback_date"])

    ref = compute(clean, cb, cust_df)
    alt = compute(clean, cb, cust_df, reversed_mode="alt")
    pop = compute(clean, cb, cust_df, ddof=0)
    exp = ref.copy()
    exp["transaction_id"] = clean["transaction_id"]
    exp["_n_spend_30d"] = ref["n_spend_30d"]
    exp = exp.drop(columns="n_spend_30d")
    exp["_zscore_pop"] = pop["amount_zscore_30d"]
    for col in ["txn_count_24h", "amount_sum_7d", "amount_mean_30d", "distinct_merchants_7d",
                "amount_zscore_30d", "is_first_txn_at_merchant"]:
        exp["_alt_" + col] = alt[col]
    exp["_alt_zscore_pop"] = compute(clean, cb, cust_df, reversed_mode="alt", ddof=0)["amount_zscore_30d"]
    exp["_signup_orig"] = compute(clean, cb, cust_df, signup="orig")["days_since_signup"]
    first = ["transaction_id", "txn_count_24h", "amount_sum_7d", "amount_mean_30d",
             "distinct_merchants_7d", "amount_zscore_30d", "days_since_signup",
             "prior_chargebacks", "is_first_txn_at_merchant"]
    exp = exp[first + [c for c in exp.columns if c not in first]]

    # Trap subsets: rows where ignoring the rule changes a feature.
    def differs(variant, cols_, tol=1e-6):
        a, b = ref[cols_], variant[cols_]
        d = ((a - b).abs() > tol) | (a.isna() != b.isna())
        return clean.loc[d.any(axis=1), "transaction_id"].tolist()

    win_cols = ["txn_count_24h", "amount_sum_7d", "distinct_merchants_7d"]
    wall_ev = clean.assign(ts=clean["wall"])
    trap = {
        "tz": differs(compute(wall_ev, cb, cust_df), win_cols),
        "same_ts": differs(compute(clean, cb, cust_df, strict=False),
                           win_cols + ["is_first_txn_at_merchant"]),
        "unsorted": differs(compute(clean, cb, cust_df, file_order=True),
                            ["txn_count_24h", "amount_sum_7d", "amount_mean_30d"]),
        "chargeback": differs(compute(clean, cb, cust_df, chargeback="all"), ["prior_chargebacks"]),
        "signup": differs(compute(clean, cb, cust_df, signup="latest"), ["days_since_signup"], 1.0),
        "reversed": differs(alt, ["amount_sum_7d", "amount_mean_30d"]),
        "zscore_few": clean.loc[ref["n_spend_30d"] < 2, "transaction_id"].tolist(),
    }
    # Duplicates: rows whose window holds a repeated transaction (the repeat counted twice).
    dup_ids = set(vis.loc[vis.duplicated("transaction_id"), "transaction_id"])
    nodedupe = vis.copy()
    nodedupe["pos"] = np.arange(len(nodedupe))
    nd = compute(nodedupe, cb, cust_df)
    nd = nd.loc[~vis.duplicated("transaction_id", keep="first")]
    nd.index = clean.index
    trap["dups"] = differs(nd, ["txn_count_24h", "amount_sum_7d", "amount_mean_30d"])
    trap["dup_rows"] = sorted(dup_ids)

    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "hidden").mkdir(parents=True, exist_ok=True)
    vis[cols].to_csv(ROOT / "data" / "transactions.csv", index=False)
    cust_out = pd.DataFrame(crow).sample(frac=1, random_state=SEED).reset_index(drop=True)
    cust_out[["customer_id", "signup_date", "updated_at", "country"]].to_csv(
        ROOT / "data" / "customers.csv", index=False)
    cb_df.to_csv(ROOT / "data" / "chargebacks.csv", index=False)
    exp.sort_values("transaction_id").to_csv(
        ROOT / "hidden" / "expected_features.csv", index=False, float_format="%.6f")
    ev_out = clean[["transaction_id", "customer_id", "merchant_id", "amount", "status", "ts"]].copy()
    ev_out = ev_out.merge(cb[["transaction_id", "cb_ts"]], on="transaction_id", how="left")
    ev_out.to_csv(ROOT / "hidden" / "events_utc.csv", index=False)
    (ROOT / "hidden" / "trap_ids.json").write_text(json.dumps(trap, indent=1))
    print(f"raw rows {len(vis)}, unique {len(clean)}, chargebacks {len(cb_df)}")
    print({k: len(v) for k, v in trap.items()})


if __name__ == "__main__":
    main()
