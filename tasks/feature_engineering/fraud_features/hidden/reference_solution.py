"""Reference solution: point-in-time features in UTC with strict windows."""

import numpy as np
import pandas as pd

DAY = 86400

tx = pd.read_csv("transactions.csv", dtype=str, keep_default_na=False)
tx = tx.drop_duplicates("transaction_id").reset_index(drop=True)


def to_utc(row):
    ts = pd.Timestamp(row["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize(row["tz"] or "UTC")
    return int(ts.tz_convert("UTC").timestamp())


tx["ts"] = tx.apply(to_utc, axis=1)
tx["amount"] = tx["amount"].astype(float)
tx["spend"] = tx["status"] != "reversed"

cust = pd.read_csv("customers.csv")
cust["signup_ts"] = (pd.to_datetime(cust["signup_date"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
cust["updated_ts"] = (pd.to_datetime(cust["updated_at"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
cust = cust.sort_values("updated_ts")

cb = pd.read_csv("chargebacks.csv").drop_duplicates("transaction_id")
cb = cb.merge(tx[["transaction_id", "customer_id"]], on="transaction_id")
cb["cb_ts"] = (pd.to_datetime(cb["chargeback_date"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
cb_by = {k: np.sort(v["cb_ts"].to_numpy()) for k, v in cb.groupby("customer_id")}

out = []
for cid, g in tx.groupby("customer_id"):
    ts, amt = g["ts"].to_numpy(), g["amount"].to_numpy()
    spend, mer = g["spend"].to_numpy(), g["merchant_id"].to_numpy()
    c = cust[cust["customer_id"] == cid]
    cbs = cb_by.get(cid, np.array([]))
    for i, tid in enumerate(g["transaction_id"]):
        t = ts[i]
        before = ts < t
        in24 = before & (ts >= t - DAY)
        in7 = before & (ts >= t - 7 * DAY)
        in30 = before & (ts >= t - 30 * DAY) & spend
        prior = amt[in30]
        mean = prior.mean() if len(prior) else np.nan
        sd = prior.std(ddof=1) if len(prior) >= 2 else np.nan
        z = (amt[i] - mean) / sd if sd and sd > 0 else np.nan
        valid = c[c["updated_ts"] <= t]
        signup = (valid.iloc[-1] if len(valid) else c.iloc[0])["signup_ts"]
        out.append((
            tid, int(in24.sum()), amt[in7 & spend].sum(), mean, len(set(mer[in7])), z,
            (t - signup) / DAY, int((cbs < t).sum()), int(not (before & (mer == mer[i])).any()),
        ))

cols = ["transaction_id", "txn_count_24h", "amount_sum_7d", "amount_mean_30d",
        "distinct_merchants_7d", "amount_zscore_30d", "days_since_signup",
        "prior_chargebacks", "is_first_txn_at_merchant"]
pd.DataFrame(out, columns=cols).sort_values("transaction_id").to_csv("features.csv", index=False)

with open("notes.md", "w") as f:
    f.write("""# Leakage notes

- Time zones: timestamps use naive local time with a tz column, ISO offsets or Z. All are converted to UTC before any window.
- Windows are time based (24h, 7d, 30d) and use only strictly earlier transactions of the same customer.
- Same timestamp ties: transactions with the identical timestamp do not see each other, so no tie leaks future information.
- Unsorted file: the rows are not in time order, so the data is sorted by UTC time and never windowed in file order.
- Duplicate rows: repeated transaction_id rows are dropped before any count.
- Chargebacks are label leakage: prior_chargebacks counts only chargebacks with chargeback_date before the current timestamp, not all chargebacks of the customer.
- Signup: customers.csv holds corrected signup_date rows with a later updated_at. I use the row valid at scoring time.
- Reversed transactions are events, so they count in txn_count and merchants, but they are not spend, so they are excluded from amounts.
- zscore: NaN when there are fewer than 2 prior spend rows or the std is 0, so there is no divide by zero.
""")
