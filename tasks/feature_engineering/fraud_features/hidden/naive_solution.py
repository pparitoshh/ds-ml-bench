"""Naive solution: row-based rolling on file order, tz ignored, chargebacks joined by customer."""

import pandas as pd

tx = pd.read_csv("transactions.csv")
tx["t"] = pd.to_datetime(tx["timestamp"].str[:19].str.replace("T", " "))
cust = pd.read_csv("customers.csv").drop_duplicates("customer_id", keep="last")
cb = pd.read_csv("chargebacks.csv").merge(tx[["transaction_id", "customer_id"]], on="transaction_id")
cb_n = cb.groupby("customer_id").size()

g = tx.groupby("customer_id")
tx["txn_count_24h"] = g["amount"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).count()).fillna(0)
tx["amount_sum_7d"] = g["amount"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
tx["amount_mean_30d"] = g["amount"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
tx["distinct_merchants_7d"] = g["merchant_id"].transform(
    lambda s: pd.Series([len(set(s.iloc[max(0, i - 10):i])) for i in range(len(s))], index=s.index)
)
std = g["amount"].transform(lambda s: s.shift(1).rolling(30, min_periods=2).std())
tx["amount_zscore_30d"] = (tx["amount"] - tx["amount_mean_30d"]) / std
tx = tx.merge(cust[["customer_id", "signup_date"]], on="customer_id", how="left")
tx["days_since_signup"] = (tx["t"] - pd.to_datetime(tx["signup_date"])).dt.days
tx["prior_chargebacks"] = tx["customer_id"].map(cb_n).fillna(0).astype(int)
tx["is_first_txn_at_merchant"] = (tx.groupby(["customer_id", "merchant_id"]).cumcount() == 0).astype(int)
cols = ["transaction_id", "txn_count_24h", "amount_sum_7d", "amount_mean_30d",
        "distinct_merchants_7d", "amount_zscore_30d", "days_since_signup",
        "prior_chargebacks", "is_first_txn_at_merchant"]
tx.drop_duplicates("transaction_id")[cols].to_csv("features.csv", index=False)
open("notes.md", "w").write("Used rolling windows per customer.\n")
