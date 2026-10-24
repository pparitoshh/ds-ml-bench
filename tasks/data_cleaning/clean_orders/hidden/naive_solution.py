"""Naive solution: follows the visible prompt quickly, no data exploration."""

import re

import pandas as pd

df = pd.read_csv("orders.csv", dtype=str)
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed").dt.strftime("%Y-%m-%d")
df["amount_usd"] = pd.to_numeric(
    df["amount"].map(lambda s: re.sub(r"[^0-9.\-]", "", str(s)) or None)
)
df.loc[df["currency"].str.strip().str.lower() == "cents", "amount_usd"] /= 100
df = df.drop_duplicates().dropna(subset=["amount_usd"])
df["amount_usd"] = df["amount_usd"].round(2)
df["country"] = df["country"].fillna("UNKNOWN").str.strip().str.upper()
df["is_refund"] = df["amount_usd"] < 0
df["order_id"] = df["order_id"].astype(int)
df[["order_id", "order_date", "amount_usd", "country", "is_refund"]].to_csv(
    "clean_orders.csv", index=False
)
