import numpy as np
import pandas as pd

df = pd.read_csv("transactions.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["customer_id", "date", "amount"]).reset_index(drop=True)

rows = []
for customer_id, group in df.groupby("customer_id", sort=True):
    group = group.sort_values("date").reset_index(drop=True)
    amounts = group["amount"].to_numpy()
    dates = group["date"].to_numpy()
    spend = np.empty(len(group), dtype=float)
    count = np.empty(len(group), dtype=int)
    for i in range(len(group)):
        mask = (dates[i] - dates < np.timedelta64(7, "D")) & (dates < dates[i])
        spend[i] = amounts[mask].sum()
        count[i] = int(mask.sum())
    group["rolling_spend_7d"] = spend
    group["rolling_count_7d"] = count
    rows.append(group)

result = pd.concat(rows, ignore_index=True)
result["date"] = result["date"].dt.strftime("%Y-%m-%d")
result = result[[
    "customer_id",
    "date",
    "amount",
    "rolling_spend_7d",
    "rolling_count_7d",
]]
result.to_csv("features.csv", index=False)
