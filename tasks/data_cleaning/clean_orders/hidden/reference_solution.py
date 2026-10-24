import pandas as pd

df = pd.read_csv("orders.csv")

# Normalize dates.
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=False)
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")

# Normalize currency and convert cents to dollars.
df["currency"] = df["currency"].str.strip().str.lower()
cents_mask = df["currency"] == "cents"
df.loc[cents_mask, "amount"] = df.loc[cents_mask, "amount"] / 100
df["amount"] = df["amount"].round(2)

# Normalize country.
df["country"] = df["country"].str.strip().str.upper()

# Drop duplicates and keep only required columns.
df = df.drop_duplicates().reset_index(drop=True)
df = df[["order_id", "order_date", "amount", "country"]]

df.to_csv("clean_orders.csv", index=False)
