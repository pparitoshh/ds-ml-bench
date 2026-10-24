"""Reference solution. Every rule is derived from the data, not from the generator.

Findings (also written to notes.md):
- Dates: ISO, MM/DD/YYYY (us_shop) and DD.MM.YYYY (eu_shop). Format follows the source.
- Amounts: symbols and codes, thousands separators. EUR uses "1.234,50"; other currencies use "1,234.50".
- currency "cents": amount is USD cents, divide by 100.
- FX: EUR and GBP convert at the order date. Weekends are often missing: use the last earlier rate.
- Refunds: negative rows and status "refunded" rows both point to the original through refund_of.
  Some refunds appear in both forms. Keep one row per original, amount negative, is_refund True.
- Test orders: email @example.com or country TEST are removed.
- Duplicates: normalise text (strip, lower case), then keep the latest updated_at for each order_id.
- Countries: names and aliases map to ISO-2. Missing country becomes UNKNOWN.
- Cents bug: us_shop rows with currency USD in one date range have amounts 100x too large.
  The daily median of that source jumps to more than 8x the overall median in that range.
  The range is found from the data and the amounts there are divided by 100.
- Missing amount: row is dropped.
"""

import re

import pandas as pd

COUNTRY = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "ca": "CA", "canada": "CA",
    "gb": "GB", "uk": "GB", "great britain": "GB", "united kingdom": "GB",
    "de": "DE", "germany": "DE", "deutschland": "DE",
    "fr": "FR", "france": "FR",
    "es": "ES", "spain": "ES", "españa": "ES",
    "it": "IT", "italy": "IT", "italia": "IT",
}

df = pd.read_csv("orders.csv", dtype=str, keep_default_na=False)
fx = pd.read_csv("fx_rates.csv", parse_dates=["date"])

# Normalise text columns.
for col in ["order_date", "amount", "currency", "country", "customer_email", "status", "refund_of"]:
    df[col] = df[col].str.strip()
df["currency"] = df["currency"].str.lower()
df["status"] = df["status"].str.lower()
df["customer_email"] = df["customer_email"].str.lower()

# Test orders.
is_test = df["customer_email"].str.endswith("@example.com") | (df["country"].str.upper() == "TEST")
df = df[~is_test]

# Duplicates: exact, whitespace/case only, and older versions of the same order_id.
df = df.sort_values("updated_at", kind="stable").drop_duplicates("order_id", keep="last")


def parse_date(s):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return pd.Timestamp(s)
    if "/" in s:
        return pd.to_datetime(s, format="%m/%d/%Y")
    return pd.to_datetime(s, format="%d.%m.%Y")


df["date"] = df["order_date"].map(parse_date)


def parse_amount(s, currency):
    if s == "" or s.upper() == "N/A":
        return float("nan")
    neg = s.startswith("-")
    body = re.sub(r"[^0-9.,]", "", s)
    if currency == "eur":
        body = body.replace(".", "").replace(",", ".")
    else:
        body = body.replace(",", "")
    val = float(body)
    return -val if neg else val


df["amt"] = [parse_amount(a, c) for a, c in zip(df["amount"], df["currency"])]
n_null = int(df["amt"].isna().sum())
df = df.dropna(subset=["amt"]).copy()

# Cents are USD cents.
cents = df["currency"] == "cents"
df.loc[cents, "amt"] = df.loc[cents, "amt"] / 100
df.loc[cents, "currency"] = "usd"

# Cents bug: a date range where the daily median of us_shop/usd is far above normal.
mask = (df["source"] == "us_shop") & (df["currency"] == "usd") & (df["amt"] > 0) & ~cents.reindex(df.index, fill_value=False)
base = df[mask]
daily = base.groupby("date")["amt"].median()
bad_days = daily[daily > 8 * base["amt"].median()].index
if len(bad_days):
    in_range = mask & (df["date"] >= bad_days.min()) & (df["date"] <= bad_days.max())
    df.loc[in_range, "amt"] = df.loc[in_range, "amt"] / 100
    BUG = (bad_days.min().date(), bad_days.max().date(), int(in_range.sum()))
else:
    BUG = None

# FX: last available rate on or before the order date.
df["rate"] = 1.0
for cur in ["eur", "gbp"]:
    r = fx[fx["currency"].str.lower() == cur].sort_values("date")
    sel = df["currency"] == cur
    if sel.any():
        merged = pd.merge_asof(
            df.loc[sel].sort_values("date")[["date"]].reset_index(), r[["date", "rate_to_usd"]], on="date"
        ).set_index("index")
        df.loc[sel, "rate"] = merged["rate_to_usd"]
df["amount_usd"] = df["amt"].abs() * df["rate"]

# Refunds: one row per original. Prefer the negative-amount row.
df["is_refund"] = (df["status"] == "refunded") | (df["amt"] < 0)
refunds = df[df["is_refund"]].copy()
n_refund_raw = len(refunds)
refunds["neg"] = refunds["amt"] < 0
refunds = refunds.sort_values(["refund_of", "neg", "order_id"], ascending=[True, False, True])
refunds = refunds.drop_duplicates("refund_of", keep="first")
refunds["amount_usd"] = -refunds["amount_usd"]
out = pd.concat([df[~df["is_refund"]], refunds])

# Countries.
out["country"] = out["country"].str.lower().str.replace(".", "", regex=False).map(COUNTRY)
out["country"] = out["country"].fillna("UNKNOWN")

out["order_date"] = out["date"].dt.strftime("%Y-%m-%d")
out["amount_usd"] = out["amount_usd"].round(2)
out["order_id"] = out["order_id"].astype(int)
out = out.sort_values("order_id")[["order_id", "order_date", "amount_usd", "country", "is_refund"]]
out.to_csv("clean_orders.csv", index=False)

with open("notes.md", "w") as f:
    f.write(f"""# Data notes

- Date formats: ISO, MM/DD/YYYY for us_shop and DD.MM.YYYY for eu_shop. Parsed by source.
- Amount strings: removed currency symbols and codes; thousands separators; EUR uses comma as decimal.
- Cents: currency "cents" means USD cents; divided by 100.
- Currency conversion (fx): EUR and GBP converted with the rate of the order date; weekend dates missing, used last available rate.
- Refunds: negative rows and status refunded rows reference the original order via refund_of. {n_refund_raw} raw refund rows; one row kept per original, so no double count. Amount is negative, is_refund True.
- Test orders: removed emails ending @example.com and country TEST.
- Duplicates: exact and whitespace/case duplicates removed; for repeated order_id kept the latest updated_at.
- Country: names and aliases (USA, UK, Deutschland) mapped to ISO-2; missing country kept as UNKNOWN.
- Outlier: us_shop USD amounts {BUG} were 100x too large (cents bug); divided by 100.
- Missing amount (null): {n_null} rows dropped.
""")
