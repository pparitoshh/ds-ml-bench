"""Generate the data for the hard clean_orders task.

Run with: uv run python scripts/gen_clean_orders.py

Outputs:
  data/orders.csv, data/fx_rates.csv        visible to the model
  hidden/expected_clean_orders.csv          expected answer plus "_" helper columns
  hidden/trap_ids.json                      order ids per trap, used by the tests

Planted traps (each is detectable from the data itself):
  dates      ISO, MM/DD/YYYY (us_shop only), DD.MM.YYYY (eu_shop only)
  amounts    symbols, thousands separators, EU format for EUR ("1.234,50"),
             currency "cents" (USD cents)
  fx         EUR and GBP converted at the order date; weekends are often
             missing from fx_rates.csv, so use the last earlier rate
  refunds    negative rows (status completed) and status "refunded" rows with
             positive amounts; both point at the original order via refund_of.
             Some refunds appear in both forms: count them once
  tests      email ends with @example.com, or country is TEST
  dups       exact, whitespace/case-only, and same order_id with a later
             updated_at (the latest version wins; file order is shuffled)
  countries  names, aliases and mixed case mapped to ISO-2
  cents bug  us_shop rows with currency USD between 2023-06-12 and 2023-07-02
             hold cents: the amounts are 100x too large. The daily median of
             that source is about 100x the median of all other days.
  nulls      empty amount (drop the row), empty country (keep as UNKNOWN)
"""

import json
import math
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "tasks" / "data_cleaning" / "clean_orders"
SEED = 7
N_BASE = 1560
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2023-12-31")
BUG_START = pd.Timestamp("2023-06-12")
BUG_END = pd.Timestamp("2023-07-02")

ALIASES = {
    "US": ["US", "USA", "United States", "U.S.", "united states of america"],
    "CA": ["CA", "Canada"],
    "GB": ["GB", "UK", "Great Britain", "United Kingdom", "U.K."],
    "DE": ["DE", "Germany", "Deutschland"],
    "FR": ["FR", "France"],
    "ES": ["ES", "Spain", "España"],
    "IT": ["IT", "Italy", "Italia"],
}
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "mail.de", "web.de"]


def eu_fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mess(rng: random.Random, s: str) -> str:
    """Random casing and padding, used for text columns."""
    s = rng.choice([s, s.upper(), s.lower(), s.title()])
    return rng.choice(["", " ", "  "]) + s + rng.choice(["", " "])


def make_fx(rng: random.Random):
    days = pd.date_range("2022-12-26", END)
    series = {}
    for cur, base in (("EUR", 1.08), ("GBP", 1.25)):
        rate, rows = base, {}
        for d in days:
            rate *= math.exp(rng.gauss(0, 0.004))
            if d.day_of_week < 5 or rng.random() < 0.4:
                rows[d] = round(rate, 6)
        series[cur] = pd.Series(rows)
    return series


def main() -> None:
    rng = random.Random(SEED)
    fx = make_fx(rng)

    # ---- base orders -------------------------------------------------------
    orders = []
    for i in range(N_BASE):
        if rng.random() < 0.55:
            src = "us_shop"
            iso = rng.choices(["US", "CA", "GB"], [75, 15, 10])[0]
            cur = "cents" if rng.random() < 0.10 else "USD"
        else:
            src = "eu_shop"
            iso = rng.choices(["DE", "FR", "ES", "IT", "GB"], [35, 25, 15, 10, 15])[0]
            cur = "GBP" if iso == "GB" else "EUR"
        date = START + pd.Timedelta(days=rng.randrange(365))
        v = min(max(rng.lognormvariate(math.log(40), 0.6), 4), 250)
        if src == "eu_shop" and rng.random() < 0.06:
            v = rng.uniform(900, 2800)  # large B2B orders: thousands separators
        if cur == "cents":
            cents = int(round(v * 100))
            local = cents / 100
        else:
            local = round(v, 2)
        orders.append(
            dict(
                id=10001 + i, src=src, iso=iso, cur=cur, date=date, local=local,
                email=f"user{rng.randrange(10**5)}@{rng.choice(DOMAINS)}",
                bug=(src == "us_shop" and cur == "USD" and BUG_START <= date <= BUG_END),
                updated=date + pd.Timedelta(hours=rng.randrange(24), minutes=rng.randrange(60)),
                test=False, null_amt=False, null_country=False,
                version=False, refund=None, dup=None,
            )
        )

    # ---- assign disjoint trap categories -------------------------------------
    def pick(pool, k):
        chosen = rng.sample(pool, k)
        return chosen

    free = list(range(N_BASE))
    for o in pick(free, 45):
        orders[o]["test"] = rng.choice(["email", "country", "both"])
    free = [i for i in free if not orders[i]["test"]]
    nonbug = [i for i in free if not orders[i]["bug"]]
    for i in pick(nonbug, 25):
        orders[i]["null_amt"] = True
    nonbug = [i for i in nonbug if not orders[i]["null_amt"]]
    for i in pick(nonbug, 50):
        orders[i]["version"] = True
    nonbug = [i for i in nonbug if not orders[i]["version"] and orders[i]["cur"] != "cents"]
    for n, i in enumerate(pick(nonbug, 120)):
        orders[i]["refund"] = n % 3 + 1  # 1 negative row, 2 refunded row, 3 both
    for i in pick(free, 35):
        orders[i]["null_country"] = True
    plain = [i for i, o in enumerate(orders) if not (o["version"] or o["refund"] or o["null_amt"])]
    dups = pick(plain, 130)
    for n, i in enumerate(dups):
        orders[i]["dup"] = "exact" if n < 70 else "fuzzy"

    # ---- render -------------------------------------------------------------
    def rate_for(cur, date):
        if cur in ("USD", "cents"):
            return 1.0, False
        s = fx[cur]
        return float(s.asof(date)), date not in s.index

    def country_raw(o):
        if o["test"] in ("country", "both"):
            return mess(rng, "TEST")
        if o["null_country"]:
            return ""
        return mess(rng, rng.choice(ALIASES[o["iso"]]))

    def amount_raw(o, local, negative=False, date=None):
        bug = o["src"] == "us_shop" and o["cur"] == "USD" and BUG_START <= date <= BUG_END
        sign = "-" if negative else ""
        if o["cur"] == "cents":
            return str(int(round(local * 100))), "cents"
        if o["cur"] == "USD":
            x = local * 100 if bug else local
            style = rng.choice(["sym", "usd", "plain"])
            tag = "usd_bug" if bug else ("usd_plain" if style == "plain" else "usd_decorated")
            body = {"sym": f"${x:,.2f}", "usd": f"USD {x:,.2f}", "plain": f"{x:.2f}"}[style]
            return (sign + body if style != "usd" else f"{sign}{body}"), tag
        if o["cur"] == "EUR":
            body = rng.choice([f"€{eu_fmt(local)}", f"{eu_fmt(local)} €", eu_fmt(local), f"{eu_fmt(local)} EUR"])
            return sign + body, "eur"
        return sign + rng.choice([f"£{local:,.2f}", f"{local:,.2f} GBP"]), "gbp"

    def date_raw(o, date):
        kind = rng.choice(["iso", "local"])
        if kind == "iso":
            return date.strftime("%Y-%m-%d"), "iso"
        if o["src"] == "us_shop":
            return date.strftime("%m/%d/%Y"), "us_slash"
        return date.strftime("%d.%m.%Y"), "eu_dot"

    def raw_row(o, order_id, date, local, status, refund_of=None, negative=False,
                updated=None, amount_override=None):
        a, atag = amount_raw(o, local, negative, date)
        d, dtag = date_raw(o, date)
        cur = o["cur"] if o["cur"] != "cents" else rng.choice(["cents", "Cents"])
        if o["cur"] == "USD" and rng.random() < 0.15:
            cur = rng.choice(["usd", " USD "])
        email = o["email"]
        if o["test"] in ("email", "both"):
            email = f"qa{rng.randrange(1000)}@example.com"
        if amount_override is not None:
            a = amount_override
        raw = dict(
            order_id=order_id, order_date=d, amount=a, currency=cur, country=country_raw(o),
            source=o["src"], customer_email=email, status=status,
            refund_of="" if refund_of is None else refund_of,
            updated_at=(updated or o["updated"]).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return raw, atag, dtag

    rows, expected = [], []
    next_refund_id = 50001
    trap = {k: [] for k in ("test", "null_amount", "exact_dup", "fuzzy_dup", "version")}

    for o in orders:
        status = rng.choice(["completed", "completed", "Completed", " COMPLETED"])
        override = ""
        if o["null_amt"]:
            override = rng.choice(["", "N/A"])
        raw, atag, dtag = raw_row(o, o["id"], o["date"], o["local"], status,
                                  amount_override=override if o["null_amt"] else None)
        rows.append(raw)

        if o["test"]:
            trap["test"].append(o["id"])
        elif o["null_amt"]:
            trap["null_amount"].append(o["id"])
        else:
            rate, filled = rate_for(o["cur"], o["date"])
            usd = round(o["local"] * rate, 2) if not o["bug"] else round(o["local"], 2)
            if o["cur"] == "USD" or o["cur"] == "cents":
                usd = round(o["local"], 2)
            iso = "UNKNOWN" if o["null_country"] else o["iso"]
            expected.append(dict(
                order_id=o["id"], order_date=o["date"].strftime("%Y-%m-%d"), amount_usd=usd,
                country=iso, is_refund=False, _source=o["src"], _date_fmt=dtag, _amt_fmt=atag,
                _cur=o["cur"], _local=o["local"], _fx_filled=filled,
                _alias=(not o["null_country"]) and raw["country"].strip().upper() != o["iso"],
                _null_country=o["null_country"], _version=o["version"],
            ))

        if o["version"]:
            trap["version"].append(o["id"])
            early_local = round(o["local"] * rng.uniform(0.6, 1.4), 2)
            if abs(early_local - o["local"]) < 0.5:
                early_local = round(o["local"] + 5, 2)
            early = o["updated"] - pd.Timedelta(days=rng.randint(1, 9), hours=rng.randrange(24))
            er, _, _ = raw_row(o, o["id"], o["date"], early_local, status, updated=early)
            rows.append(er)

        if o["dup"]:
            if not o["test"]:
                trap[o["dup"] + "_dup"].append(o["id"])
            if o["dup"] == "exact":
                rows.append(dict(raw))
            else:
                f = dict(raw)
                f["country"] = mess(rng, f["country"].strip()) if f["country"].strip() else f["country"]
                f["customer_email"] = mess(rng, f["customer_email"]).upper().strip() if rng.random() < 0.5 else " " + f["customer_email"]
                f["status"] = mess(rng, f["status"].strip())
                f["currency"] = mess(rng, f["currency"].strip())
                if f == raw:
                    f["status"] = f["status"] + " "
                rows.append(f)

        if o["refund"]:
            rdate = min(o["date"] + pd.Timedelta(days=rng.randint(1, 25)), END)
            rate, filled = rate_for(o["cur"], rdate)
            usd = -round(o["local"] * rate, 2)
            kinds = {1: ["neg"], 2: ["pos"], 3: ["neg", "pos"]}[o["refund"]]
            rupd = rdate + pd.Timedelta(hours=rng.randrange(24))
            amounts = {}
            for k in kinds:
                rid = next_refund_id
                next_refund_id += 1
                rr, _, dtag = raw_row(o, rid, rdate, o["local"],
                                      "completed" if k == "neg" else "refunded",
                                      refund_of=o["id"], negative=(k == "neg"), updated=rupd)
                if k == "pos":
                    rr["date_tag"] = dtag
                rows.append(rr)
                amounts[k] = (rid, dtag)
            rid, dtag = amounts.get("neg") or amounts["pos"]
            if o["refund"] == 3:
                # both forms share one date string so the date test is unambiguous
                for r in rows[-2:]:
                    r["order_date"] = rows[-2]["order_date"]
                dtag = amounts["neg"][1]
            expected.append(dict(
                order_id=rid, order_date=rdate.strftime("%Y-%m-%d"), amount_usd=usd,
                country="UNKNOWN" if o["null_country"] else o["iso"], is_refund=True,
                _source=o["src"], _date_fmt=dtag, _amt_fmt="refund", _cur=o["cur"],
                _local=-o["local"], _fx_filled=filled, _alias=False,
                _null_country=o["null_country"], _version=False,
            ))

    for r in rows:
        r.pop("date_tag", None)
    rng.shuffle(rows)

    # ---- write ------------------------------------------------------------
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "hidden").mkdir(parents=True, exist_ok=True)
    cols = ["order_id", "order_date", "amount", "currency", "country", "source",
            "customer_email", "status", "refund_of", "updated_at"]
    pd.DataFrame(rows)[cols].to_csv(ROOT / "data" / "orders.csv", index=False)

    fx_rows = [
        (d.strftime("%Y-%m-%d"), cur, r)
        for cur, s in fx.items() for d, r in s.items()
    ]
    fx_df = pd.DataFrame(fx_rows, columns=["date", "currency", "rate_to_usd"])
    fx_df.sort_values(["date", "currency"]).to_csv(ROOT / "data" / "fx_rates.csv", index=False)

    exp = pd.DataFrame(expected).sort_values("order_id")
    exp.to_csv(ROOT / "hidden" / "expected_clean_orders.csv", index=False, float_format="%.2f")
    (ROOT / "hidden" / "trap_ids.json").write_text(json.dumps(trap, indent=1))
    print(f"raw rows {len(rows)}, expected rows {len(exp)}, refunds {int(exp.is_refund.sum())}")


if __name__ == "__main__":
    main()
