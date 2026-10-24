"""Naive solution: pools everything, no cleaning."""

import json
import math

import pandas as pd

assign = pd.read_csv("assignments.csv")
events = pd.read_csv("events.csv")

buyers = set(events.loc[events["event_type"] == "purchase", "user_id"])
rev = events.groupby("user_id")["revenue"].sum()
assign["conv"] = assign["user_id"].isin(buyers).astype(int)
assign["rev"] = assign["user_id"].map(rev).fillna(0.0)

t = assign[assign["variant"] == "treatment"]
c = assign[assign["variant"] == "control"]
pt, pc = t["conv"].mean(), c["conv"].mean()
se = math.sqrt(pt * (1 - pt) / len(t) + pc * (1 - pc) / len(c))
lift = pt / pc - 1
p = math.erfc(abs(pt - pc) / se / math.sqrt(2))

rt, rc = t["rev"].mean(), c["rev"].mean()
rse = math.sqrt(t["rev"].var() / len(t) + c["rev"].var() / len(c))
by_platform = {
    plat: g.loc[g["variant"] == "treatment", "conv"].mean() / g.loc[g["variant"] == "control", "conv"].mean() - 1
    for plat, g in assign.groupby("platform")
}
decision = "ship" if p < 0.05 and lift > 0 else "dont_ship"
json.dump(
    {
        "decision": decision,
        "primary_metric": "conversion",
        "conversion_lift": lift,
        "conversion_lift_ci_low": lift - 1.96 * se / pc,
        "conversion_lift_ci_high": lift + 1.96 * se / pc,
        "p_value": p,
        "revenue_per_user_lift": rt / rc - 1,
        "revenue_per_user_lift_ci_low": (rt - rc - 1.96 * rse) / rc,
        "revenue_per_user_lift_ci_high": (rt - rc + 1.96 * rse) / rc,
        "srm_detected": False,
        "excluded_users": 0,
        "lift_by_platform": by_platform,
    },
    open("answer.json", "w"),
    indent=2,
)
open("notes.md", "w").write("Compared conversion and revenue per user between the arms.\n")
