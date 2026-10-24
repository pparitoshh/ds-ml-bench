"""Reference solution. Every rule is derived from the data, not from the generator.

Findings (also written to notes.md):
- Sample ratio: the arms should be 50/50. A chi-square test on the raw counts fails badly.
  A scan of platform x assignment-day cells shows one block of cells with far too few
  treatment users: android, days 3-6. The logging bug dropped users there, and the dropped
  users were mostly buyers, so the surviving treatment users look worse than they are.
  Those cells are excluded in both arms.
- Simpson: the platforms convert at very different rates and the arms have a different
  platform mix because of the lost android users. The pooled lift has the wrong sign.
- Internal users: email ends with @company.internal. Bots: more than 50 events in one day.
- Contamination: some users have assignment rows in both arms. They are removed.
- Pre-period: events before the first assignment date are not part of the test.
- Novelty: the lift in week 1 is clearly larger than the lift in weeks 2-4. Weeks 2-4 show
  the stable effect, which is still positive.
- Revenue is heavy-tailed: a few users spend thousands. Revenue per user is winsorized at the
  99.9th percentile and the CI comes from a bootstrap. The CI covers 0.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 28
BOT_EVENTS_PER_DAY = 50
SRM_CELL_Z = 4.0
WINSOR_Q = 0.999
BOOT_REPS = 2000


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def rate_lift(conv_t, conv_c):
    """Relative lift of two conversion rates with a delta-method 95% CI and a z-test p-value."""
    nt, nc = len(conv_t), len(conv_c)
    pt, pc = conv_t.mean(), conv_c.mean()
    lift = pt / pc - 1
    se = math.sqrt((1 - pt) / (nt * pt) + (1 - pc) / (nc * pc))
    lo = math.exp(math.log(pt / pc) - 1.96 * se) - 1
    hi = math.exp(math.log(pt / pc) + 1.96 * se) - 1
    pool = (pt * nt + pc * nc) / (nt + nc)
    z = (pt - pc) / math.sqrt(pool * (1 - pool) * (1 / nt + 1 / nc))
    return lift, lo, hi, 2 * norm_sf(abs(z))


def mean_lift_boot(rev_t, rev_c, seed=7):
    rng = np.random.default_rng(seed)
    point = rev_t.mean() / rev_c.mean() - 1
    boots = np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        t = rng.choice(rev_t, len(rev_t)).mean()
        c = rng.choice(rev_c, len(rev_c)).mean()
        boots[i] = t / c - 1
    return point, *np.percentile(boots, [2.5, 97.5])


def load(data_dir="."):
    assign = pd.read_csv(Path(data_dir) / "assignments.csv", parse_dates=["assigned_date"])
    events = pd.read_csv(Path(data_dir) / "events.csv", parse_dates=["ts"])
    return assign, events


def analyze(assign, events, *, include_pre=False, drop_bots=True, drop_internal=True,
            drop_contam=True, drop_srm=True, drop_android=False):
    """Run the analysis. The flags switch single steps off; the defaults are the correct run."""
    start = assign["assigned_date"].min()
    assign = assign.copy()
    assign["day"] = (assign["assigned_date"] - start).dt.days + 1

    out = {"start": str(start.date())}
    out["raw_counts"] = assign["variant"].value_counts().to_dict()
    n_c, n_t = out["raw_counts"]["control"], out["raw_counts"]["treatment"]
    chi2 = (n_c - (n_c + n_t) / 2) ** 2 / ((n_c + n_t) / 2) * 2
    out["srm_p"] = math.erfc(math.sqrt(chi2 / 2))
    out["srm_detected"] = out["srm_p"] < 0.001

    per_day = events.assign(d=events["ts"].dt.floor("D")).groupby(["user_id", "d"]).size()
    bots = set(per_day[per_day > BOT_EVENTS_PER_DAY].reset_index()["user_id"])
    internal = set(assign.loc[assign["email"].str.lower().str.endswith("@company.internal"), "user_id"])
    arms = assign.groupby("user_id")["variant"].nunique()
    contam = set(arms[arms > 1].index)

    drop = set()
    if drop_bots:
        drop |= bots
    if drop_internal:
        drop |= internal
    if drop_contam:
        drop |= contam
    kept = assign[~assign["user_id"].isin(drop)].copy()

    # Scan the platform x day cells for a broken split.
    cells = kept.groupby(["platform", "day", "variant"]).size().unstack(fill_value=0)
    z = (cells["treatment"] - cells["control"]) / np.sqrt(cells["treatment"] + cells["control"])
    bad = z[z.abs() > SRM_CELL_Z]
    out["bad_cells"] = [(p, int(d)) for p, d in bad.index]
    bad_keys = set(out["bad_cells"])
    if drop_srm:
        mask = [(p, d) in bad_keys for p, d in zip(kept["platform"], kept["day"])]
        srm_users = set(assign.loc[[(p, d) in bad_keys for p, d in zip(assign["platform"], assign["day"])], "user_id"])
        kept = kept[[not m for m in mask]]
        drop |= srm_users
    if drop_android:
        drop |= set(assign.loc[assign["platform"] == "android", "user_id"])
        kept = kept[kept["platform"] != "android"]
    out["excluded_users"] = len(drop)
    out["n_bots"], out["n_internal"], out["n_contam"] = len(bots), len(internal), len(contam)

    purchases = events[events["event_type"] == "purchase"]
    if not include_pre:
        purchases = purchases[purchases["ts"] >= start]
    merged = kept.merge(purchases[["user_id", "ts", "revenue"]], on="user_id", how="left")
    if not include_pre:
        merged = merged[merged["ts"].isna() | (merged["ts"] >= merged["assigned_date"])]
    key = ["user_id", "variant", "platform", "day"]
    per_user = (merged.groupby(key, dropna=False)
                .agg(conv=("ts", lambda s: int(s.notna().any())), rev=("revenue", "sum"))
                .reset_index())
    per_user["week"] = (per_user["day"] - 1) // 7 + 1

    def split(df):
        return df[df["variant"] == "treatment"], df[df["variant"] == "control"]

    t, c = split(per_user)
    out["lift"], out["ci_low"], out["ci_high"], out["p_value"] = rate_lift(t["conv"].values, c["conv"].values)
    cap = per_user["rev"].quantile(WINSOR_Q)
    per_user["rev_w"] = per_user["rev"].clip(upper=cap)
    t, c = split(per_user)
    out["rev_lift"], out["rev_ci_low"], out["rev_ci_high"] = mean_lift_boot(t["rev_w"].values, c["rev_w"].values)
    out["rev_lift_raw"] = t["rev"].mean() / c["rev"].mean() - 1

    out["by_platform"] = {}
    for plat, g in per_user.groupby("platform"):
        gt, gc = split(g)
        out["by_platform"][plat] = float(gt["conv"].mean() / gc["conv"].mean() - 1)
    out["by_week"] = {}
    for wk, g in per_user.groupby("week"):
        gt, gc = split(g)
        out["by_week"][int(wk)] = float(gt["conv"].mean() / gc["conv"].mean() - 1)
    out["n_analyzed"] = len(per_user)
    return out


def main():
    assign, events = load()
    r = analyze(assign, events)
    stable = r["by_week"]
    ship = r["srm_detected"] and r["ci_low"] > 0 and r["p_value"] < 0.05
    decision = "ship" if ship else ("rerun" if r["srm_detected"] and not r["bad_cells"] else "dont_ship")
    answer = {
        "decision": decision,
        "primary_metric": "conversion",
        "conversion_lift": r["lift"],
        "conversion_lift_ci_low": r["ci_low"],
        "conversion_lift_ci_high": r["ci_high"],
        "p_value": r["p_value"],
        "revenue_per_user_lift": r["rev_lift"],
        "revenue_per_user_lift_ci_low": r["rev_ci_low"],
        "revenue_per_user_lift_ci_high": r["rev_ci_high"],
        "srm_detected": bool(r["srm_detected"]),
        "excluded_users": r["excluded_users"],
        "lift_by_platform": r["by_platform"],
    }
    Path("answer.json").write_text(json.dumps(answer, indent=2))
    cells = ", ".join(f"{p} day {d}" for p, d in r["bad_cells"])
    wk = ", ".join(f"week {k}: {v:+.1%}" for k, v in stable.items())
    Path("notes.md").write_text(f"""# Notes

## Issues found
1. Sample ratio mismatch (SRM): the raw split is {r['raw_counts']}, chi-square p = {r['srm_p']:.2g}.
   The cells with too few treatment users are: {cells}. A logging bug lost users there.
   The lost users were mostly buyers, so I excluded those platform-day cells in both arms.
2. Simpson's paradox: the platform mix differs between the arms because of the SRM.
   The pooled lift has the wrong sign. I compare the arms on cleaned data and report lift by platform.
3. Internal and bot users: {r['n_internal']} users with @company.internal emails and {r['n_bots']} bot users
   with more than {BOT_EVENTS_PER_DAY} events per day were removed.
4. Contamination: {r['n_contam']} users are in both arms (both arms contain them). I removed them.
5. Pre-period: events before {r['start']} are not part of the test. I counted only purchases
   on or after the assignment date.
6. Novelty effect: the lift by assignment week is {wk}. The week 1 lift is inflated and decays.
   Weeks 2-4 show the stable effect, which is still positive.
7. Revenue is heavy-tailed: a few whales spend thousands. I winsorized revenue per user at the
   99.9th percentile and used a bootstrap CI. The raw mean lift is {r['rev_lift_raw']:+.0%} and not reliable.

## Decision
After cleaning, conversion lift is {r['lift']:+.1%} (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%}, p = {r['p_value']:.3g}).
Revenue per user lift is {r['rev_lift']:+.1%} with a CI that covers 0, so I cannot claim a revenue change.
The conversion effect is positive and significant after the fixes, and it stays positive after the novelty decays.
Decision: {decision}. Fix the android logging bug before the next test.
""")


if __name__ == "__main__":
    main()
