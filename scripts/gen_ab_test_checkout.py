"""Generate the data for the hard ab_test_checkout task.

Run with: uv run python scripts/gen_ab_test_checkout.py

Outputs:
  data/assignments.csv, data/events.csv   visible to the model
  hidden/trap_ids.json                    user ids per trap
  hidden/expected.json                    expected values, ablations and tolerance bands

Planted traps (each is detectable from the data itself):
  srm        android treatment users assigned on days 3-6 are partly lost (logging bug).
             Buyers are lost more often than non-buyers. Days 3-6 have 3x traffic.
  simpson    android converts much more than ios and web, and the bug removes android
             treatment users, so the pooled lift has the wrong sign
  internal   emails ending @company.internal, mostly in treatment, high conversion
  bots       users with more than 50 events on a day, mostly in control, high conversion
  contam     users with assignment rows in both arms
  novelty    treatment multiplier on conversion decays: week 1 large, weeks 2-4 stable
  whales     a few purchases of thousands of dollars, mostly in treatment
  pre        events before the first assignment date, with purchases
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "tasks" / "statistics" / "ab_test_checkout"

N_USERS = 40000
START = pd.Timestamp("2026-03-02")
DAYS = 28
PLATFORMS = ["android", "ios", "web"]
PLATFORM_P = [0.50, 0.30, 0.20]
BASE_CONV = {"android": 0.16, "ios": 0.06, "web": 0.04}
WEEK_MULT = {1: 1.18, 2: 1.08, 3: 1.06, 4: 1.06}
N_INTERNAL, N_BOT, N_CONTAM = 600, 400, 400
LOSS_BUYER, LOSS_NONBUYER = 0.85, 0.30
N_WHALES_T, N_WHALES_C = 10, 2
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "proton.me", "company.com", "web.de"]


def load_reference():
    spec = importlib.util.spec_from_file_location("ref", TASK / "hidden" / "reference_solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ts(rng, day, n=None):
    """Random timestamp strings on a day offset (1-based) from START."""
    base = START + pd.Timedelta(days=int(day) - 1)
    return (base + pd.Timedelta(seconds=int(rng.integers(0, 86400)))).strftime("%Y-%m-%d %H:%M:%S")


def main():
    rng = np.random.default_rng(20260301)
    ids = np.arange(100001, 100001 + N_USERS)
    order = rng.permutation(N_USERS)
    role = np.full(N_USERS, "regular", dtype=object)
    role[order[:N_INTERNAL]] = "internal"
    role[order[N_INTERNAL:N_INTERNAL + N_BOT]] = "bot"
    role[order[N_INTERNAL + N_BOT:N_INTERNAL + N_BOT + N_CONTAM]] = "contam"

    day_w = np.ones(DAYS)
    day_w[2:6] = 3.0
    day_w /= day_w.sum()
    plat = rng.choice(PLATFORMS, N_USERS, p=PLATFORM_P)
    day = rng.choice(np.arange(1, DAYS + 1), N_USERS, p=day_w)
    p_treat = np.where(role == "internal", 0.75, np.where(role == "bot", 0.20, 0.5))
    treat = rng.random(N_USERS) < p_treat
    conv_p0 = np.array([BASE_CONV[p] for p in plat])
    conv_p0 = np.where(role == "internal", 0.70, np.where(role == "bot", 0.85,
                       np.where(role == "contam", 0.35, conv_p0)))
    week = (day - 1) // 7 + 1
    mult = np.array([WEEK_MULT[w] for w in week])
    p_conv = np.where((role == "regular") & treat, conv_p0 * mult, conv_p0)
    converts = rng.random(N_USERS) < p_conv

    # Logging bug: android treatment users on days 3-6 are lost, buyers more often.
    cell = (plat == "android") & treat & (day >= 3) & (day <= 6) & (role == "regular")
    lose_p = np.where(converts, LOSS_BUYER, LOSS_NONBUYER)
    lost = cell & (rng.random(N_USERS) < lose_p)
    keep = ~lost

    # Whales: converters in clean cells, mostly in treatment.
    clean_conv = np.where((role == "regular") & converts & keep & ~((plat == "android") & (day >= 3) & (day <= 6)))[0]
    whale_t = rng.choice(clean_conv[treat[clean_conv]], N_WHALES_T, replace=False)
    whale_c = rng.choice(clean_conv[~treat[clean_conv]], N_WHALES_C, replace=False)
    whales = set(whale_t) | set(whale_c)

    arr_rows, ev_rows = [], []
    for i in range(N_USERS):
        if not keep[i]:
            continue
        uid = int(ids[i])
        r = role[i]
        if r == "internal":
            email = f"{rng.choice(['qa', 'dev', 'staff'])}{uid}@" + ("Company.Internal" if rng.random() < 0.05 else "company.internal")
        else:
            email = f"user{uid}@{rng.choice(DOMAINS)}"
        a_date = (START + pd.Timedelta(days=int(day[i]) - 1)).strftime("%Y-%m-%d")
        variants = ["treatment" if treat[i] else "control"]
        dates = [a_date]
        if r == "contam":
            variants.append("control" if treat[i] else "treatment")
            d2 = min(int(day[i]) + int(rng.integers(0, 3)), DAYS)
            dates.append((START + pd.Timedelta(days=d2 - 1)).strftime("%Y-%m-%d"))
        for v, d in zip(variants, dates):
            arr_rows.append((uid, v, d, plat[i], email))

        # Window events (on or after assignment).
        for _ in range(int(rng.integers(1, 3))):
            ev_rows.append((uid, ts(rng, rng.integers(day[i], DAYS + 1)), "view", ""))
        if converts[i]:
            for _ in range(1 + int(rng.poisson(0.5))):
                amt = round(float(rng.lognormal(np.log(40), 1.3)) * (0.88 if treat[i] and r == "regular" else 1.0), 2)
                if i in whales:
                    amt = round(float(rng.uniform(8000, 30000)), 2)
                    whales.discard(i)
                ev_rows.append((uid, ts(rng, rng.integers(day[i], DAYS + 1)), "purchase", amt))
        # Pre-period events: before the first assignment date.
        if r != "bot":
            for _ in range(int(rng.poisson(1.0))):
                d = int(rng.integers(-13, 1))
                ev_rows.append((uid, ts(rng, d), "view", ""))
            if rng.random() < 0.15:
                ev_rows.append((uid, ts(rng, int(rng.integers(-13, 1))), "purchase",
                                round(float(rng.lognormal(np.log(40), 1.2)), 2)))
        if r == "bot":
            for d in rng.choice(np.arange(1, DAYS + 1), 3, replace=False):
                for _ in range(int(rng.integers(60, 111))):
                    ev_rows.append((uid, ts(rng, d), "view", ""))

    assign = pd.DataFrame(arr_rows, columns=["user_id", "variant", "assigned_date", "platform", "email"])
    assign = assign.sample(frac=1, random_state=1).reset_index(drop=True)
    events = pd.DataFrame(ev_rows, columns=["user_id", "ts", "event_type", "revenue"]).sort_values("ts")
    (TASK / "data").mkdir(parents=True, exist_ok=True)
    assign.to_csv(TASK / "data" / "assignments.csv", index=False)
    events.to_csv(TASK / "data" / "events.csv", index=False)

    # Ground truth: expected conversion ratio on the clean regular users.
    reg = (role == "regular") & keep & ~((plat == "android") & (day >= 3) & (day <= 6))
    true_lift = float((conv_p0[reg] * mult[reg]).sum() / conv_p0[reg].sum() - 1)
    regb = reg & (plat != "android")
    true_lift_b = float((conv_p0[regb] * mult[regb]).sum() / conv_p0[regb].sum() - 1)

    ref = load_reference()
    a, e = ref.load(TASK / "data")
    base = ref.analyze(a, e)
    ablate = {
        "pre": ref.analyze(a, e, include_pre=True),
        "bots": ref.analyze(a, e, drop_bots=False),
        "internal": ref.analyze(a, e, drop_internal=False),
        "contam": ref.analyze(a, e, drop_contam=False),
        "srm": ref.analyze(a, e, drop_srm=False),
    }
    alt = ref.analyze(a, e, drop_srm=False, drop_android=True)
    naive_pooled = ref.analyze(a, e, include_pre=True, drop_bots=False, drop_internal=False,
                               drop_contam=False, drop_srm=False)

    ids_json = {
        "internal": [int(x) for x in ids[role == "internal"]],
        "bot": [int(x) for x in ids[role == "bot"]],
        "contam": [int(x) for x in ids[role == "contam"]],
    }
    (TASK / "hidden").mkdir(parents=True, exist_ok=True)
    (TASK / "hidden" / "trap_ids.json").write_text(json.dumps(ids_json))

    exp = {
        "decision": "ship",
        "true_lift": true_lift,
        "true_lift_ios_web": true_lift_b,
        "reference": {k: base[k] for k in ["lift", "ci_low", "ci_high", "p_value", "rev_lift",
                                           "rev_ci_low", "rev_ci_high", "excluded_users", "by_platform", "by_week"]},
        "alt_ios_web": {k: alt[k] for k in ["lift", "ci_low", "ci_high", "p_value", "rev_lift", "excluded_users", "by_platform"]},
        "ablation_lift": {k: v["lift"] for k, v in ablate.items()},
        "naive_pooled_lift": naive_pooled["lift"],
        "bands": {
            "lift": [min(base["lift"], alt["lift"]) - 0.04, max(base["lift"], alt["lift"]) + 0.04],
            "rev_lift": [min(base["rev_lift"], alt["rev_lift"]) - 0.10, max(base["rev_lift"], alt["rev_lift"]) + 0.10],
            "p_value": [base["p_value"] / 10, min(0.05, max(base["p_value"], alt["p_value"]) * 10)],
            "excluded_users": [[round(base["excluded_users"] * 0.95), round(base["excluded_users"] * 1.05)],
                               [round(alt["excluded_users"] * 0.95), round(alt["excluded_users"] * 1.05)]],
            "platform_abs_tol": 0.06,
            "ci_max_width": 0.6,
        },
        "n_dirty": len(set(ids_json["internal"]) | set(ids_json["bot"]) | set(ids_json["contam"])),
    }
    (TASK / "hidden" / "expected.json").write_text(json.dumps(exp, indent=2))
    print(json.dumps(exp, indent=2))
    print("bad cells", base["bad_cells"], "srm_p", base["srm_p"], "raw", base["raw_counts"])
    print("rows", len(assign), len(events))


if __name__ == "__main__":
    main()
