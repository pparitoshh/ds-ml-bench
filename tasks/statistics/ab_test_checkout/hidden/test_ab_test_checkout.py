"""Hidden tests for the ab_test_checkout task. One test per trap or output field."""

import json
import math
import os
import re
from pathlib import Path

import pytest

WORKDIR = Path(os.environ["DSBENCH_WORKDIR"])
HERE = Path(__file__).parent
EXP = json.loads((HERE / "expected.json").read_text())
REF = EXP["reference"]
ALT = EXP["alt_ios_web"]
BANDS = EXP["bands"]

KEYS = [
    "decision", "primary_metric", "conversion_lift", "conversion_lift_ci_low",
    "conversion_lift_ci_high", "p_value", "revenue_per_user_lift",
    "revenue_per_user_lift_ci_low", "revenue_per_user_lift_ci_high",
    "srm_detected", "excluded_users", "lift_by_platform",
]


@pytest.fixture(scope="module")
def ans():
    return json.loads((WORKDIR / "answer.json").read_text())


def num(ans, key):
    v = ans.get(key)
    assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{key} is not a number"
    assert math.isfinite(v), f"{key} is not finite"
    return float(v)


def in_band(x, band):
    return band[0] <= x <= band[1]


def near_reference(ans, trap):
    """The answer is closer to a correct lift than to the lift with this trap left in."""
    x = num(ans, "conversion_lift")
    wrong = EXP["ablation_lift"][trap]
    tol = max(0.5 * abs(REF["lift"] - wrong), 0.03)
    return min(abs(x - REF["lift"]), abs(x - ALT["lift"])) <= tol


def test_schema(ans):
    assert set(KEYS) <= set(ans), f"missing keys: {set(KEYS) - set(ans)}"
    assert ans["decision"] in {"ship", "dont_ship", "rerun"}
    assert isinstance(ans["srm_detected"], bool)
    assert isinstance(ans["lift_by_platform"], dict) and ans["lift_by_platform"]
    for k in KEYS:
        if k not in {"decision", "primary_metric", "srm_detected", "lift_by_platform"}:
            num(ans, k)


def test_decision(ans):
    assert ans["decision"] == EXP["decision"]


def test_primary_metric(ans):
    assert ans["primary_metric"] == "conversion"


def test_srm_detected(ans):
    assert ans["srm_detected"] is True


def test_excluded_users_total(ans):
    x = num(ans, "excluded_users")
    assert any(in_band(x, b) for b in BANDS["excluded_users"]), x


def test_excluded_users_cover_dirty_users(ans):
    # Bots, internal and contaminated users alone are n_dirty; any correct run removes more.
    assert num(ans, "excluded_users") >= 0.97 * EXP["n_dirty"]


def test_conversion_lift_in_band(ans):
    x = num(ans, "conversion_lift")
    assert in_band(x, BANDS["lift"]), x


def test_pooled_simpson_sign_fixed(ans):
    assert near_reference(ans, "srm")
    assert num(ans, "conversion_lift") > 0


def test_pre_period_excluded(ans):
    assert near_reference(ans, "pre")


def test_bots_removed(ans):
    assert near_reference(ans, "bots")


def test_internal_users_removed(ans):
    assert near_reference(ans, "internal")


def test_contamination_handled(ans):
    assert near_reference(ans, "contam")


def test_lift_ci_contains_truth(ans):
    lo, hi = num(ans, "conversion_lift_ci_low"), num(ans, "conversion_lift_ci_high")
    truth = EXP["true_lift"]
    assert lo <= truth <= hi or lo <= EXP["true_lift_ios_web"] <= hi
    assert lo <= num(ans, "conversion_lift") <= hi


def test_lift_ci_width(ans):
    width = num(ans, "conversion_lift_ci_high") - num(ans, "conversion_lift_ci_low")
    assert 0.05 <= width <= BANDS["ci_max_width"], width


def test_p_value_range(ans):
    assert in_band(num(ans, "p_value"), BANDS["p_value"])


def test_revenue_lift_robust(ans):
    # Whales make the raw mean lift wrong; the robust estimate stays in this band.
    assert in_band(num(ans, "revenue_per_user_lift"), BANDS["rev_lift"])


def test_revenue_ci_covers_zero(ans):
    lo, hi = num(ans, "revenue_per_user_lift_ci_low"), num(ans, "revenue_per_user_lift_ci_high")
    assert lo <= 0 <= hi
    assert lo < num(ans, "revenue_per_user_lift") < hi
    assert hi - lo <= 1.5


def test_platform_lifts(ans):
    p = {k.lower(): v for k, v in ans["lift_by_platform"].items()}
    tol = BANDS["platform_abs_tol"]
    for plat in ("ios", "web"):
        assert isinstance(p.get(plat), (int, float)), f"{plat} missing"
        assert abs(p[plat] - REF["by_platform"][plat]) <= tol, (plat, p[plat])


def test_android_platform_lift(ans):
    p = {k.lower(): v for k, v in ans["lift_by_platform"].items()}
    v = p.get("android")
    if v is None:
        return  # the whole platform was excluded
    assert abs(v - REF["by_platform"]["android"]) <= BANDS["platform_abs_tol"] + 0.04, v


NOTE_GROUPS = {
    "srm": r"srm|sample[- ]ratio|ratio mismatch|imbalance",
    "simpson": r"simpson|platform mix|mix shift|composition",
    "novelty": r"novelty|week 1|first week|decay",
    "bot_internal": r"\bbots?\b|internal|employee|staff|company\.internal",
    "contamination": r"contaminat|both arms|more than one (arm|variant)|multiple variants",
    "heavy_tail": r"outlier|whale|heavy[- ]tail|winsor|skew|bootstrap",
    "pre_period": r"pre[- ]period|before the (test|experiment)|pre[- ]experiment|before the start",
}


def test_notes_cover_issues():
    text = (WORKDIR / "notes.md").read_text().lower()
    hits = [k for k, pat in NOTE_GROUPS.items() if re.search(pat, text)]
    assert len(hits) >= 4, hits
