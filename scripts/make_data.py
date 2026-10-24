"""Deterministically generate small CSV datasets for the benchmark tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"


def _task_path(category: str, task_id: str, *parts: str) -> Path:
    return TASKS_DIR / category / task_id / Path(*parts)


def make_clean_orders() -> None:
    rng = np.random.default_rng(42)
    n_base = 180
    n_dups = 20

    order_ids = np.arange(1000, 1000 + n_base)
    base_dates = (
        pd.date_range("2023-01-01", periods=365, freq="D")
        .to_series()
        .reset_index(drop=True)
    )
    date_offsets = rng.integers(0, 365, size=n_base)
    order_dates = base_dates.iloc[date_offsets].dt.strftime("%Y-%m-%d").values

    # Mix ISO and US date formats.
    date_strings = []
    for d in order_dates:
        if rng.random() < 0.5:
            date_strings.append(d)
        else:
            parts = d.split("-")
            date_strings.append(f"{parts[1]}/{parts[2]}/{parts[0]}")

    currency_choices = rng.choice(["USD", "cents", "  USD  ", "cents "], size=n_base)
    amounts = []
    for c in currency_choices:
        c_clean = c.strip().lower()
        if c_clean == "usd":
            amounts.append(
                round(float(rng.integers(10, 500)) + round(rng.random(), 2), 2)
            )
        else:
            amounts.append(int(rng.integers(1000, 50000)))

    country_choices = [
        "usa",
        "USA",
        "  USA  ",
        "Germany",
        "germany",
        "GERMANY",
        " FRANCE",
        "france ",
    ]
    countries = rng.choice(country_choices, size=n_base)

    base_df = pd.DataFrame({
        "order_id": order_ids,
        "order_date": date_strings,
        "amount": amounts,
        "currency": currency_choices,
        "country": countries,
    })

    dup_indices = rng.integers(0, n_base, size=n_dups)
    dup_df = base_df.iloc[dup_indices].copy()
    raw_df = pd.concat([base_df, dup_df], ignore_index=True)

    data_dir = _task_path("data_cleaning", "clean_orders", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(data_dir / "orders.csv", index=False)

    # Expected cleaned version.
    clean = base_df.copy()
    clean["order_date"] = (
        pd.to_datetime(clean["order_date"], format="mixed", dayfirst=False)
        .dt.strftime("%Y-%m-%d")
    )
    clean["currency"] = clean["currency"].str.strip().str.lower()
    cents_mask = clean["currency"] == "cents"
    clean.loc[cents_mask, "amount"] = clean.loc[cents_mask, "amount"] / 100
    clean["amount"] = clean["amount"].round(2)
    clean["country"] = clean["country"].str.strip().str.upper()
    clean = clean.drop_duplicates().reset_index(drop=True)
    clean = clean[["order_id", "order_date", "amount", "country"]]

    hidden_dir = _task_path("data_cleaning", "clean_orders", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(hidden_dir / "expected_clean_orders.csv", index=False)


def make_rolling_features() -> None:
    rng = np.random.default_rng(42)
    n_customers = 50
    min_txn = 5
    max_txn = 20
    start_date = pd.Timestamp("2023-01-01")

    rows = []
    for cust in range(1001, 1001 + n_customers):
        n_txn = rng.integers(min_txn, max_txn + 1)
        offsets = np.sort(rng.integers(0, 60, size=n_txn))
        amounts = np.round(rng.normal(50.0, 20.0, size=n_txn), 2)
        for off, amt in zip(offsets, amounts):
            rows.append({
                "customer_id": cust,
                "date": (start_date + pd.Timedelta(days=int(off))).strftime("%Y-%m-%d"),
                "amount": amt,
            })

    df = pd.DataFrame(rows)
    data_dir = _task_path("feature_engineering", "rolling_features_no_leak", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "transactions.csv", index=False)

    # Compute expected features without leakage.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["customer_id", "date", "amount"]).reset_index(drop=True)

    expected_rows = []
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
        group["customer_id"] = customer_id
        group["rolling_spend_7d"] = spend
        group["rolling_count_7d"] = count
        expected_rows.append(group)

    expected = pd.concat(expected_rows, ignore_index=True)
    expected["date"] = expected["date"].dt.strftime("%Y-%m-%d")
    expected = expected[[
        "customer_id",
        "date",
        "amount",
        "rolling_spend_7d",
        "rolling_count_7d",
    ]]

    hidden_dir = _task_path("feature_engineering", "rolling_features_no_leak", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    expected.to_csv(hidden_dir / "expected_features.csv", index=False)


def make_churn() -> None:
    rng = np.random.default_rng(42)
    n_train = 800
    n_test = 200

    def _make_features(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "x1": np.round(rng.normal(0, 1, size=n), 4),
            "x2": np.round(rng.normal(0, 1, size=n), 4),
            "x3": np.round(rng.normal(0, 1, size=n), 4),
        })

    train_x = _make_features(n_train)
    test_x = _make_features(n_test)

    # True churn driven by x1, x2, x3.
    logits = (
        0.5 * train_x["x1"].values
        - 0.4 * train_x["x2"].values
        + 0.3 * train_x["x3"].values
        + 0.1
    )
    train_y = (rng.random(n_train) < (1 / (1 + np.exp(-logits)))).astype(int)

    test_logits = (
        0.5 * test_x["x1"].values
        - 0.4 * test_x["x2"].values
        + 0.3 * test_x["x3"].values
        + 0.1
    )
    test_y = (rng.random(n_test) < (1 / (1 + np.exp(-test_logits)))).astype(int)

    train = train_x.copy()
    train["id"] = np.arange(1, n_train + 1)
    train["days_since_cancellation"] = np.nan
    churner_mask = train_y == 1
    train.loc[churner_mask, "days_since_cancellation"] = rng.integers(
        1, 365, size=churner_mask.sum()
    )
    train["churn"] = train_y

    test = test_x.copy()
    test["id"] = np.arange(n_train + 1, n_train + n_test + 1)
    test["days_since_cancellation"] = np.nan
    test["churn"] = test_y

    data_dir = _task_path("modeling", "churn_classifier", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    train[["id", "x1", "x2", "x3", "days_since_cancellation", "churn"]].to_csv(
        data_dir / "train.csv", index=False
    )
    test[["id", "x1", "x2", "x3", "days_since_cancellation"]].to_csv(
        data_dir / "test.csv", index=False
    )

    hidden_dir = _task_path("modeling", "churn_classifier", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    test[["id", "churn"]].to_csv(hidden_dir / "labels.csv", index=False)

    # Sound model AUC used to set threshold.
    features = ["x1", "x2", "x3"]
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["churn"])
    proba = model.predict_proba(test[features])[:, 1]
    sound_auc = roc_auc_score(test["churn"], proba)

    threshold = round(max(0.55, sound_auc - 0.05), 4)
    metric_yaml = (
        f"predictions_file: predictions.csv\n"
        f"labels_file: labels.csv\n"
        f"metric: auc\n"
        f"threshold: {threshold}\n"
        f"higher_is_better: true\n"
    )
    (hidden_dir / "metric.yaml").write_text(metric_yaml)


def make_ab_test() -> None:
    rows = [
        {"segment": "A", "variant": "control", "users": 1000, "conversions": 100},
        {"segment": "A", "variant": "treatment", "users": 500, "conversions": 40},
        {"segment": "B", "variant": "control", "users": 500, "conversions": 25},
        {"segment": "B", "variant": "treatment", "users": 1000, "conversions": 80},
    ]
    df = pd.DataFrame(rows)

    data_dir = _task_path("statistics", "ab_test_simpson", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "ab_data.csv", index=False)

    def _lift(segment_df: pd.DataFrame) -> float:
        ctrl = segment_df[segment_df["variant"] == "control"]
        treat = segment_df[segment_df["variant"] == "treatment"]
        ctrl_rate = ctrl["conversions"].sum() / ctrl["users"].sum()
        treat_rate = treat["conversions"].sum() / treat["users"].sum()
        return float((treat_rate - ctrl_rate) / ctrl_rate)

    overall = _lift(df)
    seg_a = _lift(df[df["segment"] == "A"])
    seg_b = _lift(df[df["segment"] == "B"])

    hidden_dir = _task_path("statistics", "ab_test_simpson", "hidden")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    answer_yaml = (
        f"answers:\n"
        f"  overall_lift: {overall}\n"
        f"  segment_a_lift: {seg_a}\n"
        f"  segment_b_lift: {seg_b}\n"
        f"rel_tol: 0.01\n"
    )
    (hidden_dir / "answer.yaml").write_text(answer_yaml)


def main() -> None:
    make_clean_orders()
    make_rolling_features()
    make_churn()
    make_ab_test()
    print("Generated task data.")


if __name__ == "__main__":
    main()
