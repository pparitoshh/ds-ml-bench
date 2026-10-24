"""Download MovieLens ml-latest-small and build a temporal train/test split."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from dsbench.scorers.metrics import compute_recall_at_k


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "ranking" / "movielens_recall_at_10"
DATA_DIR = TASK_DIR / "data"
HIDDEN_DIR = TASK_DIR / "hidden"
CACHE_DIR = TASK_DIR / ".cache"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "ml-latest-small.zip"
    if not zip_path.exists():
        print(f"Downloading {ZIP_URL} ...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    return zip_path


def extract(zip_path: Path) -> Path:
    extract_dir = CACHE_DIR / "ml-latest-small"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CACHE_DIR)
    return extract_dir


def build_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    train_rows = []
    test_rows = []
    min_ratings = 5
    for _, group in ratings.groupby("user_id", sort=False):
        n = len(group)
        if n < min_ratings:
            train_rows.append(group)
            continue
        test_size = max(1, n // 5)
        train_rows.append(group.iloc[:-test_size])
        test_rows.append(group.iloc[-test_size:])
    train = pd.concat(train_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)
    return train, test


def popularity_baseline_predictions(
    train: pd.DataFrame,
    users_to_score: pd.DataFrame,
) -> pd.DataFrame:
    item_counts = train.groupby("item_id").size().sort_values(ascending=False)
    top_items = item_counts.index.tolist()
    user_train_items = (
        train.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    rows = []
    for user_id in users_to_score["user_id"]:
        seen = user_train_items.get(user_id, set())
        recs = [item for item in top_items if item not in seen][:10]
        if len(recs) < 10:
            extra = [item for item in item_counts.index if item not in seen and item not in recs]
            recs.extend(extra[: 10 - len(recs)])
        for rank, item_id in enumerate(recs, start=1):
            rows.append({"user_id": user_id, "item_id": item_id, "rank": rank})
    return pd.DataFrame(rows)


def main() -> None:
    zip_path = download()
    extract_dir = extract(zip_path)

    ratings = pd.read_csv(extract_dir / "ratings.csv")
    ratings = ratings.rename(columns={
        "userId": "user_id",
        "movieId": "item_id",
    })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

    train, test = build_split(ratings)
    train.to_csv(DATA_DIR / "train.csv", index=False)

    labels = test[test["rating"] >= 4][["user_id", "item_id"]].copy()
    labels.to_csv(HIDDEN_DIR / "labels.csv", index=False)

    users_to_score = pd.DataFrame({"user_id": sorted(test["user_id"].unique())})
    users_to_score.to_csv(DATA_DIR / "users_to_score.csv", index=False)

    baseline_preds = popularity_baseline_predictions(train, users_to_score)
    baseline_recall = compute_recall_at_k(baseline_preds, labels, k=10)
    threshold = baseline_recall * 1.25 if baseline_recall > 0 else 0.01
    threshold = round(float(threshold), 6)
    baseline_recall = round(float(baseline_recall), 6)

    metric_yaml = (
        f"predictions_file: predictions.csv\n"
        f"labels_file: labels.csv\n"
        f"metric: recall_at_k\n"
        f"k: 10\n"
        f"threshold: {threshold}\n"
        f"baseline: {baseline_recall}\n"
        f"higher_is_better: true\n"
    )
    (HIDDEN_DIR / "metric.yaml").write_text(metric_yaml)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Relevant labels: {len(labels)}")
    print(f"Users to score: {len(users_to_score)}")
    print(f"Popularity baseline Recall@10: {baseline_recall:.6f}")
    print(f"Threshold (1.25x): {threshold:.6f}")


if __name__ == "__main__":
    main()
