import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
users_to_score = pd.read_csv("users_to_score.csv")

user_ids = sorted(train["user_id"].unique())
item_ids = sorted(train["item_id"].unique())
uid_map = {u: i for i, u in enumerate(user_ids)}
iid_map = {i: j for j, i in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)
R = np.zeros((n_users, n_items), dtype=float)
for u, i, r in zip(train["user_id"], train["item_id"], train["rating"]):
    R[uid_map[u], iid_map[i]] = float(r)

rated_counts = np.count_nonzero(R, axis=1)
user_means = R.sum(axis=1) / np.where(rated_counts == 0, 1, rated_counts)
R_centered = np.where(R != 0, R - user_means[:, None], 0.0)

norms = np.linalg.norm(R_centered, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    S = (R_centered @ R_centered.T) / np.outer(norms, norms)
S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
np.fill_diagonal(S, -1.0)

K = 20
user_train_items = {
    u: set(items) for u, items in train.groupby("user_id")["item_id"]
}

rows = []
for u in users_to_score["user_id"]:
    ui = uid_map[u]
    neighbors = np.argsort(S[ui])[::-1][:K]
    sims = S[ui, neighbors]
    denom = np.abs(sims).sum()
    if denom > 0:
        scores = user_means[ui] + (sims @ R_centered[neighbors, :]) / denom
    else:
        scores = np.zeros(n_items)

    seen = user_train_items.get(u, set())
    for item in seen:
        scores[iid_map[item]] = -np.inf

    top_indices = np.argsort(scores)[::-1][:10]
    for rank, idx in enumerate(top_indices, start=1):
        rows.append({"user_id": u, "item_id": item_ids[idx], "rank": rank})

pd.DataFrame(rows).to_csv("predictions.csv", index=False)
