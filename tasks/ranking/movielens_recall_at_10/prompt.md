# movielens_recall_at_10

Recommend movies for users and achieve a strong Recall@10 on a temporal MovieLens split.

## Input

- `train.csv`: past ratings with columns `user_id`, `item_id`, `rating`, `timestamp`
- `users_to_score.csv`: a single column `user_id` listing the users for which you must produce recommendations

## Required output

Write `predictions.csv` with exactly these columns:

- `user_id`
- `item_id`
- `rank`: integer rank from 1 to 10 (1 is highest) within each user

Requirements:

- Produce exactly 10 recommendations per user in `users_to_score.csv`.
- Do **not** recommend any item already present in `train.csv` for that user.
- Items in the training data are rated on a 1–5 scale; the test labels treat ratings of 4 or 5 as relevant.
