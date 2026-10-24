# churn_classifier

Train a churn classifier and produce probability predictions for the test set.

## Input

- `train.csv`: labeled training data with columns
  - `id`: integer identifier
  - `x1`, `x2`, `x3`: numeric features
  - `days_since_cancellation`: a post-churn field that is **non-null only for past churners** in the training set and is all-null in the test set
  - `churn`: target label (0 or 1)
- `test.csv`: unlabeled test data with the same columns except `churn`

## Required output

Write `predictions.csv` with exactly these columns:

- `id`: from `test.csv`
- `proba`: predicted probability of churn as a float in `[0, 1]`

Do **not** use `days_since_cancellation` as a feature. Relying on it will fail on the test set because it is missing for every row.
