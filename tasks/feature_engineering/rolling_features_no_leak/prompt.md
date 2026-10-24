# rolling_features_no_leak

Compute per-transaction 7-day rolling statistics without leaking the current or any future transaction.

## Input

`transactions.csv` has the following columns:

- `customer_id`: integer customer identifier
- `date`: transaction date (`YYYY-MM-DD`)
- `amount`: transaction amount

## Required output

Write `features.csv` with exactly these columns, preserving one row per transaction:

- `customer_id`
- `date`
- `amount`
- `rolling_spend_7d`: total amount of the customer's **strictly earlier** transactions whose date is within the 7-day window `(current_date - 7 days, current_date)` (exclusive of the current date, inclusive lower bound). Use a 0.0 if there are no earlier transactions in the window.
- `rolling_count_7d`: count of those strictly earlier transactions in the window. Use 0 if none.

Important:

- Sort by `customer_id` and `date` before computing.
- A transaction on the same day as the current transaction does **not** count as earlier.
- Do not include the current transaction itself.
