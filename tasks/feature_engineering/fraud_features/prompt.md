# fraud_features

Ticket from the ML lead: build point-in-time features for our fraud model from `transactions.csv`, `customers.csv` and `chargebacks.csv`.

The model scores each transaction at the moment it happens.

Write `features.csv` with one row per `transaction_id` and these columns:

- `transaction_id`
- `txn_count_24h`
- `amount_sum_7d`
- `amount_mean_30d`
- `distinct_merchants_7d`
- `amount_zscore_30d` (the amount compared with the customer's prior 30 days)
- `days_since_signup`
- `prior_chargebacks` (count known at scoring time)
- `is_first_txn_at_merchant`

Also write `notes.md` with each leakage risk you found and how you handled it.

## Input

`transactions.csv` columns: `transaction_id`, `customer_id`, `timestamp`, `tz`, `merchant_id`, `amount`, `status`.

`customers.csv` columns: `customer_id`, `signup_date`, `updated_at`, `country`.

`chargebacks.csv` columns: `transaction_id`, `chargeback_date`.
