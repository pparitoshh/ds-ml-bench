# clean_orders

Finance wants monthly net revenue in USD by country from `orders.csv` (plus `fx_rates.csv`).

Produce `clean_orders.csv` with these columns:

- `order_id`
- `order_date` (YYYY-MM-DD)
- `amount_usd` (2 decimals)
- `country` (ISO-2 code)
- `is_refund` (bool)

Also write `notes.md` with each data issue you found and your decision.

## Input

`orders.csv` columns: `order_id`, `order_date`, `amount`, `currency`, `country`, `source`, `customer_email`, `status`, `refund_of`, `updated_at`.

`fx_rates.csv` columns: `date`, `currency`, `rate_to_usd`.
