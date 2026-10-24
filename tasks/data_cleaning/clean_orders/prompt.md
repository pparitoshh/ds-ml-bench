# clean_orders

Clean the raw orders data and write the result to `clean_orders.csv`.

## Input

`orders.csv` has the following columns:

- `order_id`: integer order identifier
- `order_date`: order date in mixed formats (`YYYY-MM-DD` and `MM/DD/YYYY`)
- `amount`: numeric amount
- `currency`: indicates the unit of `amount`
  - `USD` (case-insensitive, may have surrounding whitespace): `amount` is in dollars
  - `cents` (case-insensitive, may have surrounding whitespace): `amount` is in cents and must be divided by 100
- `country`: country name with inconsistent casing and surrounding whitespace

## Required output

Write `clean_orders.csv` with exactly these columns:

- `order_id`: integer
- `order_date`: `YYYY-MM-DD` string (e.g. `2023-10-10`)
- `amount`: dollars as a float, rounded to two decimals
- `country`: uppercase string with no surrounding whitespace

Additional requirements:

1. Convert cent amounts to dollars.
2. Normalize country values to uppercase with whitespace stripped.
3. Remove duplicate rows (keep the first occurrence).
4. Do not include the `currency` column in the output.
