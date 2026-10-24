# ab_test_simpson

Analyze an A/B test that shows Simpson's paradox.

## Input

`ab_data.csv` has the following columns:

- `segment`: segment name (`A` or `B`)
- `variant`: `control` or `treatment`
- `users`: number of users in the cell
- `conversions`: number of conversions in the cell

## Required output

Write `answer.json` with exactly these keys:

- `overall_lift`: relative lift of treatment vs control across all users
- `segment_a_lift`: relative lift of treatment vs control within segment `A`
- `segment_b_lift`: relative lift of treatment vs control within segment `B`

Compute each lift as `(treatment_rate - control_rate) / control_rate`.
Write the values as floats.
