# ab_test_checkout

We ran a 4-week test of the new checkout page. Did it work? Should we ship it?

Use `assignments.csv` and `events.csv`.

Write `answer.json` with these keys:

- `decision`: `"ship"`, `"dont_ship"` or `"rerun"`
- `primary_metric`: `"conversion"`
- `conversion_lift`: relative lift, treatment versus control, after correct cleaning
- `conversion_lift_ci_low`, `conversion_lift_ci_high`: 95% interval of the lift
- `p_value`: for the conversion lift
- `revenue_per_user_lift`: relative lift of revenue per user
- `revenue_per_user_lift_ci_low`, `revenue_per_user_lift_ci_high`: 95% interval
- `srm_detected`: bool
- `excluded_users`: number of distinct users you removed from the analysis for any reason
- `lift_by_platform`: object that maps each platform to its conversion lift

Also write `notes.md` with each issue you found and the reasoning for your decision.

## Input

`assignments.csv` columns: `user_id`, `variant` (control or treatment), `assigned_date`, `platform`, `email`.

`events.csv` columns: `user_id`, `ts`, `event_type` (view or purchase), `revenue` (purchase rows only).
