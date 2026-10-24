# ds-ml-bench first run (2026-10-24)

One run per model and task. Scores are 0-100 (partial credit = tests passed / total, or the task metric).

## Summary

| Model | Harness | Mean score | Total cost | Total time | Complete |
|---|---|---|---|---|---|
| claude-opus-5-5 | claude-code (`claude-opus-5-5`) | **96.0** | $2.55† | 15m 11s | 5/5 |
| glm-5.3 | opencode (`opencode-go/glm-5.3`) | **70.0** | $0.07 | 53m 39s | 5/5 |
| kimi-k3.1-pro | opencode (`moonshotai/kimi-k3.1-pro`) | pending | $0.09 | 54m 13s | 4/5 |

## Score per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-opus-5-5 | 85 | 100 | 95 | 100 | 100 |
| glm-5.3 | 70 | 85 | 95 | 100 | 0 (timeout) |
| kimi-k3.1-pro | 60 | 35 | pending | 0 (timeout) | 0 (timeout) |

## Time per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-opus-5-5 | 2m 53s | 3m 11s | 3m 09s | 2m 39s | 3m 19s |
| glm-5.3 | 13m 15s | 13m 47s | 9m 25s | 2m 12s | 15m 00s |
| kimi-k3.1-pro | 20m 19s | 12m 54s | pending | 6m 00s | 15m 00s |

## Cost per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-opus-5-5 | $0.998 | $0.533 | $0.505 | $0.248 | $0.265 |
| glm-5.3 | $0.024 | $0.019 | $0.013 | $0.003 | $0.013 |
| kimi-k3.1-pro | $0.040 | $0.016 | pending | $0.015 | $0.021 |

Task difficulty: clean_orders, ab_test_checkout, fraud_features = hard; churn_classifier, movielens_recall_at_10 = medium.

† Claude Code cost is notional (the Claude Pro plan pays). Other costs are measured from the provider.

## Notes

- Time limit per task = task `timeout_s` x 3: 1800 s for the hard tasks, 900 s for movielens_recall_at_10, 360 s for churn_classifier. A timeout without `predictions.csv` scores 0.
- glm-5.3 movielens_recall_at_10: 0 (timeout). It built a good EASE model (validation Recall@10 0.099) but wrote no predictions in 900 s.
- kimi-k3.1-pro churn_classifier: 0 (timeout at 360 s). Kimi replies took about 60-80 s per agent step.
- kimi-k3.1-pro used the direct Moonshot API. It returned HTTP 429 "engine is currently overloaded" and one "Unexpected server error". Those runs were discarded and run again.
- Runner bug: the "Unexpected server error" run was scored 0 with status ok, not status error. Fix later.
- Opus scored 100 on 3 of 5 tasks: the tasks are not yet hard enough to separate strong models.
- Benchmark commit: f8a0f82. Config: config.opencode3.yaml (runs_per_task overridden to 1).
