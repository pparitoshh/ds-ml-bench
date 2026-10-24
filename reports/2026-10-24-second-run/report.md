# ds-ml-bench second run (2026-10-24)

One run per model and task. Scores are 0-100 (partial credit = tests passed / total, or the task metric).

## Summary

| Model | Harness | Mean score | Total cost | Total time | Complete |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | claude-code (`claude-sonnet-5-5`) | **92.0** | $0.87† | 5m 56s | 5/5 |
| deepseek-v4.1-flash | opencode (`opencode-go/deepseek-v4.1-flash`) | pending | $0.00 | 7m 01s | 3/5 |
| glm-5.3-flash | opencode (`opencode-go/glm-5.3-flash`) | pending | $0.00 | 2m 42s | 1/5 |

## Score per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | 75 | 90 | 95 | 100 | 100 |
| deepseek-v4.1-flash | pending | pending | 95 | 100 | 0 |
| glm-5.3-flash | pending | pending | pending | 100 | pending |

## Time per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | 1m 29s | 1m 11s | 1m 43s | 0m 25s | 1m 08s |
| deepseek-v4.1-flash | pending | pending | 3m 24s | 3m 02s | 0m 35s |
| glm-5.3-flash | pending | pending | pending | 2m 42s | pending |

## Cost per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | $0.294 | $0.180 | $0.196 | $0.091 | $0.105 |
| deepseek-v4.1-flash | pending | pending | $0.002 | $0.000 | $0.000 |
| glm-5.3-flash | pending | pending | pending | $0.001 | pending |

Task difficulty: clean_orders, ab_test_checkout, fraud_features = hard; churn_classifier, movielens_recall_at_10 = medium.

† Claude Code cost is notional (the Claude Pro plan pays). Other costs are measured from the provider.

## Notes

- Time limit per task = task `timeout_s` x 3: 1800 s for the hard tasks, 900 s for movielens_recall_at_10, 360 s for churn_classifier. A timeout without `predictions.csv` scores 0.
- This is a partial report. Runs still in progress are marked pending. The report will be updated when all 15 results exist.
- deepseek-v4.1-flash movielens_recall_at_10: 0 in 35 s. The agent tried to write to /tmp. OpenCode auto-rejected the external_directory permission and the run stopped. The result is kept.
- OpenCode Go reports very small costs for the flash models (less than $0.01 per task).
- Sonnet scored 100 on 2 of 5 tasks: the tasks are not yet hard enough to separate strong models.
- Benchmark commit: 7d0beaf. Config: config.flash3.yaml.

## Possible bias

- No LLM judge was used: all scores come from pytest tests or metrics (AUC, Recall@10).
- Harness: Sonnet runs in Claude Code (its native harness); DeepSeek and GLM run in OpenCode. OpenCode rejects writes outside the work directory, which hurt DeepSeek on one task.
- Claude models wrote the tasks and reference solutions, and the hard-task target was checked with an Opus smoke run.
- One run per task (n = 1): results have high random variation.
