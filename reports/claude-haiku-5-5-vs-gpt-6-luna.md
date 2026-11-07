# ds-ml-bench regular-tier run (2026-11-07)

This run selects the model for the herdr-hive `regular` worker tier. One run per model and task.
Scores are 0-100 (partial credit = tests passed / total, or the task metric).

**Result: claude-haiku-5-5 is the `regular` tier model.**

## Summary

| Model | Harness | Mean score | Total cost | Total time | Complete |
|---|---|---|---|---|---|
| claude-haiku-5-5 | opencode (`opencode-go/claude-haiku-5-5`) | **84.0** | $0.004 | 6m 47s + 4m 32s rerun | 5/5 |
| gpt-6-luna | opencode (`opencode-go/gpt-6-luna`) | **74.0** | $0.002 | 13m 21s | 5/5 |

## Score per task

| Model | clean_orders | ab_test_checkout | fraud_features | churn_classifier | movielens_recall_at_10 |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 85* | 40 | 95 | 100 | 100 |
| gpt-6-luna | 50 | 30 | 90 | 100 | 100 |

Task difficulty: clean_orders, ab_test_checkout, fraud_features = hard; churn_classifier, movielens_recall_at_10 = medium.

\* Rerun score. The first claude-haiku-5-5 clean_orders run scored 0 with status `ok`, because the provider sent
"Unexpected server error". The rerun is in `results/tiers/rerun/`.

## Notes

- Both models are cheap: less than $0.01 for all 5 tasks (cost measured from OpenCode Go).
- claude-haiku-5-5 is about 2 times faster than gpt-6-luna and has a higher score on all 3 hard tasks.
- herdr-hive runs the `regular` tier with the claude harness. This run used the opencode harness, so the
  claude harness score can be different.
- Other tier candidates (not complete at the time of this report): minimax-m3.1-flash (fast tier, running),
  deepseek-v4-pro (think tier, running). kimi-k3.1-pro-go was stopped after 24 min with no task done.
- Runner bug: a provider error scores 0 with status `ok`. The runner must mark it as an error.
- Benchmark commit: 711c95c. Config: config.tiers.yaml.

## Possible bias

- No LLM judge was used: all scores come from pytest tests or metrics.
- One run per task (n = 1): results have high random variation. One task was a rerun.
- Claude models wrote the tasks and reference solutions.
