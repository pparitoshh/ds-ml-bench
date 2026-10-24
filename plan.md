# Plan for future improvements

This file lists the known problems and the next steps for ds-ml-bench. The items come from the first two benchmark runs (2026-10-24). See `reports/` for the results.

## 1. Fix known bugs

| Item | Problem | Fix |
|---|---|---|
| Runner error status | A provider "Unexpected server error" gave score 0 with status `ok`. | Detect provider errors in the agent output. Set the run status to `error`, so the run is excluded from the mean. |
| Automatic rerun | Provider errors (HTTP 429, server errors) need a manual rerun. | Retry a run that has status `error` automatically, with a limit of 2 retries. |
| ab_test_checkout | The week 4 lift is about 0%, so "rerun" is a defensible decision. The tests accept only "ship". | Make weeks 2-4 stable and positive in `scripts/gen_ab_test_checkout.py`, or accept "rerun". Verify the reference (100%) and naive (< 40%) scores again. |
| Cost display | Very small costs show as "$0.00" in the report. | Show 4 decimal places when the cost is less than $0.01. |

## 2. Make the benchmark fair

1. Give each agent a writable temporary directory inside the sandbox, and set `TMPDIR` to it. DeepSeek Flash scored 0 on movielens_recall_at_10 because OpenCode rejected a write to /tmp.
2. Run all models in the same harness when possible, for example OpenCode for Claude models too. This removes the harness bias.
3. Use a time limit that does not depend on the provider speed. Count the agent steps, or measure the provider time apart from the tool time.
4. Record the time to first token and the tokens per second for each model step, when the harness gives these values.

## 3. Make the tasks harder

Sonnet and Opus scored 100 on churn_classifier and movielens_recall_at_10. These two tasks do not separate strong models.

1. Harden churn_classifier: use a temporal split, target leakage columns, class imbalance, ID-like columns and duplicate customers.
2. Harden movielens_recall_at_10: use a time-based split, exclude seen items, add cold-start users, and require a model that is better than the popularity baseline.
3. Add more tasks: 15-20 tasks in total, in more categories (time series, SQL, causal inference, NLP, model debugging).
4. Let a model that did not write the tasks review them, to decrease the Claude bias in the task design.

## 4. Make the results reliable

1. Use 3 runs per task, and show the mean and the standard deviation.
2. Replace the `notes.md` keyword test with the `claude-code` judge. Use a judge from a different vendor too, and compare the two.
3. Add public reference rows (DS-1000 or DABstep subset), marked "reported". Check the licenses first.

## 5. Protect the tasks

The repository is public, so the tasks can get into model training data.

1. Move the hidden tests and reference solutions to a private repository.
2. Keep only the task prompts and a small sample of the data in the public repository.

## 6. Make the reports easier to use

1. Move the report generator from a local script into `bench report --compare <models>`. It writes `reports/<model-a>-vs-<model-b>-vs-<model-c>.md`.
2. Add a combined leaderboard of all models from all runs to `reports/README.md`.
