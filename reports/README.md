# Reports

This folder keeps one report for each benchmark run. Raw run output stays in `results/` (git-ignored).

Each run has its own folder `YYYY-MM-DD-<name>/` with:

- `report.md`: score, cost and time per model and task, the config, and notes.
- `summary.json`: the numbers only, copied from the `results.json` files.

Do not overwrite an old report. Add a new folder for each run.

## Index

| Date | Report | Models | Tasks | Runs per task | Notes |
|---|---|---|---|---|---|
| 2026-10-24 | [first run](2026-10-24-first-run/report.md) | claude-opus-5-5, glm-5.3, kimi-k3.1-pro | 5 (3 hard, 2 medium) | 1 | Opus 96.0, GLM 70.0, Kimi 19.0 |
| 2026-10-24 | [second run](2026-10-24-second-run/report.md) | claude-sonnet-5-5, deepseek-v4.1-flash, glm-5.3-flash | 5 (3 hard, 2 medium) | 1 | Partial: Sonnet 92.0, others pending |
