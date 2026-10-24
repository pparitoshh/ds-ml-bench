# ds-ml-bench

`ds-ml-bench` is a benchmark harness for evaluating large language models on data-science and machine-learning tasks.

## Design

- Each task is a self-contained Python module under `tasks/`.
- Models connect through a small adapter layer in `src/dsbench/adapters/`.
- A judge adapter scores answers against a rubric.
- Results are written to `results/` (ignored by git).

## Task Categories

1. Data cleaning
2. Exploratory data analysis (EDA)
3. Feature engineering
4. Modeling
5. Evaluation and leakage traps
6. Statistics
7. Ranking and recommendation (for example, MovieLens and Recall@k)
8. ML system design

## Configuration

Copy `config.example.yaml` to `config.yaml` and fill in your API keys through environment variables:

- `MOONSHOT_API_KEY` for Kimi models
- `ANTHROPIC_API_KEY` for the Claude judge

## Running Tests

```bash
uv sync
uv run pytest
```
