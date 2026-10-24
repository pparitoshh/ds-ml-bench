"""Scorer registry for evaluating benchmark runs."""

from dsbench.scorers.base import ScoreResult, Scorer
from dsbench.scorers.judge_scorer import JudgeScorer
from dsbench.scorers.metric_scorer import MetricScorer
from dsbench.scorers.numeric_scorer import NumericScorer
from dsbench.scorers.pytest_scorer import PytestScorer

_REGISTRY = {
    "pytest": PytestScorer,
    "metric": MetricScorer,
    "numeric": NumericScorer,
    "judge": JudgeScorer,
}


def get_scorer(name: str) -> Scorer:
    """Return a scorer instance by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scorer: {name!r}. Available scorers: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


__all__ = ["Scorer", "ScoreResult", "get_scorer"]
