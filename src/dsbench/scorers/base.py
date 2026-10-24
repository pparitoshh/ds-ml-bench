"""Base scorer interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of scoring a single benchmark run."""

    score: float
    passed: bool
    details: dict


class Scorer(ABC):
    """Abstract interface for a scorer."""

    @abstractmethod
    def score(self, task, run_result, workdir, answer_text, judge=None) -> ScoreResult:
        """Score a benchmark run."""
