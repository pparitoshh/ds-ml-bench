"""Shared interface for task solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolveResult:
    """Result of a solver invocation."""

    answer_text: str
    returncode: int = 0
    timed_out: bool = False
    duration_s: float = 0.0
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    transcript: str = ""
    meta: dict = field(default_factory=dict)


class Solver(ABC):
    """Abstract interface for a solver that completes a task."""

    name: str

    @abstractmethod
    def solve(self, task, workdir: Path) -> SolveResult:
        """Solve the task, leaving output files in *workdir*."""
