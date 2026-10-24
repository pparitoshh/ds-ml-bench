"""Base solver interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolveResult:
    """Result of a single solver run."""

    answer_text: str
    returncode: int = 0
    timed_out: bool = False
    duration_s: float = 0.0
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    transcript: str = ""
    meta: dict = field(default_factory=dict)
    provider_error: str | None = None


class Solver(ABC):
    """Abstract solver that produces an answer for a task."""

    name: str

    @abstractmethod
    def solve(self, task, workdir: Path) -> SolveResult:
        """Solve the task in the provided workdir.

        The workdir already contains a copy of the task data files and
        prompt.md. The solver must leave output files in workdir.
        """
