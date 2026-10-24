"""Judge-based scorer."""

from pathlib import Path

from dsbench.scorers.base import ScoreResult, Scorer


class JudgeScorer(Scorer):
    """Score a free-text answer using a judge model."""

    def score(self, task, run_result, workdir, answer_text, judge=None) -> ScoreResult:
        rubric_file = task.hidden_dir / "rubric.md"
        if not rubric_file.exists():
            raise FileNotFoundError(f"Judge scorer missing {rubric_file}")

        if judge is None:
            raise ValueError("Judge scorer requires a judge instance")

        rubric = rubric_file.read_text()
        raw = judge.score(task.prompt, answer_text, rubric)
        score = float(raw["score"])
        return ScoreResult(
            score=score,
            passed=score >= 0.5,
            details=raw,
        )
