"""Numeric answer scorer."""

import json
import math
from pathlib import Path

import yaml

from dsbench.scorers.base import ScoreResult, Scorer


class NumericScorer(Scorer):
    """Score a run by comparing numeric answers within a tolerance."""

    def score(self, task, run_result, workdir, answer_text, judge=None) -> ScoreResult:
        workdir = Path(workdir)
        answer_file = workdir / "answer.json"
        expected_file = task.hidden_dir / "answer.yaml"

        if not expected_file.exists():
            raise FileNotFoundError(f"Numeric scorer missing {expected_file}")

        meta = yaml.safe_load(expected_file.read_text()) or {}
        if "answers" not in meta:
            raise ValueError("answer.yaml missing required key: answers")

        expected_answers = meta["answers"]
        rel_tol = float(meta.get("rel_tol", 1e-5))

        if not answer_file.exists():
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": f"answer.json not found in {workdir}"},
            )

        model_answers = json.loads(answer_file.read_text())

        per_key: dict[str, dict] = {}
        correct = 0
        for key, expected in expected_answers.items():
            actual = model_answers.get(key)
            ok = actual is not None and math.isclose(
                float(actual), float(expected), rel_tol=rel_tol
            )
            per_key[key] = {"expected": expected, "actual": actual, "correct": ok}
            if ok:
                correct += 1

        score = correct / len(expected_answers)
        return ScoreResult(
            score=score,
            passed=score == 1.0,
            details={"per_key": per_key, "rel_tol": rel_tol},
        )
