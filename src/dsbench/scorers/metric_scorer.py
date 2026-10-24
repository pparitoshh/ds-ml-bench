"""Metric-based scorer."""

from pathlib import Path

import pandas as pd
import yaml

from dsbench.scorers.base import ScoreResult, Scorer
from dsbench.scorers.metrics import (
    compute_accuracy,
    compute_auc,
    compute_f1,
    compute_recall_at_k,
    compute_rmse,
)


_METRIC_FUNCTIONS = {
    "auc": compute_auc,
    "accuracy": compute_accuracy,
    "rmse": compute_rmse,
    "f1": compute_f1,
    "recall_at_k": compute_recall_at_k,
}


class MetricScorer(Scorer):
    """Score a run by computing a machine-learning metric."""

    def score(self, task, run_result, workdir, answer_text, judge=None) -> ScoreResult:
        workdir = Path(workdir)
        metric_file = task.hidden_dir / "metric.yaml"
        if not metric_file.exists():
            raise FileNotFoundError(f"Metric scorer missing {metric_file}")

        meta = yaml.safe_load(metric_file.read_text()) or {}
        required = ["predictions_file", "labels_file", "metric", "threshold", "higher_is_better"]
        for key in required:
            if key not in meta:
                raise ValueError(f"metric.yaml missing required key: {key}")

        metric_name = meta["metric"]
        if metric_name not in _METRIC_FUNCTIONS:
            raise ValueError(
                f"Unknown metric: {metric_name!r}. "
                f"Available metrics: {sorted(_METRIC_FUNCTIONS)}"
            )

        predictions_path = workdir / meta["predictions_file"]
        labels_path = task.hidden_dir / meta["labels_file"]

        if not predictions_path.exists():
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": f"Predictions file not found: {predictions_path}"},
            )

        preds = pd.read_csv(predictions_path)
        labels = pd.read_csv(labels_path)

        kwargs = {}
        if metric_name == "recall_at_k":
            if "k" not in meta:
                raise ValueError("recall_at_k requires 'k' in metric.yaml")
            kwargs["k"] = int(meta["k"])

        value = _METRIC_FUNCTIONS[metric_name](preds, labels, **kwargs)
        threshold = float(meta["threshold"])
        higher_is_better = bool(meta["higher_is_better"])

        if higher_is_better:
            partial = max(0.0, min(value / threshold, 1.0))
            passed = value >= threshold
        else:
            if value <= 0:
                partial = 0.0
            else:
                partial = max(0.0, min(threshold / value, 1.0))
            passed = value <= threshold

        return ScoreResult(
            score=partial,
            passed=passed,
            details={
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "higher_is_better": higher_is_better,
                "k": kwargs.get("k"),
            },
        )
