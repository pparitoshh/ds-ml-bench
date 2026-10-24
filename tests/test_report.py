"""Tests for the Markdown report table built from fake results."""

from __future__ import annotations

import json
from pathlib import Path

from dsbench.leaderboard import build_leaderboard, build_report_markdown, write_report_markdown


def _entry(
    label: str,
    rating: float,
    per_task: dict,
    cost: float = 1.5,
    duration_s: float = 600.0,
    mode: str = "agent",
    harness: str = "opencode",
    model: str | None = None,
    date: str = "2026-10-24T00-00-00",
    runs: int = 3,
    judge: str = "claude-opus-5-5",
) -> dict:
    per_category: dict[str, list[float]] = {}
    for meta in per_task.values():
        per_category.setdefault(meta["category"], []).append(meta["mean"])
    return {
        "model_label": label,
        "mode": mode,
        "harness": harness,
        "model": model or f"provider/{label}",
        "date": date,
        "runs_per_task": runs,
        "judge_model": judge,
        "per_task": per_task,
        "per_category": {key: sum(vals) / len(vals) for key, vals in per_category.items()},
        "rating": rating,
        "total_cost_usd": cost,
        "total_duration_s": duration_s,
    }


def _fake_entries() -> list[dict]:
    kimi = _entry(
        "kimi-k3.1-pro",
        72.5,
        {
            "mean_price": {"category": "stats", "mean": 0.90, "std": 0.05},
            "clean_orders": {"category": "cleaning", "mean": 0.55, "std": 0.11},
        },
    )
    glm = _entry(
        "glm-5.3",
        50.4,
        {"mean_price": {"category": "stats", "mean": 0.70, "std": 0.10}},
        cost=0.25,
        duration_s=300.0,
    )
    return [kimi, glm]


REFERENCES = [
    {
        "benchmark": "MMLU-Pro",
        "model": "kimi-k3.1-pro",
        "value": 89.2,
        "source": "https://example.com/kimi-model-card",
    }
]


def test_report_table_layout() -> None:
    text = build_report_markdown(
        _fake_entries(), references=REFERENCES, commit="4ac12d8"
    )
    body = text.splitlines()

    assert body[0] == "# ds-ml-bench report"
    assert body[2] == "| benchmark/task | kimi-k3.1-pro | glm-5.3 |"

    assert "| **Overall** | **72.5** | **50.4** |" in body
    assert "| Cost (USD, measured) | 1.50 | 0.25 |" in body
    assert "| Time (min, measured) | 10.0 | 5.0 |" in body

    assert "| **Cleaning** | | |" in body
    assert "| clean_orders | 55.0 ± 11.0 | - |" in body
    assert "| **Stats** | | |" in body
    assert "| mean_price | 90.0 ± 5.0 | 70.0 ± 10.0 |" in body

    assert body.index("| **Cleaning** | | |") < body.index(
        "| clean_orders | 55.0 ± 11.0 | - |"
    )


def test_report_public_reference_rows_and_footnotes() -> None:
    text = build_report_markdown(
        _fake_entries(), references=REFERENCES, commit="4ac12d8"
    )
    assert "| MMLU-Pro (reported) | 89.2 * | - |" in text
    assert "- **kimi-k3.1-pro**: mode agent; harness opencode; provider/model id provider/kimi-k3.1-pro; 3 run(s) per task; judge model claude-opus-5-5; date 2026-10-24T00-00-00; benchmark commit 4ac12d8" in text
    assert "- **glm-5.3**: mode agent; harness opencode; provider/model id provider/glm-5.3; 3 run(s) per task; judge model claude-opus-5-5; date 2026-10-24T00-00-00; benchmark commit 4ac12d8" in text
    assert "reported by the external source" in text
    assert "measured by ds-ml-bench" in text
    assert "MMLU-Pro / kimi-k3.1-pro: value 89.2, source: https://example.com/kimi-model-card" in text


def test_write_report_markdown_reads_yaml(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "public_reference.yaml").write_text(
        "- benchmark: MMLU-Pro\n"
        "  model: kimi-k3.1-pro\n"
        "  value: 89.2\n"
        "  source: https://example.com/kimi-model-card\n"
    )
    report_path = write_report_markdown(
        _fake_entries(), results_dir, commit="4ac12d8"
    )
    text = report_path.read_text()
    assert report_path.name == "report.md"
    assert "| MMLU-Pro (reported) | 89.2 * | - |" in text


def test_report_via_leaderboard_pipeline(tmp_path: Path) -> None:
    """End-to-end: results.json files -> leaderboard entries -> report."""
    for label, rating in (("model-a", 80.0), ("model-b", 90.0)):
        data = _entry(label, rating, {"task_one": {"category": "stats", "mean": rating / 100, "std": 0.0}})
        out = tmp_path / "results" / label / "2026-02-01T00-00-00"
        out.mkdir(parents=True)
        (out / "results.json").write_text(json.dumps(data))
    entries = build_leaderboard(tmp_path / "results")
    text = build_report_markdown(entries, commit="4ac12d8")
    assert "| benchmark/task | model-b | model-a |" in text
    assert "| task_one | 90.0 ± 0.0 | 80.0 ± 0.0 |" in text
    assert "| **Overall** | **90.0** | **80.0** |" in text


def test_report_shows_ok_counts_and_notional_cost():
    from dsbench.leaderboard import build_report_markdown

    entry = {
        "model_label": "m",
        "rating": 50.0,
        "total_cost_usd": 1.5,
        "cost_kind": "notional",
        "per_task": {
            "a": {"category": "c", "mean": 0.5, "std": 0.1, "n_ok": 2, "n_runs": 3},
            "b": {"category": "c", "mean": None, "std": None, "n_ok": 0, "n_runs": 3},
        },
    }
    text = build_report_markdown([entry], commit="abc")
    assert "(2/3 ok)" in text
    assert "error (0/3 ok)" in text
    assert "1.50 †" in text
    assert "notional (subscription)" in text
