"""Tests for leaderboard aggregation and rendering."""

import json
from pathlib import Path

from dsbench.leaderboard import build_leaderboard, render_html, render_markdown


def _write_results(results_dir: Path, label: str, timestamp: str, rating: float,
                   date: str, per_category: dict | None = None) -> None:
    data = {
        "model_label": label,
        "mode": "api",
        "harness": None,
        "model": f"model-for-{label}",
        "date": date,
        "per_task": {},
        "per_category": per_category or {"stats": rating / 100.0},
        "rating": rating,
        "total_cost_usd": 0.5,
        "total_duration_s": 10.0,
    }
    out_dir = results_dir / label / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(data))


def test_build_leaderboard_keeps_latest_per_label(tmp_path):
    results_dir = tmp_path / "results"
    _write_results(results_dir, "model-a", "2026-01-01T00-00-00", 50.0, "2026-01-01T00-00-00")
    _write_results(results_dir, "model-a", "2026-02-01T00-00-00", 80.0, "2026-02-01T00-00-00")
    _write_results(results_dir, "model-b", "2026-02-01T00-00-00", 90.0, "2026-02-01T00-00-00")

    entries = build_leaderboard(results_dir)

    labels = [entry["model_label"] for entry in entries]
    assert labels == ["model-b", "model-a"]  # sorted by rating desc
    ratings = {entry["model_label"]: entry["rating"] for entry in entries}
    assert ratings["model-a"] == 80.0  # latest run, not the best run
    assert ratings["model-b"] == 90.0


def test_render_markdown(tmp_path):
    results_dir = tmp_path / "results"
    _write_results(
        results_dir,
        "model-a",
        "2026-02-01T00-00-00",
        80.0,
        "2026-02-01T00-00-00",
        per_category={"stats": 0.8, "modeling": 0.6},
    )

    text = render_markdown(build_leaderboard(results_dir))

    assert text.startswith("| model | mode | rating |")
    assert "modeling" in text
    assert "model-a" in text
    assert "80.0" in text
    assert "0.60" in text
    assert "0.50" in text  # cost


def test_render_html(tmp_path):
    results_dir = tmp_path / "results"
    _write_results(results_dir, "model-a", "2026-02-01T00-00-00", 80.0, "2026-02-01T00-00-00")

    html = render_html(build_leaderboard(results_dir))

    assert html.startswith("<!DOCTYPE html>")
    assert "<table>" in html
    assert "model-a" in html
    assert "80.0" in html


def test_build_leaderboard_empty(tmp_path):
    assert build_leaderboard(tmp_path / "nothing") == []
