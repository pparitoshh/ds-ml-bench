"""Leaderboard aggregation and rendering."""

import html
import json
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    """Load every results.json under results_dir/<label>/<timestamp>/."""
    results: list[dict] = []
    if not results_dir.exists():
        return results
    for path in sorted(results_dir.glob("*/*/results.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data.setdefault("model_label", path.parent.parent.name)
        data["_path"] = str(path)
        results.append(data)
    return results


def build_leaderboard(results_dir: Path) -> list[dict]:
    """Return the latest result per model label, sorted by rating."""
    by_label: dict[str, dict] = {}
    for data in _load_results(Path(results_dir)):
        label = data.get("model_label") or "unknown"
        current = by_label.get(label)
        if current is None or str(data.get("date", "")) > str(current.get("date", "")):
            by_label[label] = data
    return sorted(by_label.values(), key=lambda d: d.get("rating", 0.0), reverse=True)


def _categories(entries: list[dict]) -> list[str]:
    categories: set[str] = set()
    for entry in entries:
        categories.update((entry.get("per_category") or {}).keys())
    return sorted(categories)


def render_markdown(entries: list[dict]) -> str:
    """Render the leaderboard as a GitHub-flavored Markdown table."""
    categories = _categories(entries)
    header = ["model", "mode", "rating", *categories, "cost_usd", "date"]
    rows = []
    for entry in entries:
        row = [
            entry.get("model_label", "unknown"),
            entry.get("mode", "api"),
            f"{float(entry.get('rating', 0.0)):.1f}",
        ]
        per_category = entry.get("per_category") or {}
        row.extend(
            f"{float(per_category.get(cat, 0.0)):.2f}" for cat in categories
        )
        cost = entry.get("total_cost_usd")
        row.append(f"{cost:.2f}" if isinstance(cost, (int, float)) else "-")
        row.append(entry.get("date", "-"))
        rows.append(row)

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def render_html(entries: list[dict]) -> str:
    """Render the leaderboard as a simple static HTML page."""
    categories = _categories(entries)
    header_cells = ["Model", "Mode", "Rating", *[c.title() for c in categories],
                    "Cost (USD)", "Date"]

    body_rows = []
    for entry in entries:
        per_category = entry.get("per_category") or {}
        cells = [
            html.escape(str(entry.get("model_label", "unknown"))),
            html.escape(str(entry.get("mode", "api"))),
            f"{float(entry.get('rating', 0.0)):.1f}",
        ]
        cells.extend(
            f"{float(per_category.get(cat, 0.0)):.2f}" for cat in categories
        )
        cost = entry.get("total_cost_usd")
        cells.append(f"{cost:.2f}" if isinstance(cost, (int, float)) else "-")
        cells.append(html.escape(str(entry.get("date", "-"))))
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    head_row = "<tr>" + "".join(f"<th>{h}</th>" for h in header_cells) + "</tr>"
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>ds-ml-bench leaderboard</title>\n<style>\n"
        "body { font-family: sans-serif; margin: 2rem; }\n"
        "table { border-collapse: collapse; }\n"
        "th, td { border: 1px solid #999; padding: 0.4rem 0.8rem; }\n"
        "th { background: #eee; }\n"
        "</style>\n</head>\n<body>\n<h1>ds-ml-bench leaderboard</h1>\n"
        f"<table>\n{head_row}\n" + "\n".join(body_rows) + "\n</table>\n"
        "</body>\n</html>\n"
    )
