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
def _percent_row_value(mean: float, std: float) -> str:
    return f"{float(mean) * 100:.1f} ± {float(std) * 100:.1f}"


def load_public_reference(path: Path) -> list[dict]:
    """Load public reference rows from a YAML file, if present."""
    import yaml

    path = Path(path)
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or []
    if isinstance(data, dict):
        data = data.get("references") or []
    if not isinstance(data, list):
        raise ValueError(f"public_reference.yaml must hold a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def benchmark_commit() -> str:
    """Return the short git commit of the benchmark repository."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return "-"
    if out.returncode == 0:
        return out.stdout.strip()
    return "-"


def build_report_markdown(
    entries: list[dict],
    references: list[dict] | None = None,
    commit: str | None = None,
) -> str:
    """Render a model-release-style Markdown report table with footnotes."""
    labels = [entry.get("model_label") or "unknown" for entry in entries]

    tasks_by_category: dict[str, list[str]] = {}
    for entry in entries:
        for task_id, meta in (entry.get("per_task") or {}).items():
            category = (meta or {}).get("category", "other")
            seen = tasks_by_category.setdefault(category, [])
            if task_id not in seen:
                seen.append(task_id)

    ncols = 1 + len(labels)
    lines = ["# ds-ml-bench report", ""]
    lines.append("| benchmark/task | " + " | ".join(labels) + " |")
    lines.append("|" + "|".join(["---"] * ncols) + "|")

    for category in sorted(tasks_by_category):
        label = category.replace("_", " ").title()
        lines.append("| **" + label + "** |" + " |" * (ncols - 1))
        for task_id in sorted(tasks_by_category[category]):
            cells: list[str] = []
            for entry in entries:
                meta = (entry.get("per_task") or {}).get(task_id)
                if meta is None:
                    cells.append("-")
                elif meta.get("mean") is None:
                    cells.append(f"error ({meta.get('n_ok', 0)}/{meta.get('n_runs', '?')} ok)")
                else:
                    value = _percent_row_value(meta.get("mean", 0.0), meta.get("std", 0.0))
                    n_ok, n_runs = meta.get("n_ok"), meta.get("n_runs")
                    if n_ok is not None and n_runs is not None and n_ok < n_runs:
                        value += f" ({n_ok}/{n_runs} ok)"
                    cells.append(value)
            lines.append("| " + task_id + " | " + " | ".join(cells) + " |")

    overall = [
        "**" + f"{float(entry.get('rating', 0.0)):.1f}" + "**" for entry in entries
    ]
    lines.append("| **Overall** | " + " | ".join(overall) + " |")

    cost_cells = [
        (
            f"{float(entry.get('total_cost_usd')):.2f}"
            + (" †" if entry.get("cost_kind") == "notional" else "")
        )
        if isinstance(entry.get("total_cost_usd"), (int, float))
        else "-"
        for entry in entries
    ]
    lines.append("| Cost (USD, measured) | " + " | ".join(cost_cells) + " |")

    time_cells = [
        f"{float(entry.get('total_duration_s')) / 60:.1f}"
        if isinstance(entry.get("total_duration_s"), (int, float))
        else "-"
        for entry in entries
    ]
    lines.append("| Time (min, measured) | " + " | ".join(time_cells) + " |")

    grouped: dict[str, dict[str, dict]] = {}
    for item in references or []:
        benchmark = str(item.get("benchmark", "-"))
        model = str(item.get("model") or item.get("model_label") or "")
        grouped.setdefault(benchmark, {})[model] = item
    for benchmark in sorted(grouped):
        cells = [
            f"{grouped[benchmark][label]['value']} *" if label in grouped[benchmark] else "-"
            for label in labels
        ]
        lines.append(
            "| " + benchmark + " (reported) | " + " | ".join(cells) + " |"
        )

    repo_commit = commit or benchmark_commit()
    lines += ["", "## Footnotes", ""]
    for entry in entries:
        label = entry.get("model_label") or "unknown"
        harness = entry.get("harness") or "-"
        model_id = entry.get("model") or "-"
        runs = entry.get("runs_per_task", 1)
        judge_model = entry.get("judge_model") or "-"
        date = entry.get("date") or "-"
        mode = entry.get("mode", "api")
        lines.append(
            f"- **{label}**: mode {mode}; harness {harness}; provider/model id {model_id}; "
            f"{runs} run(s) per task; judge model {judge_model}; date {date}; "
            f"benchmark commit {repo_commit}"
        )
    if any(entry.get("cost_kind") == "notional" for entry in entries):
        lines.append(
            "- `†` cost is notional (subscription): the total_cost_usd value that "
            "Claude Code reports, not money charged."
        )
    if any(
        meta and meta.get("n_ok") is not None and meta["n_ok"] < meta.get("n_runs", 0)
        for entry in entries
        for meta in (entry.get("per_task") or {}).values()
    ):
        lines.append(
            "- `(k/n ok)` means n runs started and k finished. The other runs failed with a "
            "provider, auth or rate-limit error. Those runs are not scored and are excluded "
            "from the mean and std."
        )
    if grouped:
        lines.append(
            "- `*` marks public reference values: reported by the external source. "
            "All other values are measured by ds-ml-bench."
        )
        for benchmark, by_model in sorted(grouped.items()):
            for model in sorted(by_model):
                item = by_model[model]
                source = item.get("source") or "-"
                lines.append(
                    f"  - {benchmark} / {model}: value {item.get('value')}, source: {source}"
                )
    return "\n".join(lines) + "\n"


def write_report_markdown(
    entries: list[dict],
    results_dir: Path,
    commit: str | None = None,
) -> Path:
    """Write results/report.md from leaderboard entries."""
    results_dir = Path(results_dir)
    references = load_public_reference(results_dir / "public_reference.yaml")
    text = build_report_markdown(entries, references=references, commit=commit)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "report.md"
    report_path.write_text(text)
    return report_path
