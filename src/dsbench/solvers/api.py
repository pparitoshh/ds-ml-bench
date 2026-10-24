"""API-based solver that uses a model adapter."""

import time
from pathlib import Path

from dsbench.adapters.base import ModelAdapter
from dsbench.sandbox import extract_code, run_code
from dsbench.solvers.base import SolveResult, Solver


def _build_file_preview(data_dir: Path) -> str:
    """Return a short listing of data files and their first five lines."""
    if not data_dir.exists():
        return ""

    lines: list[str] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        lines.append(f"\n--- {path.name} ---")
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for i, row in enumerate(f):
                    if i >= 5:
                        break
                    lines.append(row.rstrip("\n"))
        except Exception as exc:  # pragma: no cover
            lines.append(f"[could not read file: {exc}]")
    return "\n".join(lines)


class ApiSolver(Solver):
    """Solver that calls a model adapter and runs the generated code."""

    name = "api"

    def __init__(self, adapter: ModelAdapter):
        self._adapter = adapter

    def solve(self, task, workdir: Path) -> SolveResult:
        """Generate a solution and run it in workdir."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        prompt = task.prompt
        preview = _build_file_preview(task.data_dir)
        if preview:
            prompt = f"{prompt}\n\n## Data files preview\n\n{preview}"

        start = time.perf_counter()
        model_text = self._adapter.generate(prompt)
        duration = time.perf_counter() - start

        if task.answer_type == "text":
            return SolveResult(
                answer_text=model_text,
                returncode=0,
                timed_out=False,
                duration_s=duration,
                transcript=model_text,
            )

        code = extract_code(model_text)
        run_result = run_code(
            code=code,
            data_dir=task.data_dir,
            workdir=workdir,
            timeout_s=task.timeout_s,
        )

        return SolveResult(
            answer_text=model_text,
            returncode=run_result.returncode if run_result.returncode is not None else -1,
            timed_out=run_result.timed_out,
            duration_s=duration,
            transcript=model_text,
            meta={
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
            },
        )
