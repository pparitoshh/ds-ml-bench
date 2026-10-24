#!/usr/bin/env python3
"""Smoke test: run OpenCodeSolver on a trivial inline task via bwrap."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dsbench.solvers.harness import OpenCodeSolver

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class InlineTask:
    prompt: str
    timeout_s: int = 120


def main() -> int:
    run_dir = REPO_ROOT / "results" / "smoke_harness"
    workdir = run_dir / "work"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    workdir.mkdir(parents=True)

    task = InlineTask(
        prompt=(
            "Create a file named hello.txt in the current directory.\n"
            "The file must contain exactly the number 42 and nothing else.\n"
            "Then reply with the contents of the file."
        )
    )
    solver = OpenCodeSolver(model="moonshotai/kimi-k2.7-code-highspeed")
    result = solver.solve(task, workdir)

    hello = workdir / "hello.txt"
    hello_ok = hello.is_file() and "42" in hello.read_text()

    print("=== smoke_harness: OpenCodeSolver (moonshotai/kimi-k2.7-code-highspeed) ===")
    print(f"answer_text:     {result.answer_text!r}")
    print(f"returncode:      {result.returncode}")
    print(f"timed_out:       {result.timed_out}")
    print(f"duration_s:      {result.duration_s:.1f}")
    print(f"cost_usd:        {result.cost_usd}")
    print(f"tokens_in:       {result.tokens_in}")
    print(f"tokens_out:      {result.tokens_out}")
    print(f"hello.txt ok:    {hello_ok} (content: {hello.read_text() if hello.is_file() else '<missing>'!r})")
    print()
    print("--- transcript (truncated to 4000 chars) ---")
    print(result.transcript[:4000])

    return 0 if hello_ok and result.returncode == 0 and not result.timed_out else 1


if __name__ == "__main__":
    sys.exit(main())
