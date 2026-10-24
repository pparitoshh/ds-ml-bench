"""Sandbox for extracting and executing generated code."""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeResult:
    """Result of running generated code in a sandbox."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool


def extract_code(model_text: str) -> str:
    """Return the last fenced Python code block, or the full text if none."""
    matches = re.findall(r"```python\s*\n(.*?)\s*```", model_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return model_text.strip()


def _copy_data_dir(data_dir: Path, workdir: Path) -> None:
    if not data_dir.exists():
        return
    for item in data_dir.iterdir():
        dest = workdir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def _stripped_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY"):
            del env[key]
    return env


def run_code(
    code: str,
    data_dir: Path,
    workdir: Path,
    timeout_s: int,
) -> CodeResult:
    """Run code in a fresh work directory with only data_dir copied in."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    _copy_data_dir(data_dir, workdir)

    solution_file = workdir / "solution.py"
    solution_file.write_text(code)

    env = _stripped_env()

    try:
        process = subprocess.run(
            [sys.executable, str(solution_file)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return CodeResult(
            stdout=process.stdout,
            stderr=process.stderr,
            returncode=process.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if not isinstance(stdout, str):
            stdout = stdout.decode("utf-8", errors="replace")
        if not isinstance(stderr, str):
            stderr = stderr.decode("utf-8", errors="replace")
        return CodeResult(
            stdout=stdout,
            stderr=stderr,
            returncode=None,
            timed_out=True,
        )
