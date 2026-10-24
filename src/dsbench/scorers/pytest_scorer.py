"""Pytest-based scorer."""

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from dsbench.scorers.base import ScoreResult, Scorer


class PytestScorer(Scorer):
    """Score a run by running hidden pytest tests against the model workdir."""

    def score(self, task, run_result, workdir, answer_text, judge=None) -> ScoreResult:
        workdir = Path(workdir)

        if run_result is None or run_result.timed_out:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "reason": "model_code_timed_out",
                    "stderr": run_result.stderr if run_result else "",
                },
            )

        if run_result.returncode != 0:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "reason": "model_code_failed",
                    "returncode": run_result.returncode,
                    "stderr": run_result.stderr,
                },
            )

        hidden_dir = task.hidden_dir
        test_files = sorted(hidden_dir.glob("test_*.py"))
        if not test_files:
            raise FileNotFoundError(
                f"Pytest scorer found no test_*.py files in {hidden_dir}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for test_file in test_files:
                shutil.copy2(test_file, tmp_path / test_file.name)

            junit_path = tmp_path / "junit.xml"
            env = os.environ.copy()
            env["DSBENCH_WORKDIR"] = str(workdir)
            # Strip API keys from the scoring environment too.
            for key in list(env):
                if key.endswith("_API_KEY"):
                    del env[key]

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"--junitxml={junit_path}",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=task.timeout_s,
                env=env,
            )

            root = ET.parse(junit_path).getroot()
            suites = root.findall("testsuite")
            if not suites:
                raise ValueError("Pytest produced no test results")

            total = sum(int(s.attrib.get("tests", 0)) for s in suites)
            failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
            errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
            skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)

            if total == 0:
                raise ValueError("Pytest produced no test results")

            passed_count = total - failures - errors - skipped
            score = passed_count / total
            passed = failures == 0 and errors == 0

            return ScoreResult(
                score=score,
                passed=passed,
                details={
                    "total": total,
                    "passed": passed_count,
                    "failures": failures,
                    "errors": errors,
                    "skipped": skipped,
                    "pytest_stdout": proc.stdout,
                    "pytest_stderr": proc.stderr,
                },
            )
