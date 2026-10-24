"""Judge that scores answers through the Claude Code CLI login (no API key)."""

import json
import os
import subprocess
from pathlib import Path

from dsbench.judge.claude_judge import parse_verdict

JUDGE_CWD = Path.home() / ".cache" / "dsbench" / "judge_cwd"
SYSTEM_PROMPT = "You are a strict grader. Output only the requested JSON."
TIMEOUT_S = 300


class ClaudeCodeJudge:
    """Judge that runs `claude -p` with all tools disabled."""

    def __init__(self, model: str, **kwargs):
        self.model = model

    def _command(self, user_message: str) -> list[str]:
        return [
            "claude",
            "-p",
            user_message,
            "--model",
            self.model,
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--system-prompt",
            SYSTEM_PROMPT,
        ]

    def score(self, prompt: str, answer: str, rubric: str) -> dict:
        """Return a dict with 'score' (0..1) and 'reasoning'."""
        user_message = (
            f"Prompt:\n{prompt}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Rubric:\n{rubric}\n\n"
            "Return only JSON with 'score' (a float between 0 and 1) and "
            "'reasoning' (a string)."
        )
        JUDGE_CWD.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = subprocess.run(
            self._command(user_message),
            capture_output=True,
            text=True,
            cwd=JUDGE_CWD,
            env=env,
            timeout=TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude judge exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:300]}"
            )
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude judge error: {envelope.get('result')}")
        return parse_verdict(str(envelope.get("result", "")))
