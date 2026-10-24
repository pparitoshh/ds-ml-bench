"""Solvers that run coding-agent harnesses inside a bwrap sandbox."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsbench.solvers.base import SolveResult, Solver

HOME_AGENT = "/home/agent"
DEFAULT_PATH = "/opt/dsvenv/bin:/usr/local/bin:/usr/bin:/bin"
DEFAULT_TIMEOUT_FACTOR = 3.0
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_BASE = Path.home() / ".cache" / "dsbench" / "harness_state"

PROVIDER_ERROR_PATTERN = re.compile(
    r"\b(401|403|429|529)\b|unauthori[sz]ed|authentication|invalid api key|"
    r"rate.?limit|overloaded|usage limit|not logged in|please run /login|"
    r"credit balance|quota",
    re.IGNORECASE,
)

AGENT_INSTRUCTIONS = (
    "\n\n---\nCareful instructions for the agent:\n"
    "Work in the current directory.\n"
    "The files listed in the prompt are here.\n"
    "Write the required output files here.\n"
    "You may write and run Python (python3 and common DS libraries such as pandas, numpy and scikit-learn are available).\n"
)


@dataclass(frozen=True)
class MountSpec:
    """A host path bound into the sandbox at a sandbox path."""

    host_path: str
    readonly: bool = True


@dataclass
class HarnessConfig:
    """Configuration for harness solvers."""

    timeout_factor: float = DEFAULT_TIMEOUT_FACTOR
    venv_path: Path | None = None
    allowed_env: tuple[str, ...] = (
        "MOONSHOT_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TERM",
        "LANG",
        "NO_COLOR",
    )
    state_root: Path | None = None


def sandbox_cmd(
    workdir: Path,
    harness_home_mounts: dict[str, MountSpec],
    env: dict[str, str],
) -> list[str]:
    """Build the bwrap command prefix for a sandbox rooted on *workdir*."""
    workdir = Path(workdir).resolve()
    cmd: list[str] = ["bwrap"]
    cmd.extend(["--ro-bind", "/usr", "/usr"])
    cmd.extend(["--symlink", "usr/lib", "/lib"])
    cmd.extend(["--symlink", "usr/lib64", "/lib64"])
    cmd.extend(["--symlink", "usr/bin", "/bin"])
    cmd.extend(["--ro-bind", "/etc", "/etc"])
    cmd.extend(["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"])
    cmd.extend(["--dev", "/dev"])
    cmd.extend(["--proc", "/proc"])
    cmd.extend(["--tmpfs", "/tmp"])
    cmd.extend(["--tmpfs", "/home"])
    cmd.extend(["--bind", str(workdir), "/work"])
    cmd.extend(["--chdir", "/work"])

    ordered = sorted(
        harness_home_mounts.items(),
        key=lambda item: (item[0].count("/"), item[0]),
    )
    for sandbox_path, spec in ordered:
        flag = "--ro-bind" if spec.readonly else "--bind"
        cmd.extend([flag, spec.host_path, sandbox_path])

    cmd.extend(["--unshare-all", "--share-net", "--die-with-parent"])
    cmd.extend(["--clearenv"])
    cmd.extend(["--setenv", "HOME", HOME_AGENT])
    cmd.extend(["--setenv", "PATH", env.get("PATH", DEFAULT_PATH)])
    for key in sorted(env):
        if key == "PATH":
            continue
        cmd.extend(["--setenv", key, env[key]])
    return cmd


def _parse_pyvenv_home(venv: Path) -> Path | None:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    match = re.search(r"^home\s*=\s*(.+)$", cfg.read_text(), re.MULTILINE)
    if not match:
        return None
    home = Path(match.group(1).strip())
    return home.parent if home.name == "bin" else home


def _detect_provider_error(stdout: str, stderr: str, returncode: int | None) -> str | None:
    """Return a message when the run failed for a provider, auth or rate-limit reason."""
    if returncode in (0, None):
        return None
    combined = f"{stdout}\n{stderr}"
    if not stdout.strip() or PROVIDER_ERROR_PATTERN.search(combined):
        detail = (stderr.strip() or stdout.strip())[:300]
        return f"exit {returncode}: {detail}" if detail else f"exit {returncode} with no output"
    return None


class HarnessSolver(Solver):
    """Common machinery to run an agent binary inside bwrap."""

    name = "harness"

    def __init__(self, model: str, config: HarnessConfig | None = None):
        self.model = model
        self.config = config or HarnessConfig()
        self.last_command: list[str] = []
        self._state_root: Path | None = None

    def solve(self, task, workdir: Path) -> SolveResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        prompt = task.prompt + AGENT_INSTRUCTIONS
        timeout_s = self._timeout(task)
        try:
            command = self._sandbox_prefix(workdir) + ["--"] + self._agent_argv(prompt)
            self.last_command = list(command)
            started = time.monotonic()
            returncode, stdout, stderr, timed_out = self._run(command, timeout_s)
            duration = time.monotonic() - started
        finally:
            self._cleanup_state()
        result = self._parse(stdout, stderr, returncode, timed_out, duration)
        transcript_file = workdir / ".harness_transcript.txt"
        transcript_file.write_text(
            f"command: {subprocess.list2cmdline(command)}\n\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n"
        )
        result.meta["model"] = self.model
        result.meta["harness"] = self.name
        result.meta["timeout_s"] = timeout_s
        result.meta["command"] = command
        return result

    def _cleanup_state(self) -> None:
        """Delete the per-run state dir, which holds copied auth files."""
        if self._state_root is not None:
            shutil.rmtree(self._state_root, ignore_errors=True)
            self._state_root = None

    def _timeout(self, task) -> float:
        return float(task.timeout_s) * self.config.timeout_factor

    def _agent_argv(self, prompt: str) -> list[str]:
        raise NotImplementedError

    def _parse(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        timed_out: bool,
        duration: float,
    ) -> SolveResult:
        raise NotImplementedError

    def _sandbox_prefix(self, workdir: Path) -> list[str]:
        mounts = self._prepare(workdir)
        env = self._env()
        return sandbox_cmd(workdir, mounts, env)

    def _env(self) -> dict[str, str]:
        env = {"PATH": DEFAULT_PATH}
        for key in self.config.allowed_env:
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _prepare(self, workdir: Path) -> dict[str, MountSpec]:
        base = self.config.state_root or DEFAULT_STATE_BASE
        state_root = base / uuid.uuid4().hex
        state_root.mkdir(parents=True, mode=0o700)
        self._state_root = state_root
        state_home = state_root / "home" / "agent"
        (state_home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
        (state_home / ".local" / "share").mkdir(parents=True, exist_ok=True)
        (state_home / ".config").mkdir(parents=True, exist_ok=True)

        mounts: dict[str, MountSpec] = {}
        mounts[HOME_AGENT] = MountSpec(str(state_home), readonly=False)

        venv = self.config.venv_path or REPO_ROOT / ".venv"
        if not venv.is_dir():
            raise FileNotFoundError(f"venv not found: {venv}")
        mounts["/opt/dsvenv"] = MountSpec(str(venv), readonly=True)
        python_root = _parse_pyvenv_home(venv)
        if python_root is not None and python_root.is_dir():
            mounts[str(python_root)] = MountSpec(str(python_root), readonly=True)

        self._prepare_harness(state_home, mounts)
        return mounts

    def _prepare_harness(self, state_home: Path, mounts: dict[str, MountSpec]) -> None:
        pass

    def _run(self, command: list[str], timeout_s: float) -> tuple[int, str, str, bool]:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            return process.returncode, stdout, stderr, False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            return process.returncode, stdout or "", stderr or "", True


class ClaudeCodeSolver(HarnessSolver):
    """Runs Anthropic Claude Code inside the sandbox."""

    name = "claude-code"

    def _prepare_harness(self, state_home: Path, mounts: dict[str, MountSpec]) -> None:
        home = Path.home()
        claude_bin = (home / ".local" / "bin" / "claude").resolve()
        if claude_bin.is_file():
            versions_dir = claude_bin.parent
            sandbox_versions = f"{HOME_AGENT}/.local/share/claude/versions"
            mounts[sandbox_versions] = MountSpec(str(versions_dir), readonly=True)
            link = state_home / ".local" / "bin" / "claude"
            link.symlink_to(f"{sandbox_versions}/{claude_bin.name}")

        credentials = home / ".claude" / ".credentials.json"
        if credentials.is_file():
            state_claude = state_home / ".claude"
            state_claude.mkdir(parents=True, exist_ok=True)
            shutil.copy2(credentials, state_claude / ".credentials.json")
        claude_json = home / ".claude.json"
        if claude_json.is_file():
            shutil.copy2(claude_json, state_home / ".claude.json")

    def _agent_argv(self, prompt: str) -> list[str]:
        return [
            f"{HOME_AGENT}/.local/bin/claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--dangerously-skip-permissions",
        ]

    def _parse(self, stdout, stderr, returncode, timed_out, duration) -> SolveResult:
        data: dict[str, Any] = {}
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            start = stdout.find("{")
            end = stdout.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(stdout[start:end + 1])
                except json.JSONDecodeError:
                    data = {}
        usage = data.get("usage") or {}
        provider_error = None
        if not timed_out:
            if data.get("is_error") and PROVIDER_ERROR_PATTERN.search(
                f"{data.get('api_error_status')} {data.get('result')}"
            ):
                provider_error = str(data.get("result") or data.get("api_error_status"))
            else:
                provider_error = _detect_provider_error(stdout, stderr, returncode)
        return SolveResult(
            provider_error=provider_error,
            answer_text=str(data.get("result") or stdout.strip()),
            returncode=returncode if returncode is not None else 1,
            timed_out=timed_out,
            duration_s=duration,
            cost_usd=data.get("total_cost_usd"),
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            transcript=stdout + (f"\n--- stderr ---\n{stderr}" if stderr else ""),
            meta={"num_turns": data.get("num_turns")},
        )


class OpenCodeSolver(HarnessSolver):
    """Runs opencode inside the sandbox."""

    name = "opencode"

    def _prepare_harness(self, state_home: Path, mounts: dict[str, MountSpec]) -> None:
        home = Path.home()
        opencode_bin = (home / ".opencode" / "bin" / "opencode").resolve()
        if opencode_bin.is_file():
            install_root = opencode_bin.parent.parent
            if install_root != home:
                mounts[f"{HOME_AGENT}/.opencode"] = MountSpec(str(install_root), readonly=True)
        config_dir = home / ".config" / "opencode"
        if config_dir.is_dir():
            mounts[f"{HOME_AGENT}/.config/opencode"] = MountSpec(str(config_dir), readonly=True)
        auth_json = home / ".local" / "share" / "opencode" / "auth.json"
        state_oc = state_home / ".local" / "share" / "opencode"
        state_oc.mkdir(parents=True, exist_ok=True)
        if auth_json.is_file():
            shutil.copy2(auth_json, state_oc / "auth.json")

    def _agent_argv(self, prompt: str) -> list[str]:
        return [
            f"{HOME_AGENT}/.opencode/bin/opencode",
            "run",
            "--model",
            self.model,
            "--format",
            "json",
            prompt,
        ]

    def _parse(self, stdout, stderr, returncode, timed_out, duration) -> SolveResult:
        events = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        texts: list[str] = []
        tokens_in = None
        tokens_out = None
        cost = None

        def walk(node: Any) -> None:
            nonlocal tokens_in, tokens_out, cost
            if isinstance(node, dict):
                if node.get("type") == "text" and isinstance(node.get("text"), str):
                    texts.append(node["text"])
                if "part" in node and isinstance(node["part"], dict):
                    part = node["part"]
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                tok = node.get("tokens")
                if isinstance(tok, dict):
                    if isinstance(tok.get("input"), (int, float)):
                        tokens_in = int(tok["input"])
                    if isinstance(tok.get("output"), (int, float)):
                        tokens_out = int(tok["output"])
                if isinstance(node.get("cost"), (int, float)):
                    cost = float(node["cost"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        for event in events:
            walk(event)

        if texts:            answer = texts[-1]
        elif returncode == 0:
            answer = stdout.strip()
        else:
            answer = ""
            if stderr.strip():
                answer = stderr.strip()
        return SolveResult(
            provider_error=None if timed_out else _detect_provider_error(stdout, stderr, returncode),
            answer_text=answer,
            returncode=returncode if returncode is not None else 1,
            timed_out=timed_out,
            duration_s=duration,
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            transcript=stdout + (f"\n--- stderr ---\n{stderr}" if stderr else ""),
            meta={"events": len(events)},
        )

HARNESS_SOLVERS: dict[str, type[HarnessSolver]] = {
    "opencode": OpenCodeSolver,
    "claude-code": ClaudeCodeSolver,
}


def get_harness_solver(model_entry: dict) -> Solver:
    """Map an agent-mode config model entry to a harness solver.

    Entry keys: label, harness ('opencode' or 'claude-code'), model
    (provider/model id), and optional timeout_factor.
    """
    harness = model_entry.get("harness", "opencode")
    solver_cls = HARNESS_SOLVERS.get(harness)
    if solver_cls is None:
        raise ValueError(
            f"model entry {model_entry.get('label')!r}: harness must be "
            f"{sorted(HARNESS_SOLVERS)}, got {harness!r}"
        )
    model = model_entry.get("model")
    if not model:
        raise ValueError(
            f"model entry {model_entry.get('label')!r}: agent mode requires 'model'"
        )
    config = HarnessConfig(
        timeout_factor=float(model_entry.get("timeout_factor", DEFAULT_TIMEOUT_FACTOR))
    )
    return solver_cls(model=model, config=config)
