"""Tests for the bwrap sandbox command builder and harness solvers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from dsbench.solvers.harness import (
    DEFAULT_STATE_BASE,
    ClaudeCodeSolver,
    HarnessConfig,
    DEFAULT_PATH,
    HOME_AGENT,
    MountSpec,
    OpenCodeSolver,
    sandbox_cmd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeTask:
    prompt: str = "Solve it."
    timeout_s: int = 10


def _make_cmd(tmp_path: Path) -> list[str]:
    mounts = {
        "/home/agent": MountSpec(str(tmp_path / "state_home"), readonly=False),
        "/home/agent/.opencode": MountSpec(str(tmp_path / "opencode"), readonly=True),
        "/opt/dsvenv": MountSpec(str(tmp_path / "venv"), readonly=True),
    }
    return sandbox_cmd(
        tmp_path / "work",
        mounts,
        {"PATH": DEFAULT_PATH, "MOONSHOT_API_KEY": "k-test"},
    )


@pytest.fixture
def cmd(tmp_path: Path) -> list[str]:
    return _make_cmd(tmp_path)


def test_required_flags(cmd: list[str]) -> None:
    for flag in (
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "usr/lib64", "/lib64", "usr/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--bind", "--chdir", "/work",
        "--unshare-all", "--share-net", "--die-with-parent",
        "--clearenv",
        "--setenv", "HOME", HOME_AGENT,
    ):
        assert flag in cmd, f"missing {flag!r}"


def test_env_is_cleared_then_setenv(cmd: list[str]) -> None:
    first_setenv = cmd.index("--setenv")
    assert cmd[first_setenv + 1 : first_setenv + 3] == ["HOME", HOME_AGENT]
    assert cmd.count("--clearenv") == 1
    assert cmd.index("--clearenv") < first_setenv


def test_allowed_env_keys_passed_others_dropped(tmp_path: Path) -> None:
    cmd = sandbox_cmd(
        tmp_path / "work",
        {},
        {"PATH": DEFAULT_PATH, "MOONSHOT_API_KEY": "k-test"},
    )
    core = cmd[cmd.index("--clearenv") :]
    assert "MOONSHOT_API_KEY" in core and "k-test" in core
    assert "HOME" in core and "/home/agent" in core
    assert "ANTHROPIC_API_KEY" not in core


def test_workdir_mounted_at_work(tmp_path: Path) -> None:
    cmd = sandbox_cmd(tmp_path / "work", {}, {})
    workdir = str((tmp_path / "work").resolve())
    assert "--bind" in cmd and workdir in cmd
    idx = cmd.index("--bind")
    assert cmd[idx + 2 : idx + 4] == ["/work", "--chdir"]
    assert cmd[-1] == "/work" or "HOME" in cmd


def test_never_mounts_repo_or_win_drive(tmp_path: Path) -> None:
    cmd = _make_cmd(tmp_path)
    joined = " ".join(cmd)
    assert "win_drive" not in joined
    assert str(REPO_ROOT) not in joined


def test_child_mount_ordered_after_parent(tmp_path: Path) -> None:
    mounts = {
        "/home/agent/.opencode": MountSpec(str(tmp_path), readonly=True),
        "/home/agent": MountSpec(str(tmp_path), readonly=False),
    }
    cmd = sandbox_cmd(tmp_path / "work", mounts, {})
    assert cmd.index(str(tmp_path)) < cmd.index("/home/agent/.opencode")


class FakePopen:
    captured: dict = {}

    def __init__(self, command, **kwargs):
        FakePopen.captured["command"] = command

    def communicate(self, timeout=None):
        return "", ""

    returncode = 0


def test_solver_builds_sandbox_without_exposing_repo(tmp_path, monkeypatch) -> None:
    FakePopen.captured = {}
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    solver = OpenCodeSolver(model="moonshotai/kimi-k3")
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = solver.solve(FakeTask(), workdir)

    command = FakePopen.captured["command"]
    joined = " ".join(command)
    assert f"--ro-bind {REPO_ROOT} " not in joined
    assert f"--bind {REPO_ROOT} " not in joined
    assert "--ro-bind" in joined
    assert "/opt/dsvenv" in command
    assert "--chdir" in command and "/work" in command
    assert "--model" in command and "moonshotai/kimi-k3" in command
    assert "--format" in command and "json" in command
    assert result.returncode == 0


@pytest.mark.integration
def test_bwrap_isolation(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("42\n")
    cmd = sandbox_cmd(tmp_path, {}, {"PATH": DEFAULT_PATH})
    cmd.extend(
        [
            "--",
            "sh",
            "-c",
            "ls /work; ls /home; test -e /home/paritosh && echo LEAKED || echo ISOLATED",
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "hello.txt" in proc.stdout
    assert "ISOLATED" in proc.stdout
    assert "LEAKED" not in proc.stdout
    assert "paritosh" not in proc.stdout


def _fake_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text('{"tok": "x"}')
    (home / ".claude" / "history.jsonl").write_text("secret history")
    (home / ".claude" / "projects").mkdir()
    (home / ".claude.json").write_text("{}")
    (home / ".local" / "share" / "opencode").mkdir(parents=True)
    (home / ".local" / "share" / "opencode" / "auth.json").write_text("{}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.parametrize("solver_cls", [OpenCodeSolver, ClaudeCodeSolver])
def test_state_dir_removed_and_not_under_results(tmp_path, monkeypatch, solver_cls) -> None:
    _fake_home(tmp_path, monkeypatch)
    state_base = tmp_path / "cache" / "harness_state"
    seen: dict = {}

    class SpyPopen(FakePopen):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            seen["dirs"] = [p for p in state_base.iterdir()]
            seen["files"] = sorted(
                str(p.relative_to(seen["dirs"][0])) for p in seen["dirs"][0].rglob("*") if p.is_file()
            )

    monkeypatch.setattr(subprocess, "Popen", SpyPopen)
    venv = tmp_path / "venv"
    venv.mkdir()
    results = tmp_path / "results" / "m" / "d" / "t" / "run_0"
    solver = solver_cls(model="m", config=HarnessConfig(state_root=state_base, venv_path=venv))
    solver.solve(FakeTask(), results)

    assert len(seen["dirs"]) == 1  # state existed during the run
    assert not list(state_base.iterdir())  # and is gone afterwards
    assert not list((tmp_path / "results").rglob(".harness_state"))
    if solver_cls is ClaudeCodeSolver:
        assert "home/agent/.claude/.credentials.json" in seen["files"]
        assert "home/agent/.claude.json" in seen["files"]
        assert not any("history" in f or "projects" in f for f in seen["files"])


def test_state_dir_removed_when_run_raises(tmp_path, monkeypatch) -> None:
    _fake_home(tmp_path, monkeypatch)
    state_base = tmp_path / "state"
    venv = tmp_path / "venv"
    venv.mkdir()

    class Boom(FakePopen):
        def __init__(self, command, **kwargs):
            raise OSError("no bwrap")

    monkeypatch.setattr(subprocess, "Popen", Boom)
    solver = ClaudeCodeSolver(model="m", config=HarnessConfig(state_root=state_base, venv_path=venv))
    with pytest.raises(OSError):
        solver.solve(FakeTask(), tmp_path / "work")
    assert not list(state_base.iterdir())


def test_default_state_base_is_under_cache() -> None:
    assert DEFAULT_STATE_BASE == Path.home() / ".cache" / "dsbench" / "harness_state"


def test_claude_provider_error_detected() -> None:
    solver = ClaudeCodeSolver(model="m")
    auth = json.dumps({"is_error": True, "api_error_status": 401, "result": "Invalid API key"})
    assert solver._parse(auth, "", 1, False, 1.0).provider_error
    assert solver._parse("", "", 1, False, 1.0).provider_error  # non-zero, no output
    assert solver._parse("", "HTTP 429 rate limit", 1, False, 1.0).provider_error
    ok = json.dumps({"result": "done", "total_cost_usd": 0.5})
    result = solver._parse(ok, "", 0, False, 1.0)
    assert result.provider_error is None and result.cost_usd == 0.5
    assert solver._parse("", "", 1, True, 1.0).provider_error is None  # timeout is not a provider error
