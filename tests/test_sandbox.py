"""Tests for code extraction and sandbox execution."""

import os

import pytest

from dsbench.sandbox import CodeResult, extract_code, run_code


def test_extract_code_returns_last_python_block():
    text = (
        "Some explanation.\n"
        "```python\nprint(1)\n```\n"
        "More text.\n"
        "```python\nprint(2)\n```\n"
    )
    assert extract_code(text) == "print(2)"


def test_extract_code_no_fence_returns_whole_text():
    text = "print('hello')\n"
    assert extract_code(text) == "print('hello')"


def test_extract_code_non_python_fence_fallback():
    text = (
        "```json\n"
        '{"score": 1}\n'
        "```\n"
    )
    assert extract_code(text) == "```json\n{\"score\": 1}\n```"


def test_run_code_success(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "values.txt").write_text("1 2 3\n")

    code = (
        "with open('values.txt') as f:\n"
        "    total = sum(int(x) for x in f.read().split())\n"
        "print(total)\n"
    )
    workdir = tmp_path / "work"

    result = run_code(code, data_dir, workdir, timeout_s=10)

    assert isinstance(result, CodeResult)
    assert result.returncode == 0
    assert "6" in result.stdout
    assert result.stderr == ""
    assert result.timed_out is False


def test_run_code_timeout(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    code = "import time\ntime.sleep(10)\n"
    workdir = tmp_path / "work"

    result = run_code(code, data_dir, workdir, timeout_s=1)

    assert result.timed_out is True
    assert result.returncode is None


def test_run_code_hidden_dir_not_copied(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "input.txt").write_text("ok\n")

    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("do not copy\n")

    code = "import os\nprint(sorted(os.listdir('.')))\n"
    workdir = tmp_path / "work"

    result = run_code(code, data_dir, workdir, timeout_s=10)

    assert result.returncode == 0
    assert "secret.txt" not in result.stdout
    assert "input.txt" in result.stdout


def test_run_code_strips_api_key_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_TEST_API_KEY", "super-secret")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    code = (
        "import os\n"
        "for key in os.environ:\n"
        "    if key.endswith('_API_KEY'):\n"
        "        print(f'{key}={os.environ[key]}')\n"
        "print(os.environ.get('SANDBOX_TEST_API_KEY', 'NO_KEY'))\n"
    )
    workdir = tmp_path / "work"

    result = run_code(code, data_dir, workdir, timeout_s=10)

    assert result.returncode == 0
    assert "SANDBOX_TEST_API_KEY" not in result.stdout
    assert "super-secret" not in result.stdout
    assert "NO_KEY" in result.stdout


def test_run_code_does_not_read_parent_env_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthro-secret")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    code = (
        "import os\n"
        "keys = [k for k in os.environ if k.endswith('_API_KEY')]\n"
        "print(','.join(sorted(keys)))\n"
    )
    workdir = tmp_path / "work"

    result = run_code(code, data_dir, workdir, timeout_s=10)

    assert result.returncode == 0
    assert result.stdout.strip() == ""
