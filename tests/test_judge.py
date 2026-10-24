"""Tests for the Claude judge."""

from unittest.mock import Mock, patch

import pytest

from dsbench.judge.claude_judge import ClaudeJudge


def _make_mock_response(text: str):
    block = Mock()
    block.text = text
    response = Mock()
    response.content = [block]
    return response


def test_judge_parses_plain_json():
    response_text = '{"score": 0.85, "reasoning": "Correct and clear."}'
    mock_client = Mock()
    mock_client.messages.create.return_value = _make_mock_response(response_text)

    with patch("dsbench.judge.claude_judge.Anthropic", return_value=mock_client):
        judge = ClaudeJudge(model="claude-test", api_key="test-key")
        result = judge.score(
            prompt="What is 2+2?",
            answer="4",
            rubric="Correctness",
        )

    assert result == {"score": 0.85, "reasoning": "Correct and clear."}
    mock_client.messages.create.assert_called_once()


def test_judge_parses_json_in_markdown_fence():
    response_text = (
        "```json\n"
        '{"score": 0.6, "reasoning": "Partial."}\n'
        "```"
    )
    mock_client = Mock()
    mock_client.messages.create.return_value = _make_mock_response(response_text)

    with patch("dsbench.judge.claude_judge.Anthropic", return_value=mock_client):
        judge = ClaudeJudge(model="claude-test", api_key="test-key")
        result = judge.score("prompt", "answer", "rubric")

    assert result == {"score": 0.6, "reasoning": "Partial."}


def test_judge_clamps_score_above_one():
    mock_client = Mock()
    mock_client.messages.create.return_value = _make_mock_response(
        '{"score": 1.5, "reasoning": "Great."}'
    )

    with patch("dsbench.judge.claude_judge.Anthropic", return_value=mock_client):
        judge = ClaudeJudge(model="claude-test", api_key="test-key")
        result = judge.score("prompt", "answer", "rubric")

    assert result["score"] == 1.0


def test_judge_clamps_score_below_zero():
    mock_client = Mock()
    mock_client.messages.create.return_value = _make_mock_response(
        '{"score": -0.3, "reasoning": "Wrong."}'
    )

    with patch("dsbench.judge.claude_judge.Anthropic", return_value=mock_client):
        judge = ClaudeJudge(model="claude-test", api_key="test-key")
        result = judge.score("prompt", "answer", "rubric")

    assert result["score"] == 0.0


def test_judge_missing_keys_raise():
    mock_client = Mock()
    mock_client.messages.create.return_value = _make_mock_response(
        '{"score": 0.5}'
    )

    with patch("dsbench.judge.claude_judge.Anthropic", return_value=mock_client):
        judge = ClaudeJudge(model="claude-test", api_key="test-key")
        with pytest.raises(KeyError):
            judge.score("prompt", "answer", "rubric")


def test_claude_code_judge_command_and_parse(monkeypatch):
    import json
    import subprocess

    from dsbench.judge.claude_code_judge import ClaudeCodeJudge

    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-pass")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["env"] = cmd, kwargs["env"]
        out = json.dumps({"result": '```json\n{"score": 0.7, "reasoning": "ok"}\n```'})
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    verdict = ClaudeCodeJudge(model="claude-opus-5-5").score("p", "a", "r")
    assert verdict == {"score": 0.7, "reasoning": "ok"}
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert "ANTHROPIC_API_KEY" not in seen["env"]
