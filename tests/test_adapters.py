"""Tests for the model adapter registry and adapters."""

from unittest.mock import Mock, patch

import pytest

from dsbench.adapters import FakeAdapter, KimiAdapter, get_adapter
from dsbench.adapters.base import ModelAdapter


def test_get_fake_adapter():
    adapter = get_adapter("fake", response="hello")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.name == "fake"
    assert adapter.generate("ping") == "hello"


def test_fake_adapter_ignores_system():
    adapter = FakeAdapter(response="ok")
    assert adapter.generate("ping", system="be nice") == "ok"


def test_get_unknown_adapter_raises():
    with pytest.raises(ValueError, match="Unknown adapter: 'missing'"):
        get_adapter("missing")


def test_kimi_adapter_requires_model():
    with pytest.raises(ValueError, match="model name must be provided"):
        KimiAdapter(api_key="test-key")


def _base_url(adapter) -> str:
    return str(adapter._client.base_url).rstrip("/")


def test_kimi_adapter_default_base_url():
    adapter = KimiAdapter(model="moonshot-v1-8k", api_key="test-key")
    assert adapter.name == "kimi"
    assert _base_url(adapter) == "https://api.moonshot.ai/v1"


def test_kimi_adapter_base_url_override():
    adapter = KimiAdapter(
        model="moonshot-v1-8k",
        api_key="test-key",
        base_url="https://custom.example.com/v1",
    )
    assert _base_url(adapter) == "https://custom.example.com/v1"


def test_kimi_generate_uses_openai_client():
    mock_client = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "Kimi response"
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("dsbench.adapters.kimi.OpenAI", return_value=mock_client):
        adapter = KimiAdapter(model="moonshot-v1-8k", api_key="test-key")
        result = adapter.generate("What is 2+2?", system="Be helpful.")

    assert result == "Kimi response"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "moonshot-v1-8k"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
