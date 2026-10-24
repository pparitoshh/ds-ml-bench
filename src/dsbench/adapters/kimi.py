"""Moonshot Kimi adapter using the OpenAI-compatible API."""

import os

from openai import OpenAI

from dsbench.adapters.base import ModelAdapter


class KimiAdapter(ModelAdapter):
    """Adapter for Moonshot Kimi models via an OpenAI-compatible client."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        if not model:
            raise ValueError("A model name must be provided in the adapter config.")
        self._model = model

        api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        base_url = (
            base_url
            or os.environ.get("MOONSHOT_BASE_URL")
            or "https://api.moonshot.ai/v1"
        )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "kimi"

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
