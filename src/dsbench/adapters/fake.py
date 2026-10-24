"""Fake adapter for local testing."""

from dsbench.adapters.base import ModelAdapter


class FakeAdapter(ModelAdapter):
    """Adapter that returns a fixed response."""

    def __init__(self, response: str = "This is a canned response.", **kwargs):
        self._response = response

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self._response
