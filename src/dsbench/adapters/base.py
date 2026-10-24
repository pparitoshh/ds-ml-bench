"""Base adapter interface."""

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract interface for a model that generates text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response for the prompt."""
