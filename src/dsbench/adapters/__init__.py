"""Model adapter registry."""

from dsbench.adapters.base import ModelAdapter
from dsbench.adapters.fake import FakeAdapter
from dsbench.adapters.kimi import KimiAdapter

_REGISTRY = {
    "fake": FakeAdapter,
    "kimi": KimiAdapter,
}


def get_adapter(name: str, **cfg) -> ModelAdapter:
    """Return an adapter instance by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown adapter: {name!r}. "
                         f"Available adapters: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


__all__ = ["ModelAdapter", "FakeAdapter", "KimiAdapter", "get_adapter"]
