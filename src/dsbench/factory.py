"""Configuration loading and solver factory."""

import importlib
from pathlib import Path

import yaml

from dsbench.adapters import get_adapter
from dsbench.solvers.api import ApiSolver
from dsbench.solvers.base import Solver

HARNESS_MODULE = "dsbench.solvers.harness"


def load_config(path: Path | str) -> dict:
    """Load and validate a benchmark configuration file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = yaml.safe_load(path.read_text()) or {}
    if not config.get("models"):
        raise ValueError(f"{path}: config must define a non-empty 'models' list")
    judge = config.get("judge")
    if judge is not None and judge.get("adapter", "claude") != "claude":
        raise ValueError(f"{path}: only the 'claude' judge adapter is supported")
    return config


def default_solver_factory(model_entry: dict) -> Solver:
    """Map a config model entry to a Solver instance."""
    mode = model_entry.get("mode", "api")
    if mode == "api":
        adapter_name = model_entry.get("adapter")
        if not adapter_name:
            raise ValueError(
                f"model entry {model_entry.get('label')!r}: api mode requires 'adapter'"
            )
        adapter = get_adapter(adapter_name, model=model_entry.get("model"))
        return ApiSolver(adapter)
    if mode == "agent":
        try:
            harness = importlib.import_module(HARNESS_MODULE)
        except ImportError as exc:
            raise RuntimeError(
                f"Agent mode requires the harness module {HARNESS_MODULE!r}, "
                f"which is not installed. Expected for harness "
                f"{model_entry.get('harness')!r}. Original error: {exc}"
            ) from exc
        if not hasattr(harness, "get_harness_solver"):
            raise RuntimeError(
                f"{HARNESS_MODULE} does not provide get_harness_solver()"
            )
        return harness.get_harness_solver(model_entry)
    raise ValueError(
        f"model entry {model_entry.get('label')!r}: "
        f"mode must be 'api' or 'agent', got {mode!r}"
    )
