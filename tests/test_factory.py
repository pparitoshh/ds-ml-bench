"""Tests for the solver factory wiring."""

import pytest

from dsbench.factory import default_solver_factory
from dsbench.solvers.harness import ClaudeCodeSolver, HarnessSolver, OpenCodeSolver


def test_agent_mode_maps_to_opencode() -> None:
    solver = default_solver_factory(
        {
            "label": "glm-5.3",
            "mode": "agent",
            "harness": "opencode",
            "model": "opencode-go/glm-5.3",
        }
    )
    assert isinstance(solver, OpenCodeSolver)
    assert isinstance(solver, HarnessSolver)
    assert solver.model == "opencode-go/glm-5.3"


def test_agent_mode_maps_to_claude_code() -> None:
    solver = default_solver_factory(
        {
            "label": "claude-code",
            "mode": "agent",
            "harness": "claude-code",
            "model": "claude-opus-5-5",
        }
    )
    assert isinstance(solver, ClaudeCodeSolver)


def test_agent_mode_default_harness_is_opencode() -> None:
    solver = default_solver_factory(
        {"label": "kimi", "mode": "agent", "model": "moonshotai/kimi-k3.1-pro"}
    )
    assert isinstance(solver, OpenCodeSolver)


def test_agent_mode_unknown_harness_rejected() -> None:
    with pytest.raises(ValueError):
        default_solver_factory(
            {"label": "x", "mode": "agent", "harness": "nope", "model": "m"}
        )


def test_api_mode_still_maps_to_adapter() -> None:
    from dsbench.adapters import get_adapter
    from dsbench.adapters.fake import FakeAdapter
    from dsbench.solvers.api import ApiSolver

    solver = default_solver_factory(
        {"label": "f", "mode": "api", "adapter": "fake", "model": "m"}
    )
    assert isinstance(solver, ApiSolver)
    assert isinstance(solver._adapter, type(get_adapter("fake", model="m")))
