"""Tests for ApiSolver and the solver factory."""

import json
from pathlib import Path

import pytest

from dsbench.adapters import get_adapter
from dsbench.factory import default_solver_factory, load_config
from dsbench.solvers.api import ApiSolver, _build_file_preview
from dsbench.solvers.base import SolveResult
from dsbench.tasks import Task


class SpyAdapter:
    """Adapter that records the prompt and returns fixed text."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return "spy"

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        return self.response


def _make_task(tmp_path: Path, answer_type: str = "code") -> Task:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "values.csv").write_text("10\n20\n30\n40\n50\n60\n")
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir(exist_ok=True)
    (hidden_dir / "answer.yaml").write_text("answers:\n  mean: 35.0\n")

    return Task(
        id="mean_price",
        category="stats",
        difficulty="easy",
        scorer="numeric",
        weight=1.0,
        timeout_s=30,
        answer_type=answer_type,
        prompt="Compute the mean.",
        data_dir=data_dir,
        hidden_dir=hidden_dir,
    )


def test_build_file_preview_first_five_lines(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "values.csv").write_text("10\n20\n30\n40\n50\n60\n")
    preview = _build_file_preview(data_dir)
    assert "--- values.csv ---" in preview
    for value in ["10", "20", "30", "40", "50"]:
        assert value in preview
    assert "60" not in preview


def test_api_solver_code_answer(tmp_path):
    task = _make_task(tmp_path)
    workdir = tmp_path / "work"
    response = (
        "Here is the code.\n"
        "```python\n"
        "import json\n"
        "json.dump({'mean': 35.0}, open('answer.json', 'w'))\n"
        "```\n"
    )
    adapter = SpyAdapter(response)
    solver = ApiSolver(adapter)

    result = solver.solve(task, workdir)

    assert isinstance(result, SolveResult)
    assert result.answer_text == response
    assert result.transcript == response
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.duration_s > 0
    assert json.loads((workdir / "answer.json").read_text()) == {"mean": 35.0}

    # The prompt contains the task prompt and the data preview.
    assert task.prompt in adapter.last_prompt
    assert "--- values.csv ---" in adapter.last_prompt


def test_api_solver_text_answer_runs_no_code(tmp_path):
    task = _make_task(tmp_path, answer_type="text")
    workdir = tmp_path / "work"
    adapter = SpyAdapter("The mean is 35.")
    solver = ApiSolver(adapter)

    result = solver.solve(task, workdir)

    assert result.answer_text == "The mean is 35."
    assert result.returncode == 0
    assert result.duration_s > 0
    assert not (workdir / "answer.json").exists()
    assert result.meta == {}


def test_factory_api_mode(tmp_path):
    from dsbench.solvers.api import ApiSolver as ApiSolverClass

    solver = default_solver_factory({"label": "m", "mode": "api", "adapter": "fake"})
    assert isinstance(solver, ApiSolverClass)


def test_factory_api_mode_requires_adapter():
    with pytest.raises(ValueError, match="api mode requires 'adapter'"):
        default_solver_factory({"label": "m", "mode": "api"})


def test_factory_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="Unknown adapter"):
        default_solver_factory({"label": "m", "mode": "api", "adapter": "nope"})


def test_factory_agent_mode_missing_harness(monkeypatch):
    import dsbench.factory as factory_module

    real_import = factory_module.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "dsbench.solvers.harness":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        factory_module.importlib, "import_module", fake_import
    )

    with pytest.raises(RuntimeError, match="harness module"):
        default_solver_factory(
            {"label": "agent", "mode": "agent", "harness": "claude-code"}
        )


def test_factory_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be 'api' or 'agent'"):
        default_solver_factory({"label": "m", "mode": "telepathy"})


def test_load_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "models:\n  - label: m\n    mode: api\n    adapter: fake\n"
        "runs_per_task: 2\n"
    )
    config = load_config(config_file)
    assert config["models"][0]["label"] == "m"
    assert config["runs_per_task"] == 2


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_requires_models(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("runs_per_task: 2\n")
    with pytest.raises(ValueError, match="'models'"):
        load_config(config_file)


def test_get_adapter_fake():
    adapter = get_adapter("fake")
    assert adapter.generate("hi") == "This is a canned response."
