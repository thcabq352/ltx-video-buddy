"""One resolver for Buddy's managed ComfyUI folder (``comfy/root.py``).

No live Comfy, no GPU, no subprocess.

Run: PYTHONPATH=video_buddy python -m pytest tests/test_comfy_root.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.comfy import root as root_mod
from master_agent.comfy import tower
from master_agent.comfy.root import (
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_FLAG,
    SOURCE_STATE,
    managed_comfy_root,
    resolve_managed_comfy_root,
)


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    project = tmp_path / "video_buddy"
    project.mkdir()
    state = project / "state"
    state.mkdir()
    models = project / "models"
    models.mkdir()
    monkeypatch.setattr("master_agent.config.PROJECT_ROOT", project)
    monkeypatch.setattr("master_agent.config.MODELS_DIR", models)
    monkeypatch.setattr("master_agent.config.STATE_DIR", state)
    monkeypatch.setattr(tower, "STATE_DIR", state)
    monkeypatch.setattr(tower, "PROJECT_ROOT", project)
    monkeypatch.delenv("MANAGED_COMFY_ROOT", raising=False)
    monkeypatch.delenv("COMFY_MODE", raising=False)
    return {
        "project": project,
        "state": state,
        "models": models,
        "default": (project / "ComfyUI").resolve(),
        "env": (tmp_path / "E-drive" / "VideoBuddy" / "ComfyUI").resolve(),
        "saved": (tmp_path / "saved-comfy").resolve(),
        "flag": (tmp_path / "flag-comfy").resolve(),
    }


def _save_workspace(state: Path, workspace: Path) -> None:
    (state / "managed_comfy.json").write_text(json.dumps({"workspace": str(workspace)}) + "\n", encoding="utf-8")


def test_default_is_project_comfyui(paths):
    got = resolve_managed_comfy_root()
    assert got.path == paths["default"]
    assert got.source == SOURCE_DEFAULT


def test_saved_state_beats_default(paths):
    _save_workspace(paths["state"], paths["saved"])
    got = resolve_managed_comfy_root()
    assert (got.path, got.source) == (paths["saved"], SOURCE_STATE)


def test_env_beats_saved_state(paths, monkeypatch):
    # The field report: a first `comfy start` saved the default, then
    # MANAGED_COMFY_ROOT was set and everything still used the default.
    _save_workspace(paths["state"], paths["default"])
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    got = resolve_managed_comfy_root()
    assert (got.path, got.source) == (paths["env"], SOURCE_ENV)


def test_explicit_beats_env(paths, monkeypatch):
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    got = resolve_managed_comfy_root(paths["flag"])
    assert (got.path, got.source) == (paths["flag"], SOURCE_FLAG)
    assert managed_comfy_root(str(paths["flag"])) == paths["flag"]


def test_env_value_is_trimmed_and_unquoted(paths, monkeypatch):
    monkeypatch.setenv("MANAGED_COMFY_ROOT", f'  "{paths["env"]}"  ')
    assert managed_comfy_root() == paths["env"]


def test_blank_env_and_bad_state_fall_back_to_default(paths, monkeypatch):
    monkeypatch.setenv("MANAGED_COMFY_ROOT", "   ")
    (paths["state"] / "managed_comfy.json").write_text("{not json", encoding="utf-8")
    assert resolve_managed_comfy_root().source == SOURCE_DEFAULT


def test_every_consumer_honors_env(paths, monkeypatch):
    from master_agent import automatic_install as ai
    from master_agent import setup as setup_mod
    from master_agent.comfy import model_paths, updates
    from master_agent.models.weights import model_search_roots

    _save_workspace(paths["state"], paths["default"])
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    want = paths["env"]

    assert ai.managed_workspace() == want
    assert model_paths.managed_workspace() == want
    assert updates._workspace() == want
    assert tower.ManagedComfyTower().workspace == want
    assert setup_mod._managed_comfy_state()[1] == want
    assert want / "models" in model_search_roots()
    assert want in model_paths.read_only_weight_roots()

    check = ai.check_workspace(want, state="absent", mode="managed")
    assert f"{want} (from MANAGED_COMFY_ROOT)" in check.lines[0]


def test_doctor_comfy_row_names_the_resolved_folder(paths, monkeypatch):
    from master_agent import setup as setup_mod

    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    monkeypatch.setenv("COMFYUI_URL", "http://127.0.0.1:9")
    row = setup_mod.check_comfy()
    assert row["ok"] is False
    assert str(paths["env"]) in row["detail"]
    assert "from MANAGED_COMFY_ROOT" in row["detail"]
    assert str(paths["default"]) not in row["detail"]


def test_tower_only_remembers_an_explicit_workspace(paths, monkeypatch):
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    tower.ManagedComfyTower()._sync_state(last_error="")
    saved = json.loads((paths["state"] / "managed_comfy.json").read_text(encoding="utf-8"))
    assert saved["workspace"] == ""

    monkeypatch.delenv("MANAGED_COMFY_ROOT")
    tower.ManagedComfyTower(workspace=paths["flag"])._sync_state(last_error="")
    assert resolve_managed_comfy_root().path == paths["flag"]
    assert resolve_managed_comfy_root().source == SOURCE_STATE
    # A later env var still wins over the remembered flag.
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(paths["env"]))
    assert tower.ManagedComfyTower().workspace == paths["env"]


def test_resolver_is_the_only_reader_of_the_env_var():
    pkg = Path(root_mod.__file__).resolve().parents[1]
    readers = sorted(
        str(path.relative_to(pkg))
        for path in pkg.rglob("*.py")
        if 'getenv("MANAGED_COMFY_ROOT")' in path.read_text(encoding="utf-8")
        or "environ[\"MANAGED_COMFY_ROOT\"]" in path.read_text(encoding="utf-8")
    )
    assert readers == []
    assert "os.getenv(ENV_VAR)" in Path(root_mod.__file__).read_text(encoding="utf-8")
