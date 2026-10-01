"""Buddy-owned extra_model_paths.yaml and pack-download destination guard.

No live Comfy, no GPU, no Hub, no weight download.

Run: PYTHONPATH=video_buddy python -m pytest tests/test_model_paths.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from master_agent.comfy.attach import ATTACH_SCHEMA
from master_agent.comfy.model_paths import (
    ModelPathsError,
    buddy_yaml_path,
    render_extra_model_paths_yaml,
    write_external_yaml_copy,
    write_extra_model_paths,
)
from master_agent.comfy.tower import ManagedComfyTower, TowerError, cmd_tower
from master_agent.models.download import DownloadDestinationError, assert_pack_download_destination
from master_agent.models.inventory import scan_inventory
from master_agent.models.weights import WEIGHT_FILES, download_files, model_search_roots


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    models = tmp_path / "buddy-models"
    models.mkdir()
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    portable = tmp_path / "portable"
    managed = tmp_path / "managed-comfy"
    external = tmp_path / "external-comfy"
    (external / "models" / "checkpoints").mkdir(parents=True)
    sentinel = external / "models" / "checkpoints" / "keep.safetensors"
    sentinel.write_bytes(b"user-weight")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr("master_agent.config.MODELS_DIR", models)
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", comfy)
    monkeypatch.setattr("master_agent.config.PORTABLE_ROOT", portable)
    monkeypatch.setattr("master_agent.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("master_agent.config.STATE_DIR", state)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(managed))
    monkeypatch.setenv("EXTERNAL_COMFY_ROOT", str(external))
    monkeypatch.delenv("EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.delenv("LTX_MODELS_DIRS", raising=False)
    monkeypatch.delenv("COMFY_MODE", raising=False)
    return {
        "models": models,
        "comfy": comfy,
        "external": external,
        "sentinel": sentinel,
        "state": state,
        "managed": managed,
    }


def test_attach_module_stays_workflow_patch_plan():
    import master_agent.comfy.attach as attach
    import master_agent.comfy.model_paths as model_paths

    assert ATTACH_SCHEMA == "buddy.comfy.attach/v1"
    assert "WorkflowPatchPlan" in (attach.__doc__ or "")
    assert not hasattr(model_paths, "WorkflowPatchPlan")


def test_yaml_is_buddy_owned_and_user_sections_are_read_only(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    path = write_extra_model_paths()
    assert path == buddy_yaml_path()
    assert path.parent == roots["state"]
    text = path.read_text(encoding="utf-8")
    assert "WorkflowPatchPlan" in text
    data = yaml.safe_load(text)
    buddy = data["buddy_models"]
    assert buddy["is_default"] is True
    assert buddy["base_path"] == roots["models"].resolve().as_posix()
    assert "diffusion_models" in buddy["diffusion_models"]
    assert "unet" in buddy["diffusion_models"]
    user_sections = {key: value for key, value in data.items() if key.startswith("user_models_")}
    assert user_sections
    for section in user_sections.values():
        assert "is_default" not in section
    pointed = {section["base_path"] for section in user_sections.values()}
    assert roots["external"].joinpath("models").resolve().as_posix() in pointed
    assert roots["comfy"].joinpath("models").resolve().as_posix() in pointed
    assert not (roots["external"] / "extra_model_paths.yaml").exists()
    assert roots["sentinel"].read_bytes() == b"user-weight"
    assert not (roots["managed"] / "extra_model_paths.yaml").exists()
    assert not (roots["comfy"] / "extra_model_paths.yaml").exists()


def test_refuses_yaml_write_into_attached_tree_without_flag(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    dest = roots["external"] / "extra_model_paths.yaml"
    with pytest.raises(ModelPathsError, match="write-yaml-into-external"):
        write_extra_model_paths(dest)
    assert not dest.exists()
    assert roots["sentinel"].read_bytes() == b"user-weight"


def test_consent_flag_copies_yaml_and_leaves_weights_alone(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    buddy = write_extra_model_paths()
    copied = write_external_yaml_copy(buddy)
    assert copied == roots["external"] / "extra_model_paths.yaml"
    assert copied.read_text(encoding="utf-8") == buddy.read_text(encoding="utf-8")
    assert roots["sentinel"].read_bytes() == b"user-weight"
    listed = sorted(p.name for p in roots["external"].iterdir())
    assert listed == ["extra_model_paths.yaml", "models"]


def test_consent_flag_does_not_create_a_missing_install(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    missing = tmp_path / "missing-install"
    monkeypatch.setenv("EXTERNAL_COMFY_ROOT", str(missing))
    with pytest.raises(ModelPathsError, match="does not exist"):
        write_external_yaml_copy()
    assert not missing.exists()


def test_render_rejects_is_default_on_a_user_section(tmp_path: Path):
    from master_agent.comfy.model_paths import ModelPathSource

    sources = [
        ModelPathSource("buddy_models", tmp_path / "models", "buddy", True),
        ModelPathSource("user_models_1", tmp_path / "user", "user", True),
    ]
    with pytest.raises(ModelPathsError, match="is_default"):
        render_extra_model_paths_yaml(sources)


def test_download_destination_guard(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    allowed = roots["models"] / "checkpoints" / "new.safetensors"
    assert assert_pack_download_destination(allowed) == allowed.resolve()
    outside = tmp_path / "elsewhere" / "new.safetensors"
    with pytest.raises(DownloadDestinationError, match="MODELS_DIR"):
        assert_pack_download_destination(outside)
    comfy_dest = roots["comfy"] / "models" / "checkpoints" / "new.safetensors"
    with pytest.raises(DownloadDestinationError, match="MODELS_DIR"):
        assert_pack_download_destination(comfy_dest)
    # MODELS_DIR accidentally contains the attached tree: still refuse that write.
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path)
    attached = roots["external"] / "models" / "checkpoints" / "new.safetensors"
    with pytest.raises(DownloadDestinationError, match="read-only"):
        assert_pack_download_destination(attached)
    assert roots["sentinel"].read_bytes() == b"user-weight"


def test_download_files_refuses_attached_dest_root(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("hub download must not start")

    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.download.download_hub_file", boom)
    with pytest.raises(DownloadDestinationError, match="MODELS_DIR"):
        download_files(
            [WEIGHT_FILES["audio_vae"]],
            dest_root=roots["external"] / "models",
            yes=True,
        )
    assert roots["sentinel"].read_bytes() == b"user-weight"
    assert not list(roots["external"].rglob("*.safetensors")) or roots["sentinel"].is_file()


def test_download_hub_file_writes_under_models_dir_only(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    blob = tmp_path / "blob.safetensors"
    blob.write_bytes(b"not-a-real-weight")

    def fake_download(*, repo_id: str, filename: str) -> str:
        assert repo_id == "unit/test"
        return str(blob)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    dest = roots["models"] / "checkpoints" / "unit-test.safetensors"
    from master_agent.models.download import download_hub_file

    out = download_hub_file(
        repo_id="unit/test",
        repo_filename="unit-test.safetensors",
        dest=dest,
        progress=lambda _m: None,
    )
    assert out == dest.resolve()
    assert dest.read_bytes() == b"not-a-real-weight"
    blocked = roots["comfy"] / "models" / "checkpoints" / "must-not-land.safetensors"
    with pytest.raises(DownloadDestinationError):
        download_hub_file(
            repo_id="unit/test",
            repo_filename="must-not-land.safetensors",
            dest=blocked,
            progress=lambda _m: None,
        )
    assert not blocked.exists()
    assert dest.read_bytes() == b"not-a-real-weight"


def test_inventory_order_is_models_then_comfy_then_yaml_then_hf(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    name = "order-probe.safetensors"
    project = roots["models"] / "checkpoints" / name
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_bytes(b"project")
    comfy_copy = roots["comfy"] / "models" / "checkpoints" / name
    comfy_copy.write_bytes(b"comfy")
    yaml_models = tmp_path / "from-yaml"
    only = yaml_models / "loras" / "only-yaml.safetensors"
    only.parent.mkdir(parents=True)
    only.write_bytes(b"yaml-only")
    (yaml_models / "checkpoints").mkdir()
    (yaml_models / "checkpoints" / name).write_bytes(b"yaml")
    (roots["state"] / "extra_model_paths.yaml").write_text(
        "from_yaml:\n"
        f"  base_path: {yaml_models.resolve().as_posix()}\n"
        "  loras: loras\n"
        "  checkpoints: checkpoints\n",
        encoding="utf-8",
    )
    snap = tmp_path / "hf" / "hub" / "models--Example--order" / "snapshots"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    searched = model_search_roots()

    def _index(path: Path) -> int:
        resolved = path.resolve()
        for index, root in enumerate(searched):
            if root == resolved:
                return index
        raise AssertionError(f"{resolved} not in search roots")

    assert _index(roots["models"]) < _index(roots["comfy"] / "models") < _index(yaml_models)
    assert _index(yaml_models) < _index(snap)

    inv = scan_inventory(roots["models"], roots["comfy"], write=False)
    winner = inv.by_name()[name]
    assert winner.root == "project"
    assert Path(winner.abs_path).read_bytes() == b"project"
    found = inv.by_name()["only-yaml.safetensors"]
    assert found.root == "search"
    labels = [entry.root for entry in inv.entries if entry.name == name]
    assert labels[0] == "project"
    assert "comfyui" in labels


def test_external_mode_docs_cover_the_contract():
    doc = Path(__file__).resolve().parents[1].joinpath("docs", "MANAGED_COMFY.md")
    text = doc.read_text(encoding="utf-8")
    for phrase in (
        "COMFY_MODE=external",
        "refuses start, stop, and restart",
        "never deletes",
        "never writes",
        "COMFYUI_URL",
        "EXTERNAL_COMFY_ROOT",
        "--extra-model-paths-config",
        "MODELS_DIR",
        "--write-yaml-into-external",
        "WorkflowPatchPlan",
        "--where local",
        "--disable-auto-launch",
        "comfy attach",
    ):
        assert phrase in text, phrase


def test_external_status_can_consent_yaml_copy_without_launch(tmp_path: Path, monkeypatch):
    roots = _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("master_agent.comfy.tower.STATE_DIR", roots["state"])
    monkeypatch.setattr("master_agent.comfy.tower.http_ready", lambda *_a, **_k: (False, "down"))
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        raise AssertionError("status must not invoke comfy-cli")

    monkeypatch.setattr("master_agent.comfy.tower.subprocess.run", fake_run)
    monkeypatch.setenv("COMFY_MODE", "external")
    rc = cmd_tower(
        Namespace(
            comfy_command="status",
            workspace=str(roots["managed"]),
            port=None,
            base_url="http://127.0.0.1:8188",
            no_wait=False,
            no_watch=True,
            extra_model_paths=None,
            write_yaml_into_external=True,
            as_json=False,
        )
    )
    assert rc == 1
    assert calls == []
    copied = roots["external"] / "extra_model_paths.yaml"
    assert copied.is_file()
    assert "is_default: true" in copied.read_text(encoding="utf-8")
    assert roots["sentinel"].read_bytes() == b"user-weight"
    with pytest.raises(TowerError, match="comfy_mode=external"):
        ManagedComfyTower(workspace=roots["managed"]).stop()
    assert calls == []
