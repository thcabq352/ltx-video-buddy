"""Default Generate uses a local LTX 2.3 GGUF and heretic Gemma.

Catalog literals stay in MODEL_FILES. When those files are absent, base
rewrites the checkpoint / text encoder onto what is actually on disk, and
validation warns instead of failing. No download, no GPU.

Run: python -m pytest tests/test_generate_inventory.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from master_agent.comfy.validator import ValidationReport, _validate_scalar
from master_agent.comfy.workflow_patcher import load_and_patch_workflow

EROS = "10Eros_v1.5-Q4_K_M.gguf"
PROJ = "ltx-2.3-22b-dev-fp8.safetensors"
AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
FP4 = "gemma_3_12B_it_fp4_mixed.safetensors"
SULPHUR = "sulphur_dev-Q3_K_S.gguf"
HERETIC = "gemma-3-12b-it-heretic.safetensors"
DEV_FP8 = "ltx-2.3-22b-dev-fp8.safetensors"


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weight")
    return path


def _resolve_from(root: Path):
    def _resolve(name: str | None):
        if not name:
            return None
        hits = list(root.rglob(Path(str(name)).name))
        return hits[0] if hits else None

    return _resolve


def _patch(variant: str, root: Path):
    with (
        patch(
            "master_agent.comfy.workflow_patcher.resolve_model_path",
            side_effect=_resolve_from(root),
        ),
        patch("master_agent.models.weights.model_search_roots", lambda: [root]),
    ):
        return load_and_patch_workflow(variant, prompt="rain on a window", seed=1, duration_s=1.0)


def _widget(workflow: dict, class_type: str, key: str) -> list[str]:
    found: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        value = (node.get("inputs") or {}).get(key)
        if isinstance(value, str):
            found.append(value)
    return found


def test_base_uses_sulphur_and_heretic_when_literals_are_missing(tmp_path: Path):
    _write(tmp_path / "unet" / SULPHUR)
    _write(tmp_path / "text_encoders" / HERETIC)
    workflow, meta = _patch("base", tmp_path)

    assert meta["checkpoint"] == SULPHUR
    assert _widget(workflow, "CheckpointLoaderSimple", "ckpt_name") == []
    assert _widget(workflow, "LTXAVTextEncoderLoader", "text_encoder") == [HERETIC]
    assert _widget(workflow, "LTXAVTextEncoderLoader", "ckpt_name") == [PROJ]
    assert _widget(workflow, "VAELoader", "vae_name") == [
        AUDIO_VAE,
        "taeltx2_3.safetensors",
    ]
    gguf = _widget(workflow, "UnetLoaderGGUF", "unet_name")
    assert gguf == [SULPHUR]
    blob = " ".join(
        str((node.get("inputs") or {}).get(key) or "")
        for node in workflow.values()
        if isinstance(node, dict)
        for key in ("ckpt_name", "text_encoder", "unet_name")
    )
    assert EROS not in blob
    assert FP4 not in blob


def test_eros_checkpoint_stays_when_the_file_exists(tmp_path: Path):
    _write(tmp_path / "diffusion_models" / EROS)
    _write(tmp_path / "unet" / SULPHUR)
    _write(tmp_path / "text_encoders" / HERETIC)
    workflow, meta = _patch("base", tmp_path)
    assert meta["checkpoint"] == EROS
    assert workflow["1"]["class_type"] == "UnetLoaderGGUF"
    assert workflow["1"]["inputs"]["unet_name"] == EROS
    assert workflow["1"]["inputs"].keys() == {"unet_name"}
    assert _widget(workflow, "LTXAVTextEncoderLoader", "text_encoder") == [HERETIC]
    assert _widget(workflow, "LTXAVTextEncoderLoader", "ckpt_name") == [PROJ]
    assert AUDIO_VAE in _widget(workflow, "VAELoader", "vae_name")


def test_lipsync_checkpoint_is_not_rewritten_to_sulphur(tmp_path: Path):
    _write(tmp_path / "unet" / SULPHUR)
    workflow, meta = _patch("lipsync", tmp_path)
    assert meta["checkpoint"] == DEV_FP8
    assert SULPHUR not in _widget(workflow, "CheckpointLoaderSimple", "ckpt_name")
    assert not any(
        isinstance(node, dict) and node.get("class_type") == "UnetLoaderGGUF"
        for node in workflow.values()
    )


def test_missing_literals_warn_when_a_compatible_file_exists(tmp_path: Path):
    _write(tmp_path / "unet" / SULPHUR)
    _write(tmp_path / "text_encoders" / HERETIC)
    spec = [["some-other-file.safetensors"], {}]
    with patch("master_agent.models.weights.model_search_roots", lambda: [tmp_path]):
        ckpt = ValidationReport(file="mem")
        _validate_scalar(ckpt, "1", "ckpt_name", EROS, spec, inventory=None)
        te = ValidationReport(file="mem")
        _validate_scalar(te, "2", "text_encoder", FP4, spec, inventory=None)
        absent = ValidationReport(file="mem")
        _validate_scalar(
            absent, "9", "ckpt_name", "not-a-model.safetensors", spec, inventory=None
        )
    assert ckpt.ok and ckpt.warnings
    assert "sulphur_dev-Q3_K_S.gguf" in ckpt.warnings[0].message
    assert te.ok and te.warnings
    assert HERETIC in te.warnings[0].message
    assert not absent.ok
