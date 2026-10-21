"""GGUF wins whenever a compatible file is on disk.

Not only when FORCE_LOADER=gguf or VRAM is under 14. fp8, bf16, and the
EROS all-in-one stay available when that slot has no GGUF. No GPU.

Run: python -m pytest tests/test_gguf_first.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import MODEL_FILES
from master_agent.models.vram_policy import (
    LTX23_DEV_GGUF,
    LTX23_DISTILLED_GGUF,
    LTX23_PREFERENCE,
    preference_order,
    workflow_row,
)
from master_agent.models.weights import (
    WEIGHT_FILES,
    resolve_ltx23_gguf,
    resolve_weight,
    transformer_preference_order,
)

Q4_DISTILLED = "LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"
Q4_DEV = "LTX-2.3-dev-Q4_K_S.gguf"
SULPHUR = "sulphur_dev-Q3_K_S.gguf"
EROS = "10Eros_v1.5-Q4_K_M.gguf"
OLD_BAKE = "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors"
PROJ = "ltx-2.3-22b-dev-fp8.safetensors"
AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
VIDEO_VAE = "taeltx2_3.safetensors"
GEMMA = "gemma_3_12B_it_fp4_mixed.safetensors"
LTX25_GGUF = "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"
LTX25_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"


def _write(path: Path, blob: bytes = b"weight") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def test_catalog_names_gguf_before_eros_fallback():
    assert MODEL_FILES["base"]["diffusion"] == Q4_DISTILLED
    assert MODEL_FILES["eros"]["diffusion"] == Q4_DISTILLED
    assert MODEL_FILES["directors"]["diffusion"] == Q4_DEV
    assert MODEL_FILES["base"]["checkpoint"] == EROS
    assert LTX23_PREFERENCE[0] == Q4_DISTILLED
    assert LTX23_PREFERENCE[-1] == EROS
    assert workflow_row("base").default_pack == LTX23_DISTILLED_GGUF[0]
    assert workflow_row("directors").default_pack == LTX23_DEV_GGUF[0]
    assert workflow_row("base").default_pack.endswith(".gguf")


def test_16gb_without_force_still_ranks_gguf_first():
    order = transformer_preference_order(vram_gb=16, force_loader="")
    assert order[0].endswith(".gguf")
    assert any(name.endswith(LTX25_BF16) for name in order)
    names = (
        "model-bf16.safetensors",
        "model_fp8_scaled.safetensors",
        "model-Q4_K_S.gguf",
    )
    ranked = preference_order(names, vram_gb=16, force_loader="")
    assert ranked[0].endswith(".gguf")
    assert "model-bf16.safetensors" in ranked


def test_ltx25_on_disk_gguf_beats_bf16_at_16gb(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("master_agent.config.VRAM_GB", 16.0)
    monkeypatch.setattr("master_agent.config.FORCE_LOADER", "")
    gguf = _write(tmp_path / "diffusion_models" / LTX25_GGUF)
    _write(tmp_path / "diffusion_models" / LTX25_BF16)
    found = resolve_weight(WEIGHT_FILES["transformer"], [tmp_path])
    assert found == gguf


def test_ltx25_bf16_still_loads_when_no_gguf(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("master_agent.config.VRAM_GB", 16.0)
    monkeypatch.setattr("master_agent.config.FORCE_LOADER", "")
    bf16 = _write(tmp_path / "diffusion_models" / LTX25_BF16)
    found = resolve_weight(WEIGHT_FILES["transformer"], [tmp_path])
    assert found == bf16


def test_wan_gguf_beats_fp8_without_force(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("master_agent.config.VRAM_GB", 16.0)
    monkeypatch.setattr("master_agent.config.FORCE_LOADER", "")
    gguf = _write(tmp_path / "diffusion_models" / "Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf")
    _write(tmp_path / "diffusion_models" / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
    found = resolve_weight(WEIGHT_FILES["wan22_high"], [tmp_path])
    assert found == gguf


def test_quantstack_q4_beats_sulphur_and_eros(tmp_path: Path):
    q4 = _write(tmp_path / "unet" / Q4_DISTILLED)
    _write(tmp_path / "unet" / SULPHUR)
    _write(tmp_path / "diffusion_models" / EROS)
    assert resolve_ltx23_gguf("base", [tmp_path]) == q4
    assert resolve_ltx23_gguf("eros", [tmp_path]) == q4


def test_directors_prefers_dev_q4_over_sulphur(tmp_path: Path):
    dev = _write(tmp_path / "unet" / Q4_DEV)
    _write(tmp_path / "unet" / SULPHUR)
    _write(tmp_path / "unet" / Q4_DISTILLED)
    assert resolve_ltx23_gguf("directors", [tmp_path]) == dev


def test_sulphur_used_when_it_is_the_only_gguf(tmp_path: Path):
    sulphur = _write(tmp_path / "unet" / SULPHUR)
    assert resolve_ltx23_gguf("base", [tmp_path]) == sulphur
    assert resolve_ltx23_gguf("directors", [tmp_path]) == sulphur


def test_no_gguf_returns_none_so_eros_can_stay(tmp_path: Path):
    _write(tmp_path / "checkpoints" / OLD_BAKE)
    assert resolve_ltx23_gguf("base", [tmp_path]) is None


def _patch(variant: str, roots: list[Path]):
    def _resolve(name: str | None):
        if not name:
            return None
        return Path("models") / str(name).replace("\\", "/").rsplit("/", 1)[-1]

    with (
        patch("master_agent.comfy.workflow_patcher.resolve_model_path", side_effect=_resolve),
        patch("master_agent.models.weights.model_search_roots", lambda: roots),
    ):
        return load_and_patch_workflow(variant, prompt="gguf first", seed=1, duration_s=1.0)


@pytest.mark.parametrize(
    ("variant", "gguf_name"),
    (
        ("base", Q4_DISTILLED),
        ("eros", Q4_DISTILLED),
        ("directors", Q4_DEV),
    ),
)
def test_ltx23_graph_uses_gguf_for_model_and_keeps_eros_for_vae(variant: str, gguf_name: str, tmp_path: Path):
    _write(tmp_path / "unet" / Q4_DISTILLED)
    _write(tmp_path / "unet" / Q4_DEV)
    _write(tmp_path / "diffusion_models" / EROS)
    workflow, meta = _patch(variant, [tmp_path])
    gguf_nodes = [
        node
        for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type") == "UnetLoaderGGUF"
    ]
    assert len(gguf_nodes) == 1
    assert gguf_nodes[0]["class_type"] == "UnetLoaderGGUF"
    assert gguf_nodes[0]["inputs"] == {"unet_name": EROS}
    assert gguf_nodes[0]["inputs"]["unet_name"] != gguf_name
    assert "dequant_dtype" not in gguf_nodes[0]["inputs"]
    assert workflow["4"]["inputs"]["model"] == ["1", 0]
    assert workflow["1"]["class_type"] == "UnetLoaderGGUF"
    assert workflow["2"]["inputs"]["text_encoder"] == GEMMA
    assert workflow["2"]["inputs"]["ckpt_name"] == PROJ
    assert workflow["3"]["class_type"] == "VAELoader"
    assert workflow["3"]["inputs"]["vae_name"] == AUDIO_VAE
    assert workflow["5"]["inputs"]["vae_name"] == VIDEO_VAE
    assert workflow["70"]["class_type"] == "VAEDecode"
    assert workflow["70"]["inputs"] == {"samples": ["60", 0], "vae": ["5", 0]}
    assert workflow["71"]["inputs"]["audio_vae"] == ["3", 0]
    assert OLD_BAKE not in str(workflow)
    assert meta["gguf_unet"] == EROS
    assert meta["checkpoint"] == EROS
    assert not any(
        isinstance(node, dict) and "fp8" in str(node.get("class_type") or "").lower()
        for node in workflow.values()
    )


def test_lipsync_is_not_rewired_to_gguf(tmp_path: Path):
    _write(tmp_path / "unet" / SULPHUR)
    workflow, meta = _patch("lipsync", [tmp_path])
    assert meta["gguf_unet"] is None
    assert meta["checkpoint"] == "ltx-2.3-22b-dev-fp8.safetensors"
    assert not any(
        isinstance(node, dict)
        and node.get("class_type") == "UnetLoaderGGUF"
        and "sulphur" in str((node.get("inputs") or {}).get("unet_name") or "").lower()
        for node in workflow.values()
    )
