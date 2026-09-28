"""Patched prompts keep each variant's own weights.

The LTX 2.3 base bundle used to fill any variant that had no MODEL_FILES
entry, then the heuristic wrote the 10Eros bake, the rank-111 LoRA, and
taeltx2_3 onto unrelated graphs. These tests fake every weight as present
so that leak would show up. No GPU.

Run: python -m pytest tests/test_workflow_weights.py -q
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.comfy.catalog import H3_META
from master_agent.comfy.workflow_patcher import (
    load_and_patch_workflow,
    load_workflow_template,
)
from master_agent.models.weights import (
    DEV_FP8_CKPT,
    LTX23_LATENT_UPSCALER,
    STUB_ALIASES,
    WEIGHT_FILES,
    dev_fp8_trailing_bytes,
    ic_ingredients_placement,
    ltx23_latent_upscaler_placement,
    safetensors_declared_size,
    safetensors_trailer,
    scan_bundle,
)
from master_agent.orchestrator.talking import H3_R2V_AUDIO_LABEL
from master_agent.setup import snapshot

BAKE = "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors"
RANK111 = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
TAE = "taeltx2_3.safetensors"
DEV_FP8 = "ltx-2.3-22b-dev-fp8.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
IC_INGREDIENTS = "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"
IC_23 = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
DEBLUR_23 = "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
HERETIC = "gemma4-12b-heretic-ltx25-int8convrot.safetensors"
ENHANCER = "gemma4_e2b_it_bf16.safetensors"
LEAK = (BAKE, RANK111, TAE)

# (variant, substrings that must survive patching)
KEEP = (
    ("vb_ccc_adv", ("flux-2-klein-9b-fp8.safetensors", "Flux2-Klein-9B-consistency-V2.safetensors", "flux2-vae.safetensors", "qwen_3_8b_fp8mixed.safetensors")),
    ("vb_ideogram", ("ideogram4_fp8_scaled.safetensors", "ideogram4_unconditional_fp8_scaled.safetensors", "flux2-vae.safetensors", "Multiple_Realistic_02-Ideogram")),
    ("vb_aivfx_adv", ("Wan2.1_T2V_14B_FusionX_LoRA.safetensors", "wan_2.1_vae.safetensors", "wan-14B_vace_skyreels")),
    ("vb_aivfx_adv_13", ("Wan2.1_T2V_14B_FusionX_LoRA.safetensors", "wan_2.1_vae.safetensors", "vace")),
    ("vb_ccc41_krea2", ("krea2_turbo_fp8_scaled.safetensors", "krea2_identity_edit_v1_2.safetensors", "wan_2.1_vae.safetensors")),
    ("vb_movie_builder", ("flux-2-klein-9B-360-erp-outpaint-lora_V1.safetensors", "Flux2-Klein-9B-consistency-V2.safetensors", DISTILLED_LORA, "ltx-2.3-id-lora-talkvid-3k.safetensors", "flux2-vae.safetensors", "LTX23_video_vae_bf16.safetensors", "LTX23_audio_vae_bf16.safetensors")),
    ("260615_video-buddy_flux_klein_9b_v01", ("flux-2-klein-9b-fp8.safetensors", "Flux2-Klein-9B-consistency-V2.safetensors", "qwen_3_8b_fp8mixed.safetensors", "flux2-vae.safetensors")),
    ("vb_krea2_img", ("krea2", "qwen3vl")),
    ("vb_wan22_vid", ("umt5_xxl", "wan_2.1_vae.safetensors")),
    ("vb_qwen_edit_360", ("qwen",)),
    ("vb_aivfx_startimage", ("qwen",)),
    ("wan_fun_inpaint", ("umt5_xxl", "wan_2.1_vae.safetensors", "fun_inpaint_mask.png")),
    ("wan22", ("umt5_xxl", "wan_2.1_vae.safetensors")),
    ("krea2_img", ("krea2",)),
    ("ltx25_t2v_i2v", ("ltx-2.5",)),
    ("h3_t2v", ("minimax",)),
)

LTX_BASE = ("base", "eros", "directors")
RENDER = (
    "ltx23_lipsync_v08",
    "air_render_030",
    "air_render_050",
    "air_render_businesswoman",
)


def _resolve_present(name: str | None) -> Path | None:
    if not name:
        return None
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return Path("models") / base


def _blob(workflow: dict) -> str:
    import json

    return json.dumps(workflow)


def _load_fallback(variant: str):
    """air_render JSON is gitignored; it is the same node-4826 graph as v08."""
    if variant not in RENDER:
        return None
    try:
        load_workflow_template(variant)
        return None
    except (FileNotFoundError, KeyError):
        return load_workflow_template("ltx23_lipsync_v08")


def _patch(variant: str, *, resolve=_resolve_present):
    fallback = _load_fallback(variant)

    def _load(requested: str):
        if fallback is not None and requested == variant:
            return copy.deepcopy(fallback)
        return load_workflow_template(requested)

    with (
        patch("master_agent.comfy.workflow_patcher.resolve_model_path", side_effect=resolve),
        patch("master_agent.comfy.workflow_patcher.load_workflow_template", side_effect=_load),
    ):
        return load_and_patch_workflow(variant, prompt="weight audit", seed=1, duration_s=1.0)


def _try_patch(variant: str):
    try:
        return _patch(variant)
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"{variant} template is not in this checkout: {exc}")


@pytest.mark.parametrize("variant,needles", KEEP, ids=[row[0] for row in KEEP])
def test_non_ltx_graphs_do_not_inherit_the_base_bundle(variant, needles):
    workflow, meta = _try_patch(variant)
    blob = _blob(workflow)
    for leak in LEAK:
        assert leak not in blob, f"{variant} leaked {leak}"
    for needle in needles:
        assert needle in blob, f"{variant} lost {needle}"
    if variant == "vb_ideogram":
        classes = {
            node.get("class_type")
            for node in workflow.values()
            if isinstance(node, dict)
        }
        assert "LoraLoaderModelOnly" not in classes
    if variant not in ("ltx25_t2v_i2v", "h3_t2v", "vb_aivfx_adv_13", "vb_qwen_edit_360", "vb_aivfx_startimage", "vb_krea2_img", "krea2_img", "wan22", "vb_wan22_vid"):
        assert meta.get("lora") in (None, "")
        assert "10Eros" not in str(meta.get("checkpoint") or "")


@pytest.mark.parametrize("variant", LTX_BASE)
def test_ltx23_base_family_still_uses_its_own_bundle(variant):
    workflow, meta = _patch(variant)
    blob = _blob(workflow)
    assert BAKE in blob
    assert RANK111 in blob
    assert meta["checkpoint"] == BAKE


def test_lipsync_keeps_dev_fp8_and_lipdub():
    workflow, meta = _patch("lipsync")
    blob = _blob(workflow)
    assert DEV_FP8 in blob
    assert "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors" in blob
    assert RANK111 not in blob
    assert BAKE not in blob
    assert meta["checkpoint"] == DEV_FP8


@pytest.mark.parametrize("variant", RENDER)
def test_pr37_render_variants_stay_on_dev_fp8(variant):
    workflow, meta = _patch(variant)
    blob = _blob(workflow)
    assert DEV_FP8 in blob
    assert DISTILLED_LORA in blob
    assert BAKE not in blob
    assert RANK111 not in blob
    assert meta["checkpoint"] == DEV_FP8
    assert workflow["4826"]["inputs"]["ckpt_name"] == DEV_FP8


@pytest.mark.parametrize("variant", ("ltx25_msr", "ltx25_v2v_ic_lora"))
def test_ic_lora_rewrites_to_the_25_ingredients_file(variant):
    assert STUB_ALIASES[IC_23] == IC_INGREDIENTS
    assert STUB_ALIASES[DEBLUR_23] == IC_INGREDIENTS
    present, _meta = _patch(variant)
    blob = _blob(present)
    assert IC_INGREDIENTS in blob
    assert IC_23 not in blob
    assert DEBLUR_23 not in blob

    missing, _meta = _patch(variant, resolve=lambda _name: None)
    blob = _blob(missing)
    assert IC_INGREDIENTS in blob
    assert IC_23 not in blob
    assert DEBLUR_23 not in blob


def test_ltx25_t2a_enhancer_and_encoder_match_the_other_graphs(tmp_path, monkeypatch):
    root = tmp_path / "models"
    te = root / "text_encoders" / HERETIC
    enhancer = root / "text_encoders" / ENHANCER
    te.parent.mkdir(parents=True)
    te.write_bytes(b"te")
    enhancer.write_bytes(b"e2b")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])

    workflow, _meta = load_and_patch_workflow("ltx25_t2a", prompt="room tone", seed=1, duration_s=1.0)
    by_title = {}
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "CLIPLoader":
            continue
        title = str((node.get("_meta") or {}).get("title") or "")
        by_title[title] = node["inputs"]["clip_name"]
    assert by_title["Load CLIP - Text Enhancer"] == ENHANCER
    assert by_title["Load CLIP - Text Encoder"] == HERETIC


def test_ltx25_t2a_enhancer_falls_back_when_e2b_is_absent(tmp_path, monkeypatch):
    root = tmp_path / "models"
    te = root / "text_encoders" / HERETIC
    te.parent.mkdir(parents=True)
    te.write_bytes(b"te")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])
    workflow, _meta = load_and_patch_workflow("ltx25_t2a", prompt="room tone", seed=1, duration_s=1.0)
    blob = _blob(workflow)
    assert ENHANCER not in blob
    assert blob.count(HERETIC) >= 2


def test_doctor_lists_ic_lora_and_latent_upscaler():
    names = {row["name"] for row in snapshot()}
    assert "ltx25-ic-lora" in names
    assert "ltx23-upscaler" in names
    assert "h3-weights" in names
    assert "dev-fp8-trailer" in names


def _safetensors_blob(payload: bytes, junk: bytes = b"") -> bytes:
    import json
    import struct

    header = json.dumps(
        {
            "tensor": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + payload + junk


def test_dev_fp8_trailer_warns_without_rewriting_the_file(tmp_path):
    missing = dev_fp8_trailing_bytes([tmp_path])
    assert missing["ok"] is True
    assert "not on disk" in missing["detail"]

    path = tmp_path / "checkpoints" / DEV_FP8_CKPT
    path.parent.mkdir()
    payload = b"weights"
    junk = b"J" * 64
    blob = _safetensors_blob(payload, junk)
    path.write_bytes(blob)
    before = path.read_bytes()

    declared = len(blob) - len(junk)
    warned = dev_fp8_trailing_bytes([tmp_path])
    assert warned["ok"] is False
    assert f"actual size {len(blob)} bytes" in warned["detail"]
    assert f"header-declared size {declared} bytes" in warned["detail"]
    assert f"{len(junk)} extra bytes" in warned["detail"]
    assert warned["fix"].startswith("Back up the file, then truncate it")
    assert f"({declared} bytes)" in warned["fix"]
    assert "do not truncate" not in warned["detail"]
    assert "do not truncate" not in warned["fix"]
    assert path.read_bytes() == before
    assert safetensors_declared_size(path) == declared

    other = tmp_path / "loras" / "any-weights.safetensors"
    other.parent.mkdir()
    other.write_bytes(blob)
    generic = safetensors_trailer(other)
    assert generic["ok"] is False
    assert generic["actual"] == len(blob)
    assert generic["declared"] == declared
    assert generic["extra"] == len(junk)
    assert "any-weights.safetensors" in generic["detail"]
    assert other.read_bytes() == blob

    exact = _safetensors_blob(payload)
    path.write_bytes(exact)
    clean = dev_fp8_trailing_bytes([tmp_path])
    assert clean["ok"] is True
    assert "matches its safetensors header" in clean["detail"]
    assert path.read_bytes() == exact

    path.write_bytes(b"not-a-safetensors-file")
    unreadable = dev_fp8_trailing_bytes([tmp_path])
    assert unreadable["ok"] is True
    assert path.read_bytes() == b"not-a-safetensors-file"


def test_ic_ingredients_placement_messages(tmp_path, monkeypatch):
    name = WEIGHT_FILES["ic_lora"].filename
    missing = ic_ingredients_placement([tmp_path])
    assert missing["ok"] is False
    assert "models/loras/" in missing["detail"]
    assert "download-models --ltx25" in missing["detail"]

    wrong = tmp_path / "checkpoints" / name
    wrong.parent.mkdir()
    wrong.write_bytes(b"x" * 16)
    moved = ic_ingredients_placement([tmp_path])
    assert moved["ok"] is False
    assert "mv " in moved["detail"]
    assert "models/loras/" in moved["fix"]
    assert "will not move" in moved["detail"]

    role = tmp_path / "loras" / name
    role.parent.mkdir()
    role.write_bytes(b"short")
    monkeypatch.setattr("master_agent.models.weights._IC_INGREDIENTS_MIN_BYTES", 100)
    short = ic_ingredients_placement([tmp_path / "also-missing", tmp_path])
    # The checkpoints copy is still there; a short file in loras/ is the one we judge.
    assert short["ok"] is False
    assert "1.31" in short["detail"] or "loras" in short["detail"]

    monkeypatch.setattr("master_agent.models.weights._IC_INGREDIENTS_MIN_BYTES", 4)
    role.write_bytes(b"x" * 8)
    ok = ic_ingredients_placement([tmp_path])
    assert ok["ok"] is True
    assert name in ok["detail"]


def test_pixel_ic_lora_does_not_satisfy_the_ingredients_check(tmp_path, monkeypatch):
    pixel = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    dest = tmp_path / "loras" / pixel
    dest.parent.mkdir()
    dest.write_bytes(b"pixel-lora")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [tmp_path])
    status = scan_bundle("ltx25_iclora", roots=[tmp_path])
    assert status.found_paths.get("ic_lora", "").endswith(pixel)
    place = ic_ingredients_placement([tmp_path])
    assert place["ok"] is False
    assert IC_INGREDIENTS in place["detail"]


def test_latent_upscaler_placement_is_a_move_not_a_download(tmp_path):
    missing = ltx23_latent_upscaler_placement([tmp_path])
    assert missing["ok"] is False
    assert "latent_upscale_models" in missing["detail"]

    src = tmp_path / "upscale_models" / LTX23_LATENT_UPSCALER
    src.parent.mkdir()
    src.write_bytes(b"upscaler")
    wrong = ltx23_latent_upscaler_placement([tmp_path])
    assert wrong["ok"] is False
    assert "mv " in wrong["detail"]
    assert "latent_upscale_models" in wrong["fix"]
    assert "will not move" in wrong["detail"]
    assert "vb_movie_builder" in wrong["detail"]

    role = tmp_path / "latent_upscale_models" / LTX23_LATENT_UPSCALER
    role.parent.mkdir()
    role.write_bytes(b"upscaler")
    ok = ltx23_latent_upscaler_placement([tmp_path])
    assert ok["ok"] is True
    assert "Restart Comfy" in ok["detail"]


def test_h3_catalog_marks_us_unavailable_without_hiding_the_voice_label():
    for meta in H3_META.values():
        assert "Unavailable in the US" in meta["description"]
    assert H3_R2V_AUDIO_LABEL in H3_META["h3_r2v"]["description"]
