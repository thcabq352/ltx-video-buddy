"""flux_ltx_i2v: FLUX.2 [klein] 4B still → LTX 2.5 i2v 25-frame motion pass, one shared prompt."""

from __future__ import annotations

import json

from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR
from master_agent.models.vram_policy import family_for_slug, workflow_row
from master_agent.models.weights import (
    TRANSFORMER_PREFERENCE,
    bundle_for_variant,
    is_ltx25_bundle,
)
from master_agent.orchestrator.director import choose_variant, rule_based_variant

VARIANT = "flux_ltx_i2v"
HERETIC_TE = "gemma4-12b-heretic-ltx25-int8convrot.safetensors"
FLUX_FP8 = "flux-2-klein-4b-fp8.safetensors"
FLUX_TEXT = "qwen_3_4b_fp4_flux2.safetensors"
FLUX_VAE = "flux2-vae.safetensors"


def _raw() -> dict:
    return json.loads((WORKFLOWS_DIR / WORKFLOW_FILES[VARIANT]).read_text(encoding="utf-8"))


def _by_class(wf: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(nid, n) for nid, n in wf.items() if isinstance(n, dict) and n.get("class_type") == class_type]


def test_router_picks_flux_ltx_before_startimage_and_flux():
    assert rule_based_variant("flux to ltx: neon alley, slow push-in") == VARIANT
    assert rule_based_variant("flux->ltx portrait with rain audio") == VARIANT
    assert rule_based_variant("make a flux still to video shot") == VARIANT
    assert rule_based_variant("flux start frame then animate") == VARIANT
    assert rule_based_variant("use flux_ltx_i2v") == VARIANT
    assert rule_based_variant("generate an aivfx start-image") == "vb_aivfx_startimage"
    assert rule_based_variant("flux character sheet") == "flux"
    assert rule_based_variant("ltx 2.5 alley push-in") == "ltx25_t2v_i2v"
    variant, _source = choose_variant("flux to ltx rainy rooftop")
    assert variant == VARIANT


def test_template_hands_flux_still_to_ltx_image_input():
    wf = _raw()
    assert not _by_class(wf, "LoadImage")
    decode_id = "9108"
    assert wf[decode_id]["class_type"] == "VAEDecode"
    assert wf[decode_id]["inputs"]["vae"] == ["9103", 0]
    assert wf["5014:4990"]["inputs"]["input"] == [decode_id, 0]
    (_sid, save), = _by_class(wf, "SaveImage")
    assert save["inputs"]["images"] == [decode_id, 0]
    assert save["inputs"]["filename_prefix"] == "flux_ltx_i2v_still"
    assert wf["5014:5506"]["inputs"]["value"] is True
    assert wf["9101"]["class_type"] == "UNETLoader"
    assert wf["9101"]["inputs"]["unet_name"] == FLUX_FP8
    assert wf["9101"]["inputs"]["weight_dtype"] == "default"
    assert wf["9102"]["class_type"] == "CLIPLoader"
    assert wf["9102"]["inputs"]["clip_name"] == FLUX_TEXT
    assert wf["9102"]["inputs"]["type"] == "flux2"
    assert wf["9103"]["inputs"]["vae_name"] == FLUX_VAE
    assert wf["9106"]["class_type"] == "EmptyFlux2LatentImage"
    assert _by_class(wf, "LTXVEmptyLatentAudio")
    assert _by_class(wf, "SaveVideo")


def test_one_aesthetic_prompt_drives_both_models():
    wf = _raw()
    assert wf["9104"]["inputs"]["text"] == ["5508", 0]
    concat = wf["9002"]
    assert concat["class_type"] == "StringConcatenate"
    assert concat["inputs"]["string_a"] == ["5508", 0]
    assert concat["inputs"]["string_b"] == ["9001", 0]
    assert wf["5014:5556"]["inputs"]["on_false"] == ["9002", 0]
    seed_links = [
        (nid, key) for nid, n in wf.items() for key, v in (n.get("inputs") or {}).items() if v == ["9010", 0]
    ]
    assert ("9107", "seed") in seed_links
    assert ("5516:4832", "noise_seed") in seed_links
    assert wf["9106"]["inputs"]["width"] == ["9011", 0]
    assert wf["5514:3059"]["inputs"]["width"] == ["9011", 0]
    assert wf["9106"]["inputs"]["height"] == ["9012", 0]
    assert wf["5514:3059"]["inputs"]["height"] == ["9012", 0]


def test_patch_defaults_to_25_frames_and_keeps_flux_and_motion_widgets():
    wf, meta = load_and_patch_workflow(VARIANT, prompt="neon rain portrait, 35mm", seed=7)
    assert meta["frames"] == 25
    assert meta["filename_prefix"] == VARIANT
    assert wf["5508"]["inputs"]["value"] == "neon rain portrait, 35mm"
    assert wf["9001"]["inputs"]["value"] != "neon rain portrait, 35mm"
    assert wf["9010"]["inputs"]["value"] == 7
    assert wf["9107"]["inputs"]["steps"] == 20
    assert wf["9109"]["inputs"]["filename_prefix"] == "flux_ltx_i2v_still"
    assert wf["9101"]["class_type"] == "UNETLoader"
    assert wf["9101"]["inputs"]["unet_name"] == FLUX_FP8
    (_vid, video), = _by_class(wf, "SaveVideo")
    assert video["inputs"]["filename_prefix"] == VARIANT
    (_aid, audio), = _by_class(wf, "LTXVEmptyLatentAudio")
    assert audio["inputs"]["frames_number"] == 25
    ltx_unets = [
        n["inputs"]["unet_name"]
        for _nid, n in _by_class(wf, "UnetLoaderGGUF")
        if "flux" not in n["inputs"]["unet_name"]
    ]
    assert ltx_unets == [TRANSFORMER_PREFERENCE[0]]
    explicit, _ = load_and_patch_workflow(VARIANT, prompt="x", seed=1, frames=49)
    assert explicit["5514:3059"]["inputs"]["length"] == 49


def test_bundle_family_and_vram_row():
    bundle = bundle_for_variant(VARIANT)
    assert bundle == "ltx25_flux_i2v"
    assert is_ltx25_bundle(bundle)
    assert family_for_slug(VARIANT) == "ltx25"
    row = workflow_row(VARIANT)
    assert row.vram_class == "tight"
    assert row.safer_alternate == "ltx25_t2v_i2v"


def test_patched_graph_validates_offline_with_tower_weights(tmp_path, monkeypatch):
    # Plant Klein 4B plus the LTX text encoder. The patcher rewrites the still
    # loader to the Klein file resolve_weight finds.
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    root = tmp_path / "models"
    for rel in (
        f"text_encoders/{HERETIC_TE}",
        f"text_encoders/{FLUX_TEXT}",
        f"diffusion_models/{FLUX_FP8}",
        f"vae/{FLUX_VAE}",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"w")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])
    wf, _meta = load_and_patch_workflow(VARIANT, prompt="a wet street", seed=1)
    assert wf["9101"]["class_type"] == "UNETLoader"
    assert wf["9101"]["inputs"]["unet_name"] == FLUX_FP8
    assert wf["9102"]["inputs"]["type"] == "flux2"
    assert wf["9102"]["inputs"]["clip_name"] == FLUX_TEXT
    assert wf["9103"]["inputs"]["vae_name"] == FLUX_VAE
    assert wf["9106"]["class_type"] == "EmptyFlux2LatentImage"
    patched = validate_workflow(wf, info, file_label="patched")
    assert patched.ok, [str(item) for item in patched.errors]
