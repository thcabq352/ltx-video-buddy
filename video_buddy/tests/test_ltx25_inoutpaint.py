"""LTX 2.5 in/outpaint graph, patch, routing, and offline validation. No GPU.

Run: python -m pytest tests/test_ltx25_inoutpaint.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from master_agent.comfy.inoutpaint import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    LTX25_VARIANT,
    MASK_PLACEHOLDER,
    OFFICIAL_NEGATIVE,
    ensure_inoutpaint_mask,
    finalize_inoutpaint_graph,
    outpaint_layout,
    record_comfy_provenance,
)
from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR
from master_agent.models.weights import ltx23_inoutpaint_lora_placement
from master_agent.orchestrator.director import choose_variant, rule_based_variant
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, missing_required

GGUF = "gguf/ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"
IC_LORA = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
OFFICIAL_TE = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
HERETIC_TE = "gemma4-12b-heretic-ltx25-int8convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.7250, 0.4219, 0.0"


def _classes(wf: dict) -> set[str]:
    return {
        node.get("class_type")
        for node in wf.values()
        if isinstance(node, dict)
    }


def test_raw_graph_follows_official_25_loaders():
    raw = json.loads((WORKFLOWS_DIR / WORKFLOW_FILES[LTX25_VARIANT]).read_text(encoding="utf-8"))
    assert raw["10"]["class_type"] == "UnetLoaderGGUF"
    assert raw["10"]["inputs"]["unet_name"] == GGUF
    assert raw["11"]["class_type"] == "CLIPLoader"
    assert raw["11"]["inputs"]["clip_name"] == OFFICIAL_TE
    assert raw["11"]["inputs"]["type"] == "ltxv"
    assert raw["12"]["inputs"]["vae_name"] == VIDEO_VAE
    assert raw["13"]["inputs"]["vae_name"] == AUDIO_VAE
    assert raw["15"]["inputs"]["lora_name"] == IC_LORA
    assert raw["15"]["inputs"]["strength_model"] == 1.0
    assert raw["41"]["inputs"]["latent_downscale_factor"] == 1
    assert raw["41"]["inputs"]["use_tiled_encode"] is False
    assert raw["41"]["inputs"]["attention_mask"] == ["36", 0]
    assert raw["38"]["inputs"]["img_compression"] == 18
    assert raw["40"]["inputs"]["strength"] == 0.7
    assert raw["40"]["class_type"] == "LTXVImgToVideoInplace"
    assert raw["51"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert raw["70"]["inputs"]["sampler_name"] == "euler"
    assert raw["52"]["inputs"]["sigmas"] == STAGE1_SIGMAS
    assert raw["71"]["inputs"]["sigmas"] == STAGE2_SIGMAS
    assert raw["53"]["inputs"]["cfg"] == 1
    assert raw["55"]["inputs"]["av_latent"] == ["54", 1]
    assert raw["74"]["inputs"]["av_latent"] == ["73", 0]
    assert raw["57"]["class_type"] == "VAEDecodeTiled"
    assert raw["57"]["inputs"]["tile_size"] == 512
    assert raw["57"]["inputs"]["overlap"] == 64
    assert raw["57"]["inputs"]["temporal_size"] == 128
    assert raw["57"]["inputs"]["temporal_overlap"] == 32
    assert raw["64"]["inputs"]["resize_type"] == "scale by multiplier"
    assert raw["64"]["inputs"]["resize_type.multiplier"] == 2
    assert raw["64"]["inputs"]["scale_method"] == "lanczos"
    assert raw["65"]["class_type"] == "VAEEncodeTiled"
    assert raw["58"]["inputs"]["mask_low_res_dilation"] == 5
    assert raw["76"]["inputs"]["mask_low_res_dilation"] == 6
    banned = _classes(raw)
    assert "LoraLoaderModelOnly" not in banned
    assert "LTXVLatentUpsampler" not in banned
    assert "LTXVTiledVAEDecode" not in banned
    assert "GemmaAPITextEncode" not in banned
    assert "LatentUpscaleModelLoader" not in banned


def test_patch_writes_prompt_and_keeps_ic_lora():
    wf, meta = load_and_patch_workflow(
        LTX25_VARIANT,
        prompt="UNIQUE_SANDY_BEACH",
        seed=7,
        duration_s=5.0,
    )
    assert wf["20"]["inputs"]["text"] == "UNIQUE_SANDY_BEACH"
    assert wf["21"]["inputs"]["text"] == OFFICIAL_NEGATIVE
    assert wf["39"]["inputs"]["length"] == 25
    assert wf["39"]["inputs"]["width"] == DEFAULT_WIDTH // 2
    assert wf["39"]["inputs"]["height"] == DEFAULT_HEIGHT // 2
    assert meta["width"] == DEFAULT_WIDTH
    assert meta["height"] == DEFAULT_HEIGHT
    assert meta["frames"] == 25
    assert wf["15"]["inputs"]["lora_name"] == IC_LORA
    assert wf["15"]["inputs"]["strength_model"] == 1.0
    assert wf["36"]["inputs"]["spatial_radius"] == 15
    assert wf["62"]["inputs"]["spatial_radius"] == 30
    assert wf["58"]["inputs"]["mask_low_res_dilation"] == 5
    assert wf["76"]["inputs"]["mask_low_res_dilation"] == 6
    assert wf["33"]["inputs"]["image"] == MASK_PLACEHOLDER
    assert wf["30"]["inputs"]["file"] == "robot_unitree.mp4"
    assert wf["40"]["class_type"] == "LTXVImgToVideoInplace"
    assert wf["40"]["inputs"]["strength"] == 0.7
    assert wf["79"]["inputs"]["filename_prefix"] == "ltx25_inoutpaint"
    # No local 2.5 file in this checkout: the patcher falls back to official bf16.
    assert wf["10"]["class_type"] == "UNETLoader"
    assert wf["10"]["inputs"]["unet_name"] == "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    assert "ltx-2.3" not in wf["10"]["inputs"]["unet_name"]


def test_local_gguf_and_heretic_stay_on_the_loaders(tmp_path, monkeypatch):
    root = tmp_path / "models"
    gguf = root / "diffusion_models" / "gguf" / "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf"
    te = root / "text_encoders" / HERETIC_TE
    gguf.parent.mkdir(parents=True)
    te.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    te.write_bytes(b"te")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])
    wf, _meta = load_and_patch_workflow(
        LTX25_VARIANT,
        prompt="tower layout",
        seed=1,
        frames=25,
    )
    assert wf["10"]["class_type"] == "UnetLoaderGGUF"
    assert wf["10"]["inputs"]["unet_name"].endswith("ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf")
    assert wf["11"]["inputs"]["clip_name"] == HERETIC_TE
    assert wf["11"]["inputs"]["type"] == "ltxv"
    assert wf["15"]["inputs"]["lora_name"] == IC_LORA


def test_outpaint_keeps_official_blend_and_uses_condition_only():
    plan = outpaint_layout(1280, 720, aspect="9:16")
    wf, meta = load_and_patch_workflow(
        LTX25_VARIANT,
        prompt="continue the street",
        seed=3,
        width=plan["width"],
        height=plan["height"],
        frames=25,
        inoutpaint={
            "mode": "outpaint",
            "pad": plan["pad"],
            "mask_png": plan["mask_png"],
        },
    )
    pad = wf["32"]["inputs"]
    assert pad["left"] == plan["pad"]["left"]
    assert pad["top"] == plan["pad"]["top"]
    assert pad["right"] == plan["pad"]["right"]
    assert pad["bottom"] == plan["pad"]["bottom"]
    assert wf["36"]["inputs"]["spatial_radius"] == 0
    assert wf["62"]["inputs"]["spatial_radius"] == 0
    assert wf["58"]["inputs"]["mask_low_res_dilation"] == 5
    assert wf["76"]["inputs"]["mask_low_res_dilation"] == 6
    assert wf["40"]["class_type"] == "LTXVImgToVideoConditionOnly"
    assert wf["66"]["class_type"] == "LTXVImgToVideoConditionOnly"
    assert wf["40"]["inputs"]["strength"] == 0.7
    assert "inoutpaint_png_b64" in (wf["33"].get("_meta") or {})
    assert meta["width"] == plan["width"]
    assert wf["20"]["inputs"]["text"] == "continue the street"

    uploaded: list[str] = []

    def _upload(path: Path) -> str:
        assert path.name == "ltx25_inoutpaint_mask.png"
        assert path.read_bytes().startswith(b"\x89PNG")
        uploaded.append("ltx25_mask_uploaded.png")
        return uploaded[-1]

    name = ensure_inoutpaint_mask(wf, _upload)
    assert name == "ltx25_mask_uploaded.png"
    assert wf["33"]["inputs"]["image"] == name
    assert "inoutpaint_png_b64" not in (wf["33"].get("_meta") or {})


def test_user_mask_name_is_not_replaced():
    wf, _meta = load_and_patch_workflow(
        LTX25_VARIANT,
        prompt="fill the sign",
        seed=1,
        frames=25,
        mask_name="my_mask.png",
    )
    assert wf["33"]["inputs"]["image"] == "my_mask.png"

    def _upload(_path: Path) -> str:
        raise AssertionError("user mask should not be regenerated")

    assert ensure_inoutpaint_mask(wf, _upload) is None


def test_template_and_patched_graph_validate_offline():
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    raw = json.loads((WORKFLOWS_DIR / WORKFLOW_FILES[LTX25_VARIANT]).read_text(encoding="utf-8"))
    report = validate_workflow(raw, info, file_label="ltx25_inoutpaint_api.json")
    assert report.ok, [str(item) for item in report.errors]
    wf, _meta = load_and_patch_workflow(LTX25_VARIANT, prompt="a wet street", seed=1, frames=17)
    patched = validate_workflow(wf, info, file_label="patched")
    assert patched.ok, [str(item) for item in patched.errors]
    warning_text = " ".join(str(item) for item in report.warnings + patched.warnings)
    assert "in-outpainting" in warning_text
    assert "Q4_K_M" in warning_text or "distilled-transformer-bf16" in warning_text


def test_director_routes_25_without_stealing_23_or_wan():
    assert rule_based_variant("please inpaint this plate") == "wan_fun_inpaint"
    assert rule_based_variant("ltx outpaint the frame") == "ltx23_inoutpaint"
    assert rule_based_variant("extend the canvas to vertical") == "ltx23_inoutpaint"
    assert rule_based_variant("ltx 2.5 alley push-in") == "ltx25_t2v_i2v"
    assert rule_based_variant("ltx 2.5 inpaint the sign") == LTX25_VARIANT
    assert rule_based_variant("ltx25 outpaint the frame") == LTX25_VARIANT
    assert rule_based_variant("use ltx25_inoutpaint") == LTX25_VARIANT
    assert choose_variant("outpaint this clip to 9:16", has_video=True) == (
        "ltx23_inoutpaint",
        "input",
    )
    assert choose_variant("ltx 2.5 inpaint this plate", has_video=True) == (
        LTX25_VARIANT,
        "input",
    )
    assert choose_variant("please inpaint this plate", has_video=True) == ("lipsync", "input")


def test_doctor_lora_text_names_both_variants(tmp_path: Path):
    place = ltx23_inoutpaint_lora_placement([tmp_path])
    assert "ltx23_inoutpaint" in place["detail"]
    assert "ltx25_inoutpaint" in place["detail"]


def test_comfy_run_provenance_uses_existing_schema(tmp_path: Path):
    wf, _meta = load_and_patch_workflow(
        LTX25_VARIANT,
        prompt="UNIQUE_SANDY_BEACH",
        seed=11,
        frames=25,
        video_name="robot_unitree.mp4",
    )
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real mp4")
    payload = record_comfy_provenance(wf, clip, variant=LTX25_VARIANT)
    assert payload is not None
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(payload) == []
    assert payload["prompts"]["positive"] == "UNIQUE_SANDY_BEACH"
    assert payload["engine"]["variant"] == LTX25_VARIANT
    assert payload["params"]["fps"] == 24
    assert "robot_unitree.mp4" in payload["params"]["refs"]
    assert record_comfy_provenance(wf, clip, variant="lipsync") is None
    assert record_comfy_provenance(wf, clip, variant="ltx23_inoutpaint") is not None


def test_finalize_halves_stage_size_and_keeps_23_blend():
    wf = {
        "34": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"width": 1, "height": 1},
            "_meta": {"title": "Resize Frames Stage 1"},
        },
        "60": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {"width": 1, "height": 1},
            "_meta": {"title": "Resize Frames Stage 2"},
        },
        "39": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 9, "height": 9, "length": 25}},
        "40": {"class_type": "LTXVImgToVideoInplace", "inputs": {"strength": 0.7, "bypass": True}},
        "58": {
            "class_type": "LTXVLaplacianPyramidBlend",
            "inputs": {"mask_low_res_dilation": 1},
            "_meta": {"title": "Blend Stage 1"},
        },
        "76": {
            "class_type": "LTXVLaplacianPyramidBlend",
            "inputs": {"mask_low_res_dilation": 1},
            "_meta": {"title": "Blend Stage 2"},
        },
    }
    finalize_inoutpaint_graph(wf, width=768, height=448, mode="outpaint", ltx25=True)
    assert wf["60"]["inputs"]["width"] == 768
    assert wf["39"]["inputs"]["width"] == 384
    assert wf["39"]["inputs"]["height"] == 224
    assert wf["58"]["inputs"]["mask_low_res_dilation"] == 5
    assert wf["76"]["inputs"]["mask_low_res_dilation"] == 6
    assert wf["40"]["class_type"] == "LTXVImgToVideoConditionOnly"

    legacy = {
        "75": {
            "class_type": "LTXVLaplacianPyramidBlend",
            "inputs": {"mask_low_res_dilation": 6},
            "_meta": {"title": "Blend Stage 2"},
        },
        "42": {"class_type": "LTXVImgToVideoInplace", "inputs": {"strength": 1, "bypass": True}},
    }
    finalize_inoutpaint_graph(legacy, width=768, height=448, mode="outpaint")
    assert legacy["75"]["inputs"]["mask_low_res_dilation"] == 2
    assert legacy["42"]["class_type"] == "LTXVImgToVideoInplace"
