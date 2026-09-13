"""Director brain — LLM request routing.

Decides which workflow variant a request should use. The allowlist is
every ``workflows/manifests.yaml`` slug (via ``WORKFLOW_FILES`` /
``load_workflow_files``). Hard constraints (forced variant, source video)
always win; otherwise the local LLM picks, with keyword rules as fallback.
The LLM never sees secrets and its answer is validated against the known
variant list before use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.catalog import default_variant_ids, is_known_variant
from master_agent.config import DIRECTOR_LLM, WORKFLOW_FILES

# First match wins. Keep specific family phrases before generic ones
# (`vfx` / `ccc` / `shots`). Hard constraints in choose_variant still win.
_VARIANT_KEYWORDS = [
    ("lipsync", ("lip-sync", "lipsync", "lip sync", "lipdub", "dub")),
    ("ltx23_lipsync_v08", (
        "3d-rendering lip", "clay lipsync", "omninft", "lip-snyc",
        "ltx23 lipsync", "ltx 2.3 lipsync",
    )),
    ("vb_aivfx_preprocess", (
        "vfx preprocess", "aivfx preprocess", "ai-vfx preprocess",
        "sam3", "depthcrafter", "cotracker", "rmbg",
    )),
    ("vb_aivfx_startimage", (
        "vfx start", "aivfx start", "ai-vfx start", "start-image",
        "start image", "qwen start",
    )),
    ("vb_aivfx_adv_13", ("aivfx 1.3", "vfx 1.3", "aivfx_13", "aivfx 13", "ai-vfx 1.3")),
    ("vb_aivfx_adv", (
        "aivfx", "ai-vfx", "ai vfx", "possession", "vace", "phantom",
        "composite", "vfx",
    )),
    ("vb_movie_builder", (
        "movie builder", "movie-builder", "shot assembler", "shot-by-shot",
        "feature pipeline",
    )),
    ("vb_ccc41_krea2", (
        "ccc 4.1", "ccc41", "krea2-edit", "krea2 edit", "grounded character",
    )),
    ("vb_ccc_adv", (
        "consistent character", "character creator", "ccc 4.01", "ccc4", "ccc",
    )),
    ("vb_dataset_tagger", (
        "dataset tagger", "auto-tag", "auto tag", "caption pairs", "lora dataset",
    )),
    ("vb_tag_review", ("tag review", "caption review")),
    ("vb_qwen_edit_360", (
        "qwen edit", "qwen-image-edit", "360 turnaround", "360 panorama",
        "equirectangular",
    )),
    ("vb_ideogram", ("ideogram",)),
    ("krea2_img", ("krea2", "krea-2", "krea 2")),
    ("flux", ("flux.1", "flux1", "character sheet", "text-to-image", "flux")),
    ("vb_zimage_turbo_cn", ("z-image", "zimage", "fun-controlnet", "turbo cn")),
    ("vb_rtx_superres", (
        "rtx super", "super resolution", "nvidia superres", "rtx vsr",
    )),
    ("vb_ai_renderer_smpl", (
        "renderer simple", "ai-renderer smpl", "smpl renderer",
    )),
    ("vb_ai_renderer_adv_20", ("renderer 2.0", "ai-renderer 2.0")),
    ("vb_ai_renderer_adv", ("ai-renderer", "ai renderer", "renderer adv")),
    ("air_render_030", ("bear minimum right", "right this way")),
    ("air_render_050", ("bear minimum bar", "bear bar")),
    ("air_render_businesswoman", ("businesswoman", "business woman")),
    ("wan22", ("wan 2.2", "wan2.2", "wan22", "photoreal", "photo-real", "stock photo", "film grain")),
    ("h3_r2v", ("ref2va", "h3 r2v", "h3_r2v", "reference-to-video", "minimax r2v", "minimax reference")),
    ("h3_flf", ("h3 flf", "h3_flf", "minimax first-last", "minimax flf")),
    ("h3_i2v", ("h3 i2v", "h3_i2v", "minimax i2v", "minimax image-to-video")),
    ("h3_t2v", ("minimax h3", "minimax-h3", "minimax_h3", "fl2va", "h3 t2v", "h3_t2v", "native stereo")),
    ("ltx25_flf2v", ("flf2v", "first-last", "first last frame", "last frame", "start and end frame")),
    ("ltx25_msr", ("multi-reference", "multi reference", "msr", "pic1", "reference sheet")),
    ("ltx25_v2v_ic_lora", ("ic-lora", "iclora", "ic lora", "video-to-video", "v2v")),
    ("ltx25_a2v", ("audio-to-video", "audio to video", "a2v")),
    ("ltx25_t2a", ("text-to-audio", "text to audio", "t2a", "audio only")),
    ("ltx25_t2v_i2v_two_stage", ("two-stage", "two stage", "latent upscale", "ltx 2.5 two")),
    ("ltx25_t2v_i2v", ("ltx 2.5", "ltx2.5", "ltx25", "ltx-2.5")),
    ("directors", ("director", "storyboard", "scene", "shots", "multi-shot")),
    ("eros", ("eros", "10eros")),
]


def _allowed_variants() -> set[str]:
    allowed = set(WORKFLOW_FILES.keys())
    try:
        allowed.update(default_variant_ids())
    except Exception:
        pass
    return allowed


def rule_based_variant(request: str) -> str:
    text = (request or "").lower()
    allowed = set(WORKFLOW_FILES)
    for variant, keywords in _VARIANT_KEYWORDS:
        if variant not in allowed:
            continue
        if any(k in text for k in keywords):
            return variant
    return "base"


def choose_variant(
    request: str,
    *,
    has_video: bool = False,
    force: str | None = None,
) -> tuple[str, str]:
    """Returns (variant, source) where source is forced|input|llm|rules."""
    if force:
        return force, "forced"
    if has_video:
        return "lipsync", "input"
    fallback = rule_based_variant(request)
    if not DIRECTOR_LLM:
        return fallback, "rules"
    llm_variant = _llm_variant(request, fallback=fallback)
    if llm_variant:
        return llm_variant, "llm"
    return fallback, "rules"


def _llm_variant(request: str, *, fallback: str) -> Optional[str]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from master_agent.llm import get_llm
    except Exception:
        return None

    path = Path(__file__).resolve().parent / "prompts" / "director.md"
    system = path.read_text(encoding="utf-8") if path.is_file() else (
        "Pick the workflow variant. Return JSON variant/reason."
    )
    payload = {
        "request": request,
        "allowed_variants": sorted(_allowed_variants()),
        "rule_based_suggestion": fallback,
    }
    try:
        llm = get_llm(temperature=0.1)
        resp = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Route this request:\n" + json.dumps(payload, indent=2)
                ),
            ]
        )
        text = getattr(resp, "content", None) or str(resp)
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        variant = str(data.get("variant") or "").strip().lower()
        return variant if variant in _allowed_variants() or is_known_variant(variant) else None
    except Exception:
        return None


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
