"""Director brain — LLM request routing.

Decides which workflow variant a request should use. Hard constraints
(forced variant, source video) always win; otherwise the local LLM picks,
with the old keyword rules as fallback. The LLM never sees secrets and its
answer is validated against the known variant list before use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.catalog import default_variant_ids, is_known_variant
from master_agent.config import DIRECTOR_LLM, WORKFLOW_FILES

_VARIANT_KEYWORDS = [
    ("lipsync", ("lip-sync", "lipsync", "lip sync", "lipdub", "dub")),
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
    for variant, keywords in _VARIANT_KEYWORDS:
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
