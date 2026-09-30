"""Local LTX family lock (Pack B).

Video Buddy does not call the LTX Cloud API. Routing picks local Comfy
checkpoints and workflow families only.

- New scene and synced dialogue use LTX 2.5 graphs when those graphs exist.
- Multi-cut stays on LTX 2.3 directors. There is no 2.5 multi-cut graph.
- Retake and temporal extend of an existing plate stay on LTX 2.3 Pro.
  LTX 2.5 still has no retake or extend graph.
- Spatial canvas extend stays on ``ltx23_inoutpaint`` (2.3 dev-fp8).

2.3 Pro locally is the dev checkpoint, not the distilled / EROS fast path:

- ``directors`` diffusion slot: ``LTX-2.3-dev-Q4_K_S.gguf``
- ``lipsync``, ``ltx23_lipsync_v08``, ``ltx23_inoutpaint``:
  ``ltx-2.3-22b-dev-fp8.safetensors``
"""

from __future__ import annotations

# Text-only retake / temporal extend: 2.3 dev GGUF (Pro), not distilled base.
LTX23_PRO_PLATE = "directors"
# Source video already attached: 2.3 dev-fp8 lipsync graph.
LTX23_PRO_EXISTING_VIDEO = "lipsync"
LTX23_CANVAS_EXTEND = "ltx23_inoutpaint"

LTX25_NEW_SCENE = "ltx25_t2v_i2v"
LTX25_SYNCED_DIALOGUE = "ltx25_a2v"

# Filenames the Pro graphs load. Documented here so the lock has one home.
LTX23_PRO_DIFFUSION = "LTX-2.3-dev-Q4_K_S.gguf"
LTX23_PRO_CHECKPOINT = "ltx-2.3-22b-dev-fp8.safetensors"

# V1-era (LTX-2 19B) IC-LoRAs still authored on the 2.3 lipsync graph.
# Union-control has a 22B file, but the graph note says the 19B adapter
# behaves better on this plate. Detailer has no 2.3 replacement
# (Lightricks MODELS-LTX-2.3.md still lists the 19B detailer).
LTX2_UNION_CONTROL_LORA = "ltx/ltx-2-19b-ic-lora-union-control-ref0.5.safetensors"
LTX2_DETAILER_LORA = "ltx-2-19b-ic-lora-detailer.safetensors"
LTX23_UNION_CONTROL_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

NEW_SCENE_PHRASES = (
    "new scene",
    "fresh scene",
    "opening scene",
)
SYNCED_DIALOGUE_PHRASES = (
    "synced dialogue",
    "synced dialog",
    "sync dialogue",
    "sync dialog",
    "dialogue sync",
    "dialog sync",
)
# Temporal extend of a plate. "extend the canvas" / "extend the frame"
# stay on ltx23_inoutpaint and must not match these phrases.
RETAKE_EXTEND_PHRASES = (
    "retake",
    "extend the shot",
    "extend the clip",
    "extend this video",
    "extend the plate",
    "extend the take",
    "extend the existing",
)
MULTICUT_PHRASES = (
    "multi-cut",
    "multicut",
    "multi cut",
)


def is_plate_retake_or_extend(request: str | None) -> bool:
    """True for retake or temporal extend. False for canvas outpaint."""
    text = (request or "").lower()
    return any(phrase in text for phrase in RETAKE_EXTEND_PHRASES)
