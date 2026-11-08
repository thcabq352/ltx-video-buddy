"""Look checks stay on the local llama.cpp Qwen3-VL heretic judge.

SuperGrok is not this path. A look check sends the generated image to
llama.cpp's OpenAI-compatible vision route and accepts only a visual JSON
verdict. The score must be a finite number in range, ``pass`` must be a
JSON boolean, and some field must describe the image. The word "ready" by
itself, including as the only issue, is not that description. This module
does not start, stop, or restart llama-server.
"""

from __future__ import annotations

import base64
import hashlib
import math
from typing import Any


def look_provider_spec() -> str:
    from master_agent.config import VISION_MODEL

    model = (VISION_MODEL or "qwen3-vl-heretic").strip() or "qwen3-vl-heretic"
    return f"llamacpp:{model}"


def configured() -> dict[str, Any]:
    """Static check. Does not start a server."""
    from master_agent.config import JUDGE_ENABLED, VISION_ENABLED, VISION_MODEL

    model = (VISION_MODEL or "").strip()
    if not JUDGE_ENABLED:
        return {"fired": False, "ready": False, "reason": "JUDGE_ENABLED is off"}
    if not VISION_ENABLED:
        return {"fired": False, "ready": False, "reason": "VISION_ENABLED is off"}
    if "qwen3-vl-heretic" not in model:
        return {
            "fired": False,
            "ready": False,
            "reason": f"VISION_MODEL is {model or 'unset'}; look checks require qwen3-vl-heretic on llama.cpp",
        }
    return {"fired": False, "ready": True, "provider": "llamacpp", "model": model}


def look_capability() -> dict[str, Any]:
    """Whether the local judge can see an image. Does not call the model.

    ``fired`` stays false. A capability check is not a visual pass.
    """
    state = configured()
    if not state.get("ready"):
        return {
            "fired": False,
            "ready": False,
            "reason": state.get("reason") or "the local judge is not configured",
            "provider": "llamacpp",
        }
    from master_agent.llm import LLAMACPP_URL, provider_available

    if not provider_available("llamacpp"):
        return {
            "fired": False,
            "ready": False,
            "provider": "llamacpp",
            "model": state.get("model"),
            "reason": (
                f"llama.cpp is not reachable at {LLAMACPP_URL}. "
                "The readiness check will not start or restart a server."
            ),
        }
    from master_agent.models.llm_gguf import configured_slots, mmproj_present

    brain = next((slot for slot in configured_slots() if getattr(slot, "vision", False)), None)
    projector = mmproj_present(brain) if brain is not None else None
    if projector is None:
        return {
            "fired": False,
            "ready": False,
            "provider": "llamacpp",
            "model": state.get("model"),
            "reason": "no mmproj is on disk, so the local judge cannot see images. Refusing the still.",
        }
    return {
        "fired": False,
        "ready": True,
        "multimodal": True,
        "provider": "llamacpp",
        "model": state.get("model"),
        "projector": str(projector),
    }


def fire_look_check(
    *,
    kind: str = "look",
    image: bytes | None = None,
    mime: str = "image/png",
    filename: str = "still.png",
    prompt: str = "",
    ping=None,
) -> dict[str, Any]:
    """Score one still. ``ping`` is ignored; a text ping is not a visual result."""
    del ping
    if not image:
        return {
            "fired": False,
            "accepted": False,
            "saw_image": False,
            "ready": False,
            "reason": "no still image was supplied",
            "provider": "llamacpp",
            "kind": kind,
        }
    return assess_still(image, mime=mime, filename=filename, prompt=prompt, kind=kind)


def assess_still(
    image: bytes | None,
    *,
    mime: str = "image/png",
    filename: str = "still.png",
    prompt: str = "",
    kind: str = "still",
) -> dict[str, Any]:
    """POST the exact still bytes to local llama.cpp and read a visual verdict."""
    state = configured()
    if not image:
        return {
            "fired": False,
            "accepted": False,
            "saw_image": False,
            "reason": "no still image was supplied",
            "provider": "llamacpp",
            "kind": kind,
        }
    if not state.get("ready"):
        return {
            "fired": False,
            "accepted": False,
            "saw_image": False,
            "reason": state.get("reason") or "the local judge is not configured",
            "provider": "llamacpp",
            "kind": kind,
        }
    digest = hashlib.sha256(image).hexdigest()
    encoded = base64.b64encode(image).decode("ascii")
    from master_agent.judge.vision import _vision_chat, vision_system_prompt

    system = vision_system_prompt(user_request=prompt or kind)
    user_text = (
        "Review this single generated still. Grade only what is visible in the image.\n"
        f"filename: {filename}\n"
        f"sha256: {digest}\n"
        f"brief: {prompt or kind}\n"
        "Return one JSON object. score must be a finite number from 0 to 1 "
        "(or 0 to 100). pass must be the JSON boolean true or false, not a word or number. "
        "Describe what is visible in reason, artifacts, subject_lock, or issues. "
        "The word ready alone is not a verdict."
    )
    try:
        raw = _vision_chat("llamacpp", system, user_text, [(mime, encoded)])
    except Exception as exc:
        return {
            "fired": False,
            "accepted": False,
            "saw_image": True,
            "reason": f"local judge unreachable: {type(exc).__name__}: {exc}",
            "provider": "llamacpp",
            "backend": "llamacpp",
            "kind": kind,
            "image_sha256": digest,
        }
    accepted, reason, score = _visual_verdict(raw if isinstance(raw, str) else str(raw or ""))
    return {
        "fired": accepted,
        "accepted": accepted,
        "saw_image": True,
        "reason": reason,
        "score": score,
        "provider": "llamacpp",
        "backend": "llamacpp",
        "model": state.get("model"),
        "kind": kind,
        "image_sha256": digest,
    }


def _ready_word(text: str) -> bool:
    token = (text or "").strip().lower().strip(".! ")
    return token == "ready"


def _meaningful_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _ready_word(text):
        return None
    return text


def _finite_score(value: object) -> float | None:
    """A real confidence. Booleans, NaN, infinities, and non-numbers are not scores."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            number = float(token)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _visual_verdict(raw: str) -> tuple[bool, str, float | None]:
    from master_agent.judge.vision import _extract_json

    text = (raw or "").strip()
    if not text:
        return False, "local judge returned an empty response", None
    if _ready_word(text):
        return False, "local judge replied with the word ready and did not assess the image", None
    data = _extract_json(text)
    if not isinstance(data, dict):
        return False, "local judge did not return a visual JSON verdict", None
    score = _finite_score(data.get("score"))
    if score is None:
        return False, "local judge score is not a finite number", None
    if score > 1.0:
        score = score / 100.0
    if score < 0.0 or score > 1.0 or not math.isfinite(score):
        return False, "local judge score is outside 0-1", score
    described: list[str] = []
    for key in ("reason", "artifacts", "subject_lock", "temporal_consistency", "temporal"):
        meaningful = _meaningful_text(data.get(key))
        if meaningful:
            described.append(meaningful)
    issues = data.get("issues")
    if isinstance(issues, list):
        for item in issues:
            meaningful = _meaningful_text(item)
            if meaningful:
                described.append(meaningful)
    if not described:
        return False, "local judge did not describe the image", score
    flag = data.get("pass")
    if not isinstance(flag, bool):
        return False, "local judge pass flag is missing or not a boolean", score
    if flag is False or score < 0.75:
        return False, f"local judge rejected the still ({described[0]}; score={score:.2f})", score
    return True, described[0], score
