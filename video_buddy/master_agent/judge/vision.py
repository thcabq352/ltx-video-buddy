"""Vision evaluator — frame-level quality analysis via a local VL model.

Extracts evenly-spaced frames with ffmpeg and asks the local VL model
(Ollama ``/api/chat`` or llama.cpp OpenAI-compat ``/v1/chat/completions``
with image parts) to check temporal consistency, subject lock, and visible
artifacts. Result feeds the judge as a third scoring leg alongside heuristics
and the text LLM verdict. Best-effort: any failure returns None and the
judge proceeds without it (heuristic-only). Does **not** require Ollama —
llama.cpp is enough when a VL GGUF is loaded. A text-only llama.cpp server
degrades the same way (None).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from master_agent.config import (
    VISION_ENABLED,
    VISION_FRAMES,
    VISION_MODEL,
    VISION_TIMEOUT_S,
)
from master_agent.llm import active_local_backend, vision_endpoint

log = logging.getLogger(__name__)


def vision_available() -> bool:
    if not VISION_ENABLED:
        return False
    if not shutil.which("ffmpeg"):
        return False
    try:
        return active_local_backend() is not None
    except Exception:
        return False


def extract_frames(
    video_path: str | Path,
    n: int | None = None,
) -> list[Path]:
    """Up to n JPEGs spread across the whole clip (earliest first). [] on failure."""
    return [path for path, _t in extract_frames_timed(video_path, n)]


def extract_frames_timed(
    video_path: str | Path,
    n: int | None = None,
) -> list[tuple[Path, float]]:
    """``(jpeg, time_s)`` pairs, earliest first. [] on failure.

    Samples by frame index across the full duration. A 1 fps cap on the
    first N seconds misreads a held opening as a frozen clip and hands the
    vision model frames that are not the ones it describes.
    """
    from master_agent.judge.probe import extract_sampled_jpegs

    p = Path(video_path)
    if not p.is_file():
        return []
    count = int(n or VISION_FRAMES or 4)
    try:
        td = Path(tempfile.mkdtemp(prefix="ma_vision_"))
        return extract_sampled_jpegs(p, count, scale_width=384, dest_dir=td)
    except Exception:
        return []


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def _mime_for(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "image/jpeg")


def _vision_chat(
    backend: str,
    system: str,
    user_text: str,
    images: list[tuple[str, str]],
) -> str:
    """POST the VL request. *images* are (mime, b64). Returns assistant text."""
    import httpx

    url = vision_endpoint(backend)
    if backend == "llamacpp":
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for mime, enc in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{enc}"},
                }
            )
        resp = httpx.post(
            url,
            json={
                "model": VISION_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            },
            timeout=VISION_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("llama.cpp vision: empty choices (is a VL model loaded?)")
        msg = (choices[0] or {}).get("message") or {}
        return str(msg.get("content") or "")

    resp = httpx.post(
        url,
        json={
            "model": VISION_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user_text,
                    "images": [enc for _mime, enc in images],
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 2048},
        },
        timeout=VISION_TIMEOUT_S,
    )
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content") or ""


def vision_system_prompt(
    *,
    user_request: str = "",
    ltx_prompt: str = "",
    rubric: str | None = None,
) -> str:
    """Vision system prompt. ``rubric=None`` resolves env / brief keyword."""
    from master_agent.judge.rubric import KNOWN_RUBRICS, load_rubric_text, resolve_judge_rubric

    if rubric is None:
        selected = resolve_judge_rubric(user_request=user_request, ltx_prompt=ltx_prompt)
    else:
        selected = rubric if rubric in KNOWN_RUBRICS else None
    return load_rubric_text(selected, kind="vision")


def vision_review(
    video_path: str | Path | None,
    *,
    user_request: str,
    ltx_prompt: str = "",
    full_video: bool = False,
    reference_paths: list | None = None,
    rubric: str | None = None,
) -> Optional[dict[str, Any]]:
    """VL verdict for a clip: {score, pass, issues, temporal, artifacts, reason}.

    score is 0-1. None when vision is disabled/unavailable/fails.
    Still images (png/jpg/...) are reviewed directly without ffmpeg.
    When ``reference_paths`` (e.g. a hero character shot) are given, the
    references are sent along and the verdict also carries a 0-1
    ``identity_score`` measuring consistency with the reference.
    """
    if not video_path or not VISION_ENABLED:
        return None
    p = Path(video_path)
    try:
        backend = active_local_backend()
    except Exception:
        backend = None
    if not backend:
        return None
    if p.suffix.lower() in _IMAGE_SUFFIXES:
        frames = [p] if p.is_file() else []
        timed = [(path, 0.0) for path in frames]
    else:
        if not vision_available():
            return None
        timed = extract_frames_timed(p)
        frames = [path for path, _t in timed]
    if not frames:
        return None

    system = vision_system_prompt(
        user_request=user_request,
        ltx_prompt=ltx_prompt,
        rubric=rubric,
    )
    brief = {
        "user_request": user_request,
        "ltx_prompt": ltx_prompt,
        "frame_count": len(frames),
        "frame_order": "temporal, earliest first",
        "frames": [
            {"index": i + 1, "time_s": t} for i, (_path, t) in enumerate(timed)
        ],
        "full_stitched_video": full_video,
    }
    images: list[tuple[str, str]] = []
    for fp in frames:
        enc = _b64(fp)
        if enc:
            images.append((_mime_for(fp), enc))
    if not images:
        return None

    ref_images: list[tuple[str, str]] = []
    for rp in reference_paths or []:
        rp = Path(rp)
        if rp.is_file():
            enc = _b64(rp)
            if enc:
                ref_images.append((_mime_for(rp), enc))
    if ref_images:
        brief["reference_images"] = len(ref_images)
        brief["identity_check"] = True
        system += (
            "\n\nThe LAST image(s) are reference shots of the same subject. "
            "Also return identity_score (0-1): how consistently the reviewed "
            "image(s) match the reference subject's identity (face, hair, "
            "key features). 1.0 = clearly the same character."
        )

    user_text = "Review these frames against this brief:\n" + json.dumps(brief, indent=2)
    try:
        text = _vision_chat(backend, system, user_text, images + ref_images)
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        score = data.get("score")
        try:
            score = float(score)
            if score > 1.0:  # tolerate 0-100 scales
                score = score / 100.0
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return None
        result = {
            "score": score,
            "pass": bool(data.get("pass", score >= 0.75)),
            "issues": [str(x) for x in (data.get("issues") or [])][:8],
            "temporal": str(data.get("temporal_consistency") or ""),
            "artifacts": str(data.get("artifacts") or ""),
            "subject_lock": str(data.get("subject_lock") or ""),
            "reason": str(data.get("reason") or "")[:500],
            "backend": backend,
        }
        if ref_images:
            result["identity_score"] = _score_01(data.get("identity_score"), score)
        from master_agent.judge.rubric import LOOK_DIMS, clamp01

        for key in LOOK_DIMS:
            if key in data and data.get(key) is not None:
                value = clamp01(data.get(key))
                if value is not None:
                    result[key] = value
        for key in ("brief_adherence", "look_score"):
            if key in data and data.get(key) is not None:
                value = clamp01(data.get(key))
                if value is not None:
                    result[key] = value
        if "identity_morph" in data:
            result["identity_morph"] = data.get("identity_morph")
        if isinstance(data.get("hard_fails"), list):
            result["hard_fails"] = [str(item) for item in data["hard_fails"]]
        return result
    except Exception as exc:
        log.warning("vision judge skipped via %s: %s", backend, exc)
        return None


def _score_01(value: Any, fallback: float) -> float:
    """Normalize a VL score to 0-1, tolerating 0-100 scales."""
    try:
        v = float(value)
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return fallback


def vision_notes(review: dict[str, Any]) -> str:
    """One-line summary of a vision review, for the text judge's context."""
    parts = [f"vision_score={review.get('score'):.2f}"]
    if review.get("temporal"):
        parts.append(f"temporal={review['temporal'][:80]}")
    if review.get("artifacts"):
        parts.append(f"artifacts={review['artifacts'][:80]}")
    if review.get("reason"):
        parts.append(f"vision_reason={review['reason'][:160]}")
    return ", ".join(parts)


def _extract_json(text: str) -> Optional[dict]:
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
