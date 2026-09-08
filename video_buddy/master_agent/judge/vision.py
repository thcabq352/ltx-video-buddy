"""Vision evaluator — frame-level quality analysis via a local VL model.

Extracts evenly-spaced frames with ffmpeg and asks qwen3-vl-heretic (Ollama) to
check temporal consistency, subject lock, and visible artifacts. Result
feeds the judge as a third scoring leg alongside heuristics and the text
LLM verdict. Best-effort: any failure returns None and the judge proceeds
without it.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from master_agent.config import (
    OLLAMA_URL,
    VISION_ENABLED,
    VISION_FRAMES,
    VISION_MODEL,
    VISION_TIMEOUT_S,
)


def vision_available() -> bool:
    if not VISION_ENABLED:
        return False
    if not shutil.which("ffmpeg"):
        return False
    try:
        from master_agent.llm import provider_available

        return provider_available("ollama")
    except Exception:
        return False


def extract_frames(video_path: str | Path, n: int | None = None) -> list[Path]:
    """Up to n evenly-spaced JPEG frames (small, for VL context). [] on failure."""
    ffmpeg = shutil.which("ffmpeg")
    p = Path(video_path)
    if not ffmpeg or not p.is_file():
        return []
    count = int(n or VISION_FRAMES or 4)
    try:
        td = tempfile.mkdtemp(prefix="ma_vision_")
        pattern = str(Path(td) / "f_%03d.jpg")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(p),
                "-vf",
                f"fps=1,scale=384:-1",
                "-frames:v",
                str(count),
                pattern,
            ],
            capture_output=True,
            timeout=60,
        )
        return sorted(Path(td).glob("f_*.jpg"))
    except Exception:
        return []


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def vision_review(
    video_path: str | Path | None,
    *,
    user_request: str,
    ltx_prompt: str = "",
    full_video: bool = False,
    reference_paths: list | None = None,
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
    if p.suffix.lower() in _IMAGE_SUFFIXES:
        # Still image: no ffmpeg needed, just the VL provider
        try:
            from master_agent.llm import provider_available

            if not provider_available("ollama"):
                return None
        except Exception:
            return None
        frames = [p] if p.is_file() else []
    else:
        if not vision_available():
            return None
        frames = extract_frames(p)
    if not frames:
        return None

    prompt_path = Path(__file__).resolve().parent / "prompts" / "vision.md"
    system = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else (
        "Review video frames. Return JSON score/pass/issues/reason."
    )
    brief = {
        "user_request": user_request,
        "ltx_prompt": ltx_prompt,
        "frame_count": len(frames),
        "full_stitched_video": full_video,
    }
    images = []
    for fp in frames:
        enc = _b64(fp)
        if enc:
            images.append(enc)
    if not images:
        return None

    ref_images: list[str] = []
    for rp in reference_paths or []:
        rp = Path(rp)
        if rp.is_file():
            enc = _b64(rp)
            if enc:
                ref_images.append(enc)
    if ref_images:
        brief["reference_images"] = len(ref_images)
        brief["identity_check"] = True
        system += (
            "\n\nThe LAST image(s) are reference shots of the same subject. "
            "Also return identity_score (0-1): how consistently the reviewed "
            "image(s) match the reference subject's identity (face, hair, "
            "key features). 1.0 = clearly the same character."
        )

    try:
        import httpx

        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "Review these frames against this brief:\n"
                        + json.dumps(brief, indent=2),
                        "images": images + ref_images,
                    },
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0.2, "num_predict": 2048},
            },
            timeout=VISION_TIMEOUT_S,
        )
        resp.raise_for_status()
        text = (resp.json().get("message") or {}).get("content") or ""
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
        }
        if ref_images:
            result["identity_score"] = _score_01(data.get("identity_score"), score)
        return result
    except Exception:
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
