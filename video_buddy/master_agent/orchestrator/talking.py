"""Photo + voice routing helpers.

A still plus a wav/mp3 is a talking clip. The default graph is LTX 2.5
audio-to-video (``ltx25_a2v``). MiniMax H3 can do the same only on the
reference graph (``h3_r2v``), and only when the user names H3 or forces
that variant. ``lipsync`` stays video-in. fl2va H3 graphs generate their
own soundtrack and must not swallow a voice file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from master_agent.config import H3_MAX_DURATION_S, LTX25_SEGMENT_MAX_S, MAX_DURATION_S

# Video-in LipDub. A still is not a source video.
LIPSYNC_VARIANTS = frozenset({"lipsync", "ltx23_lipsync_v08"})
# Reference-to-AV: still and/or video plus optional standalone reference audio.
H3_AUDIO_VARIANTS = frozenset({"h3_r2v", "ref2va", "h3_ref2va"})
# fl2va generates native stereo. It does not consume a voice file.
H3_NO_INPUT_AUDIO = frozenset({
    "h3_t2v",
    "h3_i2v",
    "h3_flf",
    "fl2va",
    "h3_fl2va",
    "h3",
    "minimax",
    "minimax_h3",
})
A2V_VARIANTS = frozenset({"ltx25_a2v", "a2v"})

_H3_REQUEST = re.compile(r"\b(?:minimax|hailuo|ref2va|h3_r2v|h3)\b", re.IGNORECASE)
_TALKING_WORDS = (
    "lip-sync",
    "lipsync",
    "lip sync",
    "lipdub",
    "talking head",
    "talking clip",
    "talking photo",
)


def canonical_variant(variant: str | None) -> str:
    key = (variant or "").strip().lower()
    if not key or key == "auto":
        return ""
    try:
        from master_agent.comfy.catalog import H3_ALIASES, RESEARCH_ALIASES

        return H3_ALIASES.get(key, RESEARCH_ALIASES.get(key, key))
    except Exception:
        return key


def requests_h3(text: str | None) -> bool:
    """True when the brief names MiniMax / Hailuo / H3 / ref2va."""
    return bool(_H3_REQUEST.search(text or ""))


def talking_keywords(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in _TALKING_WORDS)


def preview_media_variant(
    request: str,
    *,
    variant: str | None = None,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> str:
    """Cheap route for guards and duration planning. Does not call the LLM."""
    forced = canonical_variant(variant)
    if forced:
        return forced
    if has_video:
        return "lipsync"
    if has_image and has_audio:
        return "h3_r2v" if requests_h3(request) else "ltx25_a2v"
    from master_agent.orchestrator.director import rule_based_variant

    return rule_based_variant(request)


def media_route_error(
    variant: str | None,
    *,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> str | None:
    """Fail before queue when the chosen graph cannot consume the media."""
    canon = canonical_variant(variant)
    if canon in LIPSYNC_VARIANTS and not has_video:
        return (
            "lipsync needs a source video (--video). "
            "For a still photo plus a voice file, use --variant ltx25_a2v. "
            "MiniMax H3 reference audio is --variant h3_r2v (image + audio)."
        )
    if canon in H3_NO_INPUT_AUDIO and has_audio:
        return (
            f"{canon} generates its own soundtrack and does not take a voice file. "
            "H3 audio-driven lip-sync uses --variant h3_r2v "
            "(reference image + reference audio). "
            "The default photo + voice path is --variant ltx25_a2v."
        )
    if canon in H3_AUDIO_VARIANTS and has_audio and not has_image and not has_video:
        return (
            "H3 reference audio must be accompanied by a still (--image). "
            "Audio alone is not a talking clip on h3_r2v. "
            "Use --variant ltx25_a2v for audio-only, or add the photo."
        )
    return None


def skip_music_autoroute(request: str, *, has_image: bool) -> bool:
    """Photo + voice, or an explicit talking brief, is not a music video."""
    if has_image:
        return True
    return talking_keywords(request)


def is_audio_driven(
    variant: str | None,
    *,
    has_image: bool,
    has_audio: bool,
    has_video: bool,
    request: str = "",
) -> bool:
    """True when clip length and trim should follow the voice file."""
    if has_video or not has_audio:
        return False
    canon = canonical_variant(variant)
    if canon in LIPSYNC_VARIANTS or canon in H3_NO_INPUT_AUDIO:
        return False
    if canon in A2V_VARIANTS or canon in H3_AUDIO_VARIANTS:
        return True
    if not canon and has_image:
        return True
    if not canon and requests_h3(request) and has_image:
        return True
    return False


def per_clip_cap_s(variant: str | None, *, request: str = "") -> float:
    canon = canonical_variant(variant)
    if canon in H3_AUDIO_VARIANTS or canon in H3_NO_INPUT_AUDIO:
        return float(H3_MAX_DURATION_S)
    if not canon and requests_h3(request):
        return float(H3_MAX_DURATION_S)
    return float(LTX25_SEGMENT_MAX_S)


def duration_following_audio(path: str, *, probe=None) -> tuple[float, str | None]:
    """Seconds to generate from a voice file, capped at MAX_DURATION_S."""
    if probe is None:
        from master_agent.music.beats import audio_duration

        probe = audio_duration
    try:
        raw = float(probe(path) or 0.0)
    except (OSError, TypeError, ValueError):
        raw = 0.0
    if raw <= 0:
        return 5.0, (
            "could not read audio duration (ffprobe missing or unreadable file); using 5s"
        )
    if raw > float(MAX_DURATION_S) + 0.05:
        return float(MAX_DURATION_S), (
            f"audio is {raw:.1f}s; talking clip is capped at {MAX_DURATION_S:.0f}s"
        )
    return raw, None


def plan_audio_slices(total_s: float, cap_s: float) -> list[tuple[float, float]]:
    """``(duration_s, audio_start_s)`` slices. The last slice is the remainder."""
    total = min(max(float(total_s), 0.5), float(MAX_DURATION_S))
    cap = max(float(cap_s), 0.5)
    slices: list[tuple[float, float]] = []
    start = 0.0
    while start < total - 1e-3 and len(slices) < 16:
        dur = min(cap, total - start)
        slices.append((round(dur, 4), round(start, 4)))
        start += dur
    return slices or [(min(total, cap), 0.0)]


@dataclass
class TalkingSchedule:
    durations: list[float]
    audio_starts: list[float]
    note: Optional[str] = None


def plan_talking_slices(
    total_s: float,
    *,
    variant: str | None,
    request: str = "",
) -> TalkingSchedule:
    """LTX splits and continues the audio. H3 is one clip, capped, photo reused."""
    canon = canonical_variant(variant)
    h3 = canon in H3_AUDIO_VARIANTS or (not canon and requests_h3(request))
    cap = per_clip_cap_s(variant, request=request)
    total = min(max(float(total_s), 0.5), float(MAX_DURATION_S))
    note = None
    if total_s > float(MAX_DURATION_S) + 0.05:
        note = f"audio is {float(total_s):.1f}s; talking clip is capped at {MAX_DURATION_S:.0f}s"
    if h3:
        used = min(total, float(H3_MAX_DURATION_S))
        if total > used + 0.05:
            extra = f"H3 reference clip is capped at {used:.1f}s (one clip, photo held)"
            note = f"{note}; {extra}" if note else extra
        return TalkingSchedule(durations=[round(used, 4)], audio_starts=[0.0], note=note)
    slices = plan_audio_slices(total, cap)
    if len(slices) > 1:
        extra = (
            f"audio {total:.1f}s → {len(slices)} segments "
            f"(per-clip cap {cap:.1f}s); the photo is reused and the audio continues"
        )
        note = f"{note}; {extra}" if note else extra
    return TalkingSchedule(
        durations=[d for d, _s in slices],
        audio_starts=[s for _d, s in slices],
        note=note,
    )
