"""Optional HeartMuLa track in front of the existing music / MV path.

A supplied ``--audio`` file stays the track. Lyrics and tags generate a wav
only when ``--audio`` is omitted. Dry-run writes a silent wav so beat
planning can run; it does not import heartlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.heartmula.config import HeartMuLaConfigError, default_duration_s, provenance_block
from master_agent.heartmula.generate import generate_track, plan_generate, write_silent_wav


@dataclass
class TrackSource:
    audio: Path
    heartmula: dict[str, Any] | None = None
    note: str | None = None


def _env_text(name: str) -> str:
    return (os.getenv(name) or "").strip()


def lyrics_and_tags(
    lyrics: str | None,
    tags: str | None,
) -> tuple[str, str]:
    return (
        (lyrics or "").strip() or _env_text("HEARTMULA_LYRICS"),
        (tags or "").strip() or _env_text("HEARTMULA_TAGS"),
    )


def materialize_track(
    *,
    audio: str | Path | None,
    lyrics: str | None,
    tags: str | None,
    out_dir: str | Path,
    dry_run: bool,
    duration_s: float | None = None,
    seed: int | None = None,
) -> TrackSource:
    """Return the wav the beat planner should read."""
    lyric_text, tag_text = lyrics_and_tags(lyrics, tags)
    if audio:
        note = None
        if lyric_text or tag_text:
            note = "--audio is the track; HeartMuLa lyrics/tags were not used"
        return TrackSource(audio=Path(audio), heartmula=None, note=note)
    if not lyric_text or not tag_text:
        raise HeartMuLaConfigError(
            "need --audio, or both --heartmula-lyrics and --heartmula-tags "
            "(or HEARTMULA_LYRICS and HEARTMULA_TAGS)"
        )
    seconds = float(duration_s) if duration_s is not None else default_duration_s()
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav = dest_dir / "heartmula.wav"
    if dry_run:
        write_silent_wav(wav, seconds=max(seconds, 8.0))
        block = provenance_block(
            lyrics=lyric_text,
            tags=tag_text,
            seed=seed,
            wav=str(wav),
            max_audio_length_ms=int(round(seconds * 1000)),
            dry_run=True,
            placeholder="silent-wav",
        )
        plan_generate(
            lyrics=lyric_text,
            tags=tag_text,
            out=wav,
            duration_s=seconds,
            seed=seed,
            dry_run=True,
        )
        return TrackSource(audio=wav, heartmula=block, note="dry-run silent wav (not model audio)")
    result = generate_track(
        lyrics=lyric_text,
        tags=tag_text,
        out=wav,
        duration_s=seconds,
        seed=seed,
    )
    return TrackSource(audio=result.wav, heartmula=result.provenance, note=None)
