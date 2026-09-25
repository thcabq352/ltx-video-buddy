"""H3 voice-sample mode: 2–12 s reference, spoken line, no extra deps.

A sample under 2 s is rejected before anything is queued. Over 12 s is
trimmed to the loudest continuous 12 s window (ffmpeg + RMS). The spoken
line is injected into the H3 prompt word for word. No transcription model
is downloaded; a local transcriber is used only when one is already
installed and a model file is already on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from master_agent.config import H3_MAX_DURATION_S

H3_VOICE_MIN_S = 2.0
H3_VOICE_MAX_S = float(H3_MAX_DURATION_S)
TRIM_METHOD = "rms_energy_window"
PASSTHROUGH_METHOD = "passthrough"

H3_MISSING_LINE_WARNING = (
    'H3 voice mode has no spoken line. Pass --line "the exact words" '
    "(or line / dialogue on the brief, A2A, or MCP request) so H3 says that "
    "sentence in the sample's voice and animates the mouth. "
    "The run continues, but the mouth may not follow the voice."
)

_SPOKEN_MARK = "saying word for word:"


class VoiceSampleError(Exception):
    """Raised before a queue when the H3 voice sample cannot be used."""


@dataclass
class VoiceSample:
    path: str
    original_path: str
    original_duration_s: float
    start_s: float
    end_s: float
    method: str
    trimmed: bool

    def provenance(self) -> dict[str, Any]:
        """Optional ClipProvenance params.voice_sample (same schema, extra key)."""
        return {
            "original_duration_s": round(float(self.original_duration_s), 4),
            "start_s": round(float(self.start_s), 4),
            "end_s": round(float(self.end_s), 4),
            "method": self.method,
            "trimmed": bool(self.trimmed),
        }


def _too_short_message(duration_s: float) -> str:
    return (
        f"H3 voice sample is {duration_s:.2f}s, shorter than 2 s. "
        "Record at least 2 seconds of the reference voice (up to 12 s) and "
        "pass it with --audio. Nothing was queued."
    )


def _unreadable_message() -> str:
    return (
        "Could not read the H3 voice sample duration "
        "(ffprobe missing or the file is not audio). "
        "Pass a 2–12 s wav or mp3 with --audio. Nothing was queued."
    )


def frame_rms(pcm: np.ndarray, sr: int, frame_s: float = 0.02) -> tuple[np.ndarray, float]:
    """Per-frame RMS and the frame length in seconds."""
    frame = max(1, int(round(float(sr) * frame_s)))
    n = int(len(pcm) // frame)
    if n <= 0:
        return np.zeros(1, dtype=np.float64), frame / float(sr or 1)
    shaped = np.asarray(pcm[: n * frame], dtype=np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(shaped * shaped, axis=1))
    return rms, frame / float(sr)


def best_speech_window(
    pcm: np.ndarray,
    sr: int,
    window_s: float = H3_VOICE_MAX_S,
) -> tuple[float, float]:
    """Loudest, most continuous ``window_s`` of speech. ``(start_s, end_s)``."""
    if sr <= 0 or pcm is None or len(pcm) == 0:
        return 0.0, 0.0
    duration = float(len(pcm)) / float(sr)
    window = float(window_s)
    if duration <= window + 1e-3:
        return 0.0, round(duration, 4)
    rms, frame_s = frame_rms(pcm, sr)
    n = int(len(rms))
    win = max(1, int(round(window / frame_s)))
    if n <= win:
        return 0.0, round(min(duration, window), 4)
    p90 = float(np.percentile(rms, 90))
    thresh = max(1e-4, 0.25 * p90)
    speech = rms >= thresh
    csum = np.cumsum(rms)
    ssum = np.cumsum(speech.astype(np.float64))
    best_i = 0
    best_score = -1.0
    step = max(1, int(round(0.05 / frame_s)))
    last = n - win
    for i in range(0, last + 1, step):
        j = i + win
        energy = float(csum[j - 1] - (csum[i - 1] if i else 0.0))
        speech_frames = float(ssum[j - 1] - (ssum[i - 1] if i else 0.0))
        frac = speech_frames / float(win)
        # Energy picks the loud stretch; speech fraction keeps it continuous.
        score = energy * (0.15 + 0.85 * frac)
        if score > best_score:
            best_score = score
            best_i = i
    start = best_i * frame_s
    end = start + window
    if end > duration:
        end = duration
        start = max(0.0, end - window)
    return round(start, 4), round(end, 4)


def _default_probe(path: str) -> float:
    from master_agent.music.beats import audio_duration

    try:
        return float(audio_duration(path) or 0.0)
    except (OSError, TypeError, ValueError):
        return 0.0


def _default_decode(path: str) -> tuple[np.ndarray, int]:
    from master_agent.music.beats import decode_audio

    return decode_audio(path)


def _default_trim(src: str, dest: str, start_s: float, duration_s: float) -> None:
    from master_agent.video_concat import find_ffmpeg

    ff = find_ffmpeg() or shutil.which("ffmpeg")
    if not ff:
        raise VoiceSampleError(
            "ffmpeg is required to trim an H3 voice sample longer than 12 s. "
            "Install ffmpeg and retry. Nothing was queued."
        )
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{float(start_s):.3f}",
        "-t",
        f"{float(duration_s):.3f}",
        "-vn",
        "-ac",
        "1",
        str(dest_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not dest_path.is_file():
        err = proc.stderr.decode("utf-8", errors="replace")[-300:]
        raise VoiceSampleError(
            f"ffmpeg could not trim the H3 voice sample ({err or 'no output'}). "
            "Nothing was queued."
        )


def prepare_h3_voice_sample(
    path: str,
    *,
    probe: Optional[Callable[[str], float]] = None,
    decode: Optional[Callable[[str], tuple[np.ndarray, int]]] = None,
    trim: Optional[Callable[[str, str, float, float], None]] = None,
    out_dir: Optional[str | Path] = None,
    min_s: float = H3_VOICE_MIN_S,
    max_s: float = H3_VOICE_MAX_S,
) -> VoiceSample:
    """Accept 2–12 s, reject shorter, trim longer. Does not queue."""
    src = str(path)
    measure = probe or _default_probe
    try:
        original = float(measure(src) or 0.0)
    except (OSError, TypeError, ValueError):
        original = 0.0
    if original <= 0:
        raise VoiceSampleError(_unreadable_message())
    if original + 1e-3 < float(min_s):
        raise VoiceSampleError(_too_short_message(original))
    if original <= float(max_s) + 1e-3:
        return VoiceSample(
            path=src,
            original_path=src,
            original_duration_s=original,
            start_s=0.0,
            end_s=round(original, 4),
            method=PASSTHROUGH_METHOD,
            trimmed=False,
        )
    decoder = decode or _default_decode
    try:
        pcm, sr = decoder(src)
    except Exception as exc:
        raise VoiceSampleError(
            f"Could not decode the H3 voice sample ({exc}). Nothing was queued."
        ) from exc
    pcm = np.asarray(pcm)
    if pcm.size == 0 or int(sr) <= 0:
        raise VoiceSampleError(
            "H3 voice sample decoded to empty audio. Nothing was queued."
        )
    start, end = best_speech_window(pcm, int(sr), window_s=float(max_s))
    if end - start > float(max_s) + 0.05:
        end = round(start + float(max_s), 4)
    dest_dir = Path(out_dir) if out_dir else Path(src).parent
    dest = dest_dir / f"{Path(src).stem}.h3voice.wav"
    cutter = trim or _default_trim
    cutter(src, str(dest), start, float(max_s))
    return VoiceSample(
        path=str(dest),
        original_path=src,
        original_duration_s=original,
        start_s=start,
        end_s=round(start + float(max_s), 4),
        method=TRIM_METHOD,
        trimmed=True,
    )


def line_from_brief(text: str | None) -> str:
    """Pull a word-for-word clause already written into the brief, if any."""
    import re

    raw = text or ""
    match = re.search(
        r"saying word for word:\s*(['\"])(.+?)\1",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(2).strip()
    return ""


def optional_local_transcript(audio_path: str | None) -> Optional[str]:
    """Transcribe only when a local model file is already on disk.

    Importing a package is not enough: named Whisper checkpoints download
    weights. With no ``H3_LOCAL_TRANSCRIBER_MODEL`` file, this returns None
    and does not import or fetch anything.
    """
    model = os.environ.get("H3_LOCAL_TRANSCRIBER_MODEL", "").strip()
    if not audio_path or not model or not Path(model).is_file():
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        WhisperModel = None  # type: ignore[assignment]
    if WhisperModel is not None:
        try:
            loaded = WhisperModel(model, device="cpu", compute_type="int8")
            segments, _info = loaded.transcribe(str(audio_path))
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text or None
        except Exception:
            return None
    try:
        import whisper
    except ImportError:
        return None
    try:
        loaded = whisper.load_model(model)
        result = loaded.transcribe(str(audio_path))
        text = str((result or {}).get("text") or "").strip()
        return text or None
    except Exception:
        return None


def resolve_spoken_line(
    line: str | None,
    brief: str | None = None,
    audio_path: str | None = None,
) -> str:
    """Explicit line, else a clause already in the brief, else a local transcript."""
    explicit = (line or "").strip()
    if explicit:
        return explicit
    found = line_from_brief(brief)
    if found:
        return found
    heard = optional_local_transcript(audio_path)
    return (heard or "").strip()


def _quote_line(line: str) -> str:
    """Wrap without changing the line. The line itself stays an exact substring."""
    if "'" not in line:
        return f"'{line}'"
    if '"' not in line:
        return f'"{line}"'
    return line


def inject_spoken_line(prompt: str, line: str) -> str:
    """Append ``saying word for word: '…'`` so the line is an exact substring."""
    text = (line or "").strip()
    base = (prompt or "").strip()
    if not text:
        return base
    if text in base and _SPOKEN_MARK in base.lower():
        return base
    clause = f"{_SPOKEN_MARK} {_quote_line(text)}"
    if not base:
        return clause[0].upper() + clause[1:]
    if base.endswith((".", "!", "?")):
        return f"{base} {clause[0].upper()}{clause[1:]}"
    return f"{base}, {clause}"


def h3_missing_line_warning(line: str | None, *, h3_voice: bool) -> Optional[str]:
    if not h3_voice:
        return None
    if (line or "").strip():
        return None
    return H3_MISSING_LINE_WARNING


@dataclass
class H3VoicePreflight:
    audio_path: Optional[str]
    voice_sample: Optional[dict[str, Any]]
    spoken_line: str
    line_warning: Optional[str]
    trimmed_note: Optional[str] = None


def h3_voice_preflight(
    *,
    request: str,
    variant: str | None,
    audio_path: str | None,
    has_image: bool,
    has_video: bool,
    line: str | None = None,
    probe: Optional[Callable[[str], float]] = None,
    decode: Optional[Callable[[str], tuple[np.ndarray, int]]] = None,
    trim: Optional[Callable[[str, str, float, float], None]] = None,
) -> H3VoicePreflight:
    """Length-gate and resolve the line for an H3 voice route. No queue."""
    from master_agent.orchestrator.talking import is_h3_voice_route

    h3 = is_h3_voice_route(
        request,
        variant=variant,
        has_image=has_image,
        has_audio=bool(audio_path),
        has_video=has_video,
    )
    if not h3 or not audio_path:
        spoken = (line or "").strip()
        return H3VoicePreflight(
            audio_path=audio_path,
            voice_sample=None,
            spoken_line=spoken,
            line_warning=None,
        )
    sample = prepare_h3_voice_sample(
        audio_path, probe=probe, decode=decode, trim=trim
    )
    spoken = resolve_spoken_line(line, request, sample.path)
    note = None
    if sample.trimmed:
        note = (
            f"H3 voice sample is {sample.original_duration_s:.1f}s; "
            f"using {sample.end_s - sample.start_s:.1f}s "
            f"({sample.start_s:.2f}–{sample.end_s:.2f}s, {sample.method})"
        )
    return H3VoicePreflight(
        audio_path=sample.path,
        voice_sample=sample.provenance(),
        spoken_line=spoken,
        line_warning=h3_missing_line_warning(spoken, h3_voice=True),
        trimmed_note=note,
    )
