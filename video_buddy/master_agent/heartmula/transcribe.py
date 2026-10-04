"""Audio → word timestamps via heartlib ``HeartTranscriptorPipeline``.

The class subclasses transformers ``AutomaticSpeechRecognitionPipeline``.
The official example prints ``pipe(music_path, **whisper_kwargs)`` without
timestamps. Buddy also passes ``return_timestamps="word"`` (Whisper ASR)
and maps chunks to ``{w, s, e}`` for lipdub ``load_words``.

HeartTranscriptor is an alternative to an external faster-whisper JSON file.
It does not replace that path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.heartmula.config import (
    HeartMuLaConfigError,
    codec_dtype_name,
    device_name,
    models_dir,
    provenance_block,
)
from master_agent.heartmula.generate import MissingHeartMuLaWeights, load_heartlib

log = logging.getLogger(__name__)

# Kwargs from examples/run_lyrics_transcription.py, plus word timestamps.
WHISPER_KWARGS: dict[str, Any] = {
    "return_timestamps": "word",
    "max_new_tokens": 256,
    "num_beams": 2,
    "task": "transcribe",
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.8,
    "temperature": (0.0, 0.1, 0.2, 0.4),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.4,
}


@dataclass
class TranscribeResult:
    path: Path
    words: list[dict[str, Any]]
    provenance: dict[str, Any]
    plan: dict[str, Any]


def chunks_to_words(result: Any) -> list[dict[str, Any]]:
    """Map a Whisper ASR result onto ``[{w, s, e}, ...]``."""
    chunks = []
    if isinstance(result, dict):
        chunks = result.get("chunks") or []
    words: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        stamp = chunk.get("timestamp")
        if not text or not isinstance(stamp, (list, tuple)) or not stamp:
            continue
        start = stamp[0]
        end = stamp[1] if len(stamp) > 1 else None
        if start is None:
            continue
        start_s = float(start)
        end_s = float(end) if end is not None else start_s
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        words.append({"w": text, "s": start_s, "e": end_s})
    words.sort(key=lambda row: (row["s"], row["e"]))
    return words


def words_document(words: list[dict[str, Any]]) -> dict[str, Any]:
    """``{\"words\": [...]}`` shape that lipdub ``load_words`` accepts."""
    return {"words": list(words)}


def write_words(path: str | Path, words: list[dict[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(words_document(words), indent=1) + "\n",
        encoding="utf-8",
    )
    return dest


def plan_transcribe(
    *,
    audio: str | Path,
    out: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    audio_text = str(audio or "").strip()
    if not audio_text:
        raise HeartMuLaConfigError("audio is required")
    plan = provenance_block(
        wav=None,
        transcribe_source="hearttranscriptor",
        words_path=str(out),
        dry_run=dry_run,
    )
    plan.update(
        {
            "audio": audio_text,
            "model_path": str(models_dir()),
            "transcriptor_dtype": "fp16",
            "call": (
                "HeartTranscriptorPipeline.from_pretrained(model_path, device, dtype); "
                "pipe(audio, return_timestamps='word', ...official whisper kwargs)"
            ),
            "note": (
                "Official README: HeartTranscriptor is trained on separated vocals. "
                "Demucs (or similar) before transcribe is recommended; "
                "Buddy does not run separation."
            ),
        }
    )
    return plan


def format_transcribe_plan(plan: dict[str, Any]) -> str:
    lines = [
        "heartmula transcribe",
        f"  audio: {plan.get('audio')}",
        f"  model_path: {plan.get('model_path')}",
        f"  transcriptor: {plan.get('transcriptor_repo')}",
        f"  out: {plan.get('words_path')}",
        "  shape: {\"words\": [{\"w\", \"s\", \"e\"}]}",
    ]
    if plan.get("note"):
        lines.append(f"  note: {plan['note']}")
    if plan.get("dry_run"):
        lines.append("  dry-run: plan only (no heartlib, no GPU, no words file)")
    return "\n".join(lines)


def _transcribe_dtype():
    """Official example uses torch.float16."""
    import torch

    return torch.float16


def transcribe_audio(
    *,
    audio: str | Path,
    out: str | Path,
    pipeline_cls: Any = None,
    require_weights: bool = True,
) -> TranscribeResult:
    plan = plan_transcribe(audio=audio, out=out, dry_run=False)
    audio_path = Path(plan["audio"])
    if not audio_path.is_file():
        raise HeartMuLaConfigError(f"audio file not found: {audio_path}")
    if require_weights:
        from master_agent.heartmula.doctor import assert_transcribe_weights

        assert_transcribe_weights()
    if pipeline_cls is None:
        _gen, pipeline_cls = load_heartlib()
        import torch

        pipe = pipeline_cls.from_pretrained(
            str(models_dir()),
            device=torch.device(device_name()),
            dtype=_transcribe_dtype(),
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            str(models_dir()),
            device=device_name(),
            dtype="fp16",
        )
    log.info("heartmula transcribe %s", audio_path)
    result = pipe(str(audio_path), **WHISPER_KWARGS)
    words = chunks_to_words(result)
    dest = write_words(out, words)
    provenance = provenance_block(
        transcribe_source="hearttranscriptor",
        words_path=str(dest),
        dry_run=False,
    )
    provenance["audio"] = str(audio_path)
    # codec dtype is unused here; keep the block shape stable.
    provenance["codec_dtype"] = codec_dtype_name()
    return TranscribeResult(path=dest, words=words, provenance=provenance, plan=plan)


def transcribe_requested(flag: bool) -> bool:
    import os

    if flag:
        return True
    return os.getenv("HEARTMULA_TRANSCRIBE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
