"""Lyrics + tags → wav via heartlib ``HeartMuLaGenPipeline``.

HeartCodec is not a separate Buddy call. ``from_pretrained`` loads
``HeartMuLa-oss-3B`` and ``HeartCodec-oss`` together. Official dtypes are
bf16 for HeartMuLa and fp32 for HeartCodec. ``lazy_load`` unloads one
before the other (heartlib's single-GPU path). There is no nf4/fp4 argument.

``__call__`` has no seed parameter. When ``seed`` is set, Buddy calls
``torch.manual_seed`` before the pipeline. Reconstruction
(``run_music_reconstruction.py``) is not this path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.heartmula.config import (
    HeartMuLaConfigError,
    codec_device_name,
    codec_dtype_name,
    device_name,
    lazy_load,
    models_dir,
    mula_dtype_name,
    provenance_block,
    version,
)

log = logging.getLogger(__name__)

_TORCH_DTYPES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


class HeartMuLaUnavailable(RuntimeError):
    """heartlib is not installed. Importing master_agent must not raise this."""


class MissingHeartMuLaWeights(RuntimeError):
    """Attested slots are absent. Listing them does not download."""


@dataclass
class GenerateResult:
    wav: Path
    provenance: dict[str, Any]
    plan: dict[str, Any]


def load_heartlib() -> tuple[Any, Any]:
    """Import heartlib pipelines. Clear error when the package is missing."""
    try:
        from heartlib import HeartMuLaGenPipeline, HeartTranscriptorPipeline
    except ImportError as exc:
        raise HeartMuLaUnavailable(
            "heartlib is not installed. Buddy does not install it. "
            "Clone https://github.com/HeartMuLa/heartlib (Apache-2.0) and "
            "`pip install -e .` in that checkout. "
            f"Import error: {exc}"
        ) from exc
    return HeartMuLaGenPipeline, HeartTranscriptorPipeline


def _duration_ms(duration_s: float) -> int:
    if duration_s <= 0:
        raise HeartMuLaConfigError("duration must be > 0")
    return int(round(float(duration_s) * 1000))


def plan_generate(
    *,
    lyrics: str,
    tags: str,
    out: str | Path,
    duration_s: float = 30.0,
    seed: int | None = None,
    topk: int = 50,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate args and describe the heartlib call. No package, GPU, or download."""
    lyrics_text = (lyrics or "").strip()
    tags_text = (tags or "").strip()
    if not lyrics_text:
        raise HeartMuLaConfigError("lyrics are required")
    if not tags_text:
        raise HeartMuLaConfigError("tags are required")
    if topk < 1:
        raise HeartMuLaConfigError("topk must be >= 1")
    if temperature <= 0:
        raise HeartMuLaConfigError("temperature must be > 0")
    ms = _duration_ms(duration_s)
    note = None
    if ", " in tags_text:
        note = "heartlib tags are comma-separated without spaces (piano,happy,pop)"
    plan = provenance_block(
        lyrics=lyrics_text,
        tags=tags_text,
        seed=seed,
        wav=str(out),
        max_audio_length_ms=ms,
        dry_run=dry_run,
    )
    plan.update(
        {
            "topk": int(topk),
            "temperature": float(temperature),
            "cfg_scale": float(cfg_scale),
            "model_path": str(models_dir()),
            "version": version(),
            "note": note,
            "call": (
                "HeartMuLaGenPipeline.from_pretrained(model_path, device, dtype, "
                "version, lazy_load); pipe({lyrics, tags}, max_audio_length_ms, "
                "save_path, topk, temperature, cfg_scale)"
            ),
        }
    )
    return plan


def format_plan(plan: dict[str, Any]) -> str:
    lines = [
        "heartmula generate",
        f"  model_path: {plan.get('model_path')}",
        f"  mula: {plan.get('mula_repo')} ({plan.get('mula_dtype')})",
        f"  codec: {plan.get('codec_repo')} ({plan.get('codec_dtype')})",
        f"  version: {plan.get('version')}  lazy_load: {plan.get('lazy_load')}",
        f"  device: {plan.get('device')}  codec_device: {plan.get('codec_device')}",
        f"  max_audio_length_ms: {plan.get('max_audio_length_ms')}",
        f"  topk: {plan.get('topk')}  temperature: {plan.get('temperature')}  "
        f"cfg_scale: {plan.get('cfg_scale')}",
        f"  seed: {plan.get('seed')}",
        f"  out: {plan.get('wav')}",
        f"  tags: {plan.get('tags')}",
    ]
    lyrics = str(plan.get("lyrics") or "")
    preview = lyrics if len(lyrics) <= 180 else lyrics[:177] + "..."
    lines.append(f"  lyrics: {preview}")
    if plan.get("note"):
        lines.append(f"  note: {plan['note']}")
    if plan.get("dry_run"):
        lines.append("  dry-run: plan only (no heartlib, no GPU, no wav)")
    return "\n".join(lines)


def _torch_dtype(name: str):
    import torch

    attr = _TORCH_DTYPES[name]
    return getattr(torch, attr)


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def generate_track(
    *,
    lyrics: str,
    tags: str,
    out: str | Path,
    duration_s: float = 30.0,
    seed: int | None = None,
    topk: int = 50,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    pipeline_cls: Any = None,
    require_weights: bool = True,
) -> GenerateResult:
    """Run HeartMuLaGenPipeline. ``pipeline_cls`` is for tests."""
    plan = plan_generate(
        lyrics=lyrics,
        tags=tags,
        out=out,
        duration_s=duration_s,
        seed=seed,
        topk=topk,
        temperature=temperature,
        cfg_scale=cfg_scale,
        dry_run=False,
    )
    if require_weights:
        from master_agent.heartmula.doctor import assert_generate_weights

        assert_generate_weights()
    if pipeline_cls is None:
        pipeline_cls, _transcriptor = load_heartlib()
        import torch

        pipe = pipeline_cls.from_pretrained(
            str(models_dir()),
            device={
                "mula": torch.device(device_name()),
                "codec": torch.device(codec_device_name()),
            },
            dtype={
                "mula": _torch_dtype(mula_dtype_name()),
                "codec": _torch_dtype(codec_dtype_name()),
            },
            version=version(),
            lazy_load=lazy_load(),
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            str(models_dir()),
            device={"mula": device_name(), "codec": codec_device_name()},
            dtype={"mula": mula_dtype_name(), "codec": codec_dtype_name()},
            version=version(),
            lazy_load=lazy_load(),
        )
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _apply_seed(seed)
    log.info("heartmula generate → %s", dest)
    pipe(
        {"lyrics": lyrics, "tags": tags},
        max_audio_length_ms=plan["max_audio_length_ms"],
        save_path=str(dest),
        topk=int(topk),
        temperature=float(temperature),
        cfg_scale=float(cfg_scale),
    )
    provenance = provenance_block(
        lyrics=(lyrics or "").strip(),
        tags=(tags or "").strip(),
        seed=seed,
        wav=str(dest),
        max_audio_length_ms=plan["max_audio_length_ms"],
        dry_run=False,
    )
    return GenerateResult(wav=dest, provenance=provenance, plan=plan)


def write_silent_wav(path: str | Path, seconds: float = 8.0, rate: int = 22050) -> Path:
    """Dry-run stand-in so beat planning can run without heartlib."""
    import wave

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(float(seconds) * rate), 1)
    with wave.open(str(dest), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    return dest
