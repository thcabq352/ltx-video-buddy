"""Env knobs and attested HeartMuLa weight slots.

Filenames come from the Hugging Face sibling lists read for this pack
(2026-09-30) and from heartlib ``_resolve_paths`` / ``HeartTranscriptorPipeline``.
Do not add shard names that were not on those listings.

heartlib layout (what ``HeartMuLaGenPipeline.from_pretrained`` opens):

    {HEARTMULA_MODELS_DIR}/
      tokenizer.json
      gen_config.json
      HeartMuLa-oss-3B/          # chosen 3B checkpoint, any of the repos below
      HeartCodec-oss/            # HeartCodec-oss-20260123 contents
      HeartTranscriptor-oss/

The ComfyUI custom node uses different folder names. See docs/HEARTMULA.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Official heartlib README pairs this checkpoint with HeartCodec-oss-20260123
# and downloads it into the folder name HeartMuLa-oss-3B.
DEFAULT_MULA_REPO = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"
DEFAULT_CODEC_REPO = "HeartMuLa/HeartCodec-oss-20260123"
DEFAULT_TRANSCRIPTOR_REPO = "HeartMuLa/HeartTranscriptor-oss"
DEFAULT_GEN_REPO = "HeartMuLa/HeartMuLaGen"

# Same four-shard layout on the Hub (config + index + model-0000N-of-00004).
ATTESTED_MULA_REPOS = frozenset(
    {
        "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
        "HeartMuLa/HeartMuLa-oss-3B",
        "HeartMuLa/HeartMuLa-RL-oss-3B-20260123",
    }
)
# HeartMuLa/HeartCodec-oss is named by heartlib's older instructions and by
# the Comfy node README. Its Hub sibling list was not retrieved (API 401),
# so this pack does not guess that repo's filenames.
ATTESTED_CODEC_REPOS = frozenset({"HeartMuLa/HeartCodec-oss-20260123"})

MULA_DIR = "HeartMuLa-oss-3B"
CODEC_DIR = "HeartCodec-oss"
TRANSCRIPTOR_DIR = "HeartTranscriptor-oss"
VERSION = "3B"

_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
}

MULA_SHARDS = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)
CODEC_SHARDS = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
# HeartMuLa/HeartTranscriptor-oss siblings, excluding README / .gitattributes.
TRANSCRIPTOR_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class HeartMuLaConfigError(ValueError):
    """Env or repo id this pack will not guess."""


@dataclass(frozen=True)
class WeightSlot:
    """One attested file. ``dest_rel`` is relative to ``models_dir()``."""

    group: str
    repo_id: str
    filename: str
    dest_rel: str

    @property
    def label(self) -> str:
        return f"{self.repo_id} {self.filename}"


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def models_dir() -> Path:
    raw = (os.getenv("HEARTMULA_MODELS_DIR") or "").strip()
    if raw:
        return Path(raw)
    from master_agent.config import MODELS_DIR

    return Path(MODELS_DIR) / "heartmula"


def mula_repo() -> str:
    return (os.getenv("HEARTMULA_MULA_REPO") or DEFAULT_MULA_REPO).strip()


def codec_repo() -> str:
    return (os.getenv("HEARTMULA_CODEC_REPO") or DEFAULT_CODEC_REPO).strip()


def transcriptor_repo() -> str:
    return (os.getenv("HEARTMULA_TRANSCRIPTOR_REPO") or DEFAULT_TRANSCRIPTOR_REPO).strip()


def gen_repo() -> str:
    return (os.getenv("HEARTMULA_GEN_REPO") or DEFAULT_GEN_REPO).strip()


def version() -> str:
    return (os.getenv("HEARTMULA_VERSION") or VERSION).strip() or VERSION


def _dtype(name: str, default: str) -> str:
    raw = (os.getenv(name) or default).strip().lower()
    if raw not in _DTYPE_ALIASES:
        raise HeartMuLaConfigError(
            f"{name}={raw} is not a heartlib dtype. "
            "HeartMuLaGenPipeline accepts bf16, fp16, or fp32 "
            "(example script: mula bfloat16, codec float32). "
            "heartlib does not document nf4 or fp4. "
            "On a 16GB card keep bf16 and HEARTMULA_LAZY_LOAD=1."
        )
    return _DTYPE_ALIASES[raw]


def mula_dtype_name() -> str:
    return _dtype("HEARTMULA_DTYPE", "bf16")


def codec_dtype_name() -> str:
    return _dtype("HEARTMULA_CODEC_DTYPE", "fp32")


def device_name() -> str:
    return (os.getenv("HEARTMULA_DEVICE") or "cuda").strip() or "cuda"


def codec_device_name() -> str:
    raw = (os.getenv("HEARTMULA_CODEC_DEVICE") or "").strip()
    return raw or device_name()


def lazy_load() -> bool:
    """heartlib single-GPU advice: load HeartMuLa, unload, then HeartCodec."""
    raw = os.getenv("HEARTMULA_LAZY_LOAD")
    if raw is not None and str(raw).strip():
        return _flag("HEARTMULA_LAZY_LOAD", True)
    from master_agent.config import VRAM_GB

    return float(VRAM_GB) <= 16.0


def default_duration_s() -> float:
    raw = (os.getenv("HEARTMULA_DURATION_S") or "30").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise HeartMuLaConfigError(f"HEARTMULA_DURATION_S={raw} is not a number") from exc
    if value <= 0:
        raise HeartMuLaConfigError("HEARTMULA_DURATION_S must be > 0")
    return value


def require_known_repos() -> None:
    """Refuse to invent shard names for a repo this pack has not attested."""
    mula = mula_repo()
    if mula not in ATTESTED_MULA_REPOS:
        known = ", ".join(sorted(ATTESTED_MULA_REPOS))
        raise HeartMuLaConfigError(
            f"HEARTMULA_MULA_REPO={mula} has no attested file list in this pack. "
            f"Use one of: {known}"
        )
    codec = codec_repo()
    if codec not in ATTESTED_CODEC_REPOS:
        raise HeartMuLaConfigError(
            f"HEARTMULA_CODEC_REPO={codec} has no attested file list in this pack. "
            "HeartMuLa/HeartCodec-oss is a real repo id, but its Hub filenames "
            "were not retrieved, so Buddy will not guess them. "
            "Use HeartMuLa/HeartCodec-oss-20260123 (heartlib downloads that "
            "snapshot into the HeartCodec-oss folder)."
        )
    gen = gen_repo()
    if gen != DEFAULT_GEN_REPO:
        raise HeartMuLaConfigError(
            f"HEARTMULA_GEN_REPO={gen} is not HeartMuLa/HeartMuLaGen "
            "(tokenizer.json and gen_config.json)."
        )
    tr = transcriptor_repo()
    if tr != DEFAULT_TRANSCRIPTOR_REPO:
        raise HeartMuLaConfigError(
            f"HEARTMULA_TRANSCRIPTOR_REPO={tr} is not {DEFAULT_TRANSCRIPTOR_REPO}."
        )
    if version() != "3B":
        raise HeartMuLaConfigError(
            "HEARTMULA_VERSION must be 3B. heartlib looks for HeartMuLa-oss-3B; "
            "the 7B checkpoint is not released."
        )


def _slots_for(mula: str, codec: str, gen: str, transcriptor: str) -> list[WeightSlot]:
    rows: list[WeightSlot] = [
        WeightSlot("generate", gen, "tokenizer.json", "tokenizer.json"),
        WeightSlot("generate", gen, "gen_config.json", "gen_config.json"),
    ]
    for name in MULA_SHARDS:
        rows.append(WeightSlot("generate", mula, name, f"{MULA_DIR}/{name}"))
    for name in CODEC_SHARDS:
        rows.append(WeightSlot("generate", codec, name, f"{CODEC_DIR}/{name}"))
    for name in TRANSCRIPTOR_FILES:
        rows.append(
            WeightSlot("transcribe", transcriptor, name, f"{TRANSCRIPTOR_DIR}/{name}")
        )
    return rows


def all_slots() -> list[WeightSlot]:
    require_known_repos()
    return _slots_for(mula_repo(), codec_repo(), gen_repo(), transcriptor_repo())


def slots_for(group: str) -> list[WeightSlot]:
    return [slot for slot in all_slots() if slot.group == group]


def provenance_block(
    *,
    lyrics: str | None = None,
    tags: str | None = None,
    seed: int | None = None,
    wav: str | None = None,
    transcribe_source: str | None = None,
    words_path: str | None = None,
    max_audio_length_ms: int | None = None,
    dry_run: bool = False,
    placeholder: str | None = None,
) -> dict:
    """Optional ``params.heartmula`` / run-record object. Same schema id elsewhere."""
    return {
        "lyrics": lyrics,
        "tags": tags,
        "mula_repo": mula_repo(),
        "codec_repo": codec_repo(),
        "transcriptor_repo": transcriptor_repo(),
        "gen_repo": gen_repo(),
        "version": version(),
        "mula_dtype": mula_dtype_name(),
        "codec_dtype": codec_dtype_name(),
        "lazy_load": lazy_load(),
        "device": device_name(),
        "codec_device": codec_device_name(),
        "seed": seed,
        "wav": wav,
        "transcribe_source": transcribe_source,
        "words_path": words_path,
        "max_audio_length_ms": max_audio_length_ms,
        "dry_run": bool(dry_run),
        "placeholder": placeholder,
    }
