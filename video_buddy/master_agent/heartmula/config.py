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


# heartlib ``llama3_2_3B()`` builds the backbone at this length. RoPE is
# allocated then. Buddy can shrink the KV window before ``setup_caches``;
# it cannot grow past this without rebuilding the backbone.
HEARTLIB_BACKBONE_MAX_SEQ_LEN = 8192
# ``HeartMuLaGenPipeline._forward``: ``max_audio_frames = max_audio_length_ms // 80``.
AUDIO_FRAME_MS = 80
# Passed a 5s bf16 + lazy_load smoke on 16GB. The floor of the VRAM cap,
# not the window Buddy picks for every short clip.
SMOKE_MAX_SEQ_LEN = 512
# torchtune 0.4 (heartlib's pin) allocates ``KVCache`` at ``num_heads``
# because the GQA expand has already copied K and V out to query heads.
# 3B backbone: 28 layers, 24 heads, head_dim 128, bf16. Per token, per batch.
KV_BYTES_PER_TOKEN = 28 * 2 * 24 * 128 * 2
# Leave this much of the card for bf16 weights and activations. The 8192
# window is about 5.3GB at cfg batch 2 and OOM'd on 16GB during that expand.
# 16GB then keeps about 2GB for the cache (~3072 tokens). Around 20GB the
# same reserve still fits the full 8192 window, so longer songs stay intact.
KV_WEIGHT_RESERVE_GB = 14.0
LOW_VRAM_GB = 16.0
_SEQ_ALIGN = 128


@dataclass(frozen=True)
class SeqLenChoice:
    """Backbone KV window for one generate call."""

    max_seq_len: int
    needed: int
    cap: int
    source: str
    vram_gb: float
    batch: int
    note: str | None = None


def current_vram_gb() -> float:
    from master_agent.config import VRAM_GB

    return float(VRAM_GB)


def cfg_batch(cfg_scale: float) -> int:
    """heartlib uses a batch of 2 unless ``cfg_scale`` is exactly 1."""
    return 1 if float(cfg_scale) == 1.0 else 2


def _source_chars(text: str) -> int:
    """Character count, or the file body when heartlib would read a path."""
    raw = text or ""
    if not raw or "\n" in raw or len(raw) > 1024:
        return len(raw)
    try:
        path = Path(raw)
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="replace")
            return min(len(body), 200_000)
    except (OSError, ValueError):
        return len(raw)
    return len(raw)


def prompt_token_budget(lyrics: str, tags: str) -> int:
    """Tokens to reserve before audio frames. No tokenizer import.

    Dry-run must not load heartlib. One token per character over-estimates
    English BPE and matches CJK closely enough that a long lyric does not
    silently walk off the end of the cache.
    """
    chars = _source_chars(lyrics) + _source_chars(tags)
    return max(256, chars + 32)


def tokens_for_run(duration_s: float, lyrics: str, tags: str) -> int:
    """Prompt budget plus one cache slot per 80ms audio frame."""
    ms = int(round(float(duration_s) * 1000))
    frames = max(ms, 0) // AUDIO_FRAME_MS
    return frames + prompt_token_budget(lyrics, tags)


def align_seq_len(tokens: int, step: int = _SEQ_ALIGN) -> int:
    tokens = max(int(tokens), 1)
    step = max(int(step), 1)
    aligned = ((tokens + step - 1) // step) * step
    return min(HEARTLIB_BACKBONE_MAX_SEQ_LEN, aligned)


def seq_len_cap(vram_gb: float, batch: int = 2) -> int:
    """Longest GQA-expanded KV window that stays inside ``vram_gb``."""
    batch = max(int(batch), 1)
    per = KV_BYTES_PER_TOKEN * batch
    budget = int(float(vram_gb) * (1024**3) - KV_WEIGHT_RESERVE_GB * (1024**3))
    floor = SMOKE_MAX_SEQ_LEN * per
    raw = max(budget, floor) // per
    aligned = (int(raw) // _SEQ_ALIGN) * _SEQ_ALIGN
    return min(HEARTLIB_BACKBONE_MAX_SEQ_LEN, max(SMOKE_MAX_SEQ_LEN, aligned))


def _env_max_seq_len() -> int | None:
    raw = (os.getenv("HEARTMULA_MAX_SEQ_LEN") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise HeartMuLaConfigError(
            f"HEARTMULA_MAX_SEQ_LEN={raw} is not an integer"
        ) from exc


def _env_low_vram(explicit: bool | None) -> bool:
    if explicit:
        return True
    return _flag("HEARTMULA_LOW_VRAM", False)


def select_max_seq_len(
    *,
    duration_s: float,
    lyrics: str,
    tags: str,
    cfg_scale: float = 1.5,
    vram_gb: float | None = None,
    low_vram: bool | None = None,
    override: int | None = None,
) -> SeqLenChoice:
    """Pick a backbone window that covers this clip and fits the card.

    Above the 16GB reserve the cap is heartlib's 8192, so a long song is
    unchanged. At 16GB the cap is about 3072 tokens (cfg batch 2): a 30s
    track still fits, a 240s track does not, and the plan says so before
    heartlib is imported. ``override`` (CLI or ``HEARTMULA_MAX_SEQ_LEN``)
    skips the cap. ``low_vram`` plans as if the card is 16GB.
    """
    gb = current_vram_gb() if vram_gb is None else float(vram_gb)
    tight = _env_low_vram(low_vram)
    if tight:
        gb = min(gb, LOW_VRAM_GB)
    batch = cfg_batch(cfg_scale)
    needed = tokens_for_run(duration_s, lyrics, tags)
    cap = seq_len_cap(gb, batch)
    chosen = override if override is not None else _env_max_seq_len()
    if chosen is not None:
        value = int(chosen)
        if value < 1 or value > HEARTLIB_BACKBONE_MAX_SEQ_LEN:
            raise HeartMuLaConfigError(
                f"max_seq_len={value} is outside 1..{HEARTLIB_BACKBONE_MAX_SEQ_LEN}. "
                "heartlib builds the 3B backbone RoPE at 8192; Buddy will not grow it."
            )
        note = None
        if value < needed:
            note = (
                f"max_seq_len {value} is shorter than the ~{needed} tokens this "
                "duration and lyric length need. The KV window can fill before "
                "the song ends. Raise it or shorten --duration."
            )
        return SeqLenChoice(
            max_seq_len=value,
            needed=needed,
            cap=cap,
            source="override",
            vram_gb=gb,
            batch=batch,
            note=note,
        )
    window = align_seq_len(needed)
    if window > cap:
        raise HeartMuLaConfigError(
            f"this run needs about {window} backbone tokens "
            f"({float(duration_s):g}s plus lyrics) but a {gb:g}GB card can hold "
            f"about {cap} under the GQA-expanded KV budget. heartlib's "
            f"{HEARTLIB_BACKBONE_MAX_SEQ_LEN} window OOMs on 16GB during that "
            "expand. Shorten --duration, or set HEARTMULA_MAX_SEQ_LEN if you "
            "know a longer window fits. Keep HEARTMULA_DTYPE=bf16 and "
            "HEARTMULA_LAZY_LOAD=1 on a 16GB card."
        )
    notes: list[str] = []
    if cap < HEARTLIB_BACKBONE_MAX_SEQ_LEN:
        notes.append(
            f"KV budget on {gb:g}GB caps the backbone window at {cap} "
            f"(heartlib default {HEARTLIB_BACKBONE_MAX_SEQ_LEN} OOMs during "
            f"the GQA expand). This run uses {window}."
        )
    if needed > window:
        notes.append(
            f"This clip asks for about {needed} tokens and the backbone window "
            f"is {window}. Generation can stop when the cache fills."
        )
    note = " ".join(notes) if notes else None
    return SeqLenChoice(
        max_seq_len=window,
        needed=needed,
        cap=cap,
        source="low-vram" if tight else "duration",
        vram_gb=gb,
        batch=batch,
        note=note,
    )


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
    max_seq_len: int | None = None,
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
        "max_seq_len": max_seq_len,
        "dry_run": bool(dry_run),
        "placeholder": placeholder,
    }
