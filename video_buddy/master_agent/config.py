"""Project paths, ComfyUI settings, and RTX 5060 Ti 16GB VRAM caps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Project root = parent of master_agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Optional override
if os.getenv("LTX_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["LTX_PROJECT_ROOT"]).resolve()

COMFYUI_ROOT = Path(
    os.getenv(
        "COMFYUI_ROOT",
        str(PROJECT_ROOT / "ComfyUI_windows_portable" / "ComfyUI"),
    )
).resolve()

PORTABLE_ROOT = Path(
    os.getenv(
        "COMFYUI_PORTABLE_ROOT",
        str(PROJECT_ROOT / "ComfyUI_windows_portable"),
    )
).resolve()

MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models"))).resolve()
WORKFLOWS_DIR = Path(
    os.getenv("WORKFLOWS_DIR", str(PROJECT_ROOT / "workflows"))
).resolve()
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))).resolve()
STATE_DIR = Path(os.getenv("STATE_DIR", str(PROJECT_ROOT / "state"))).resolve()

# Cached ComfyUI /object_info for offline validation
OBJECT_INFO_CACHE = STATE_DIR / "object_info.json"
# Model inventory output
MODEL_INVENTORY_JSON = STATE_DIR / "model_inventory.json"

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_OUTPUT_DIR = Path(
    os.getenv("COMFYUI_OUTPUT_DIR", str(COMFYUI_ROOT / "output"))
).resolve()

# SpaceXAI
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
SPACEXAI_MODEL = os.getenv("SPACEXAI_MODEL", "grok-4.5")
XAI_BASE_URL = "https://api.x.ai/v1"

# Local Ollama — MAIN LLM (user pivot: local-only first).
# Local text + vision: Qwen3-VL Heretic (9B-class, already on this machine)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl-heretic")

# LLM provider selection: auto (ollama -> grok) | ollama[:model] | grok
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "auto") or "auto").strip().lower()

# Storyboard LLM panel: preset (default|local | grok | grok+local|both |
# grok+claude | duo) or comma list (ollama[:model], grok, claude, …)
# default/local = local VL heretic; grok = solo; grok+local / grok+claude = panels
LLM_PANEL = (os.getenv("LLM_PANEL", "default") or "default").strip()
PANEL_JUDGE = (os.getenv("PANEL_JUDGE", "ollama") or "ollama").strip()
PANEL_MEMBER_TIMEOUT_S = int(os.getenv("PANEL_MEMBER_TIMEOUT_S", "300"))

# Claude — optional panel member (account currently has no credits)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

# Judge loop (balanced defaults, same tuning as LTX Project agent v2)
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "1").lower() in ("1", "true", "yes", "on")
MAX_JUDGE_ROUNDS = int(os.getenv("MAX_JUDGE_ROUNDS", "3"))
JUDGE_SCORE_THRESHOLD = float(os.getenv("JUDGE_SCORE_THRESHOLD", "0.78"))
JUDGE_STRICTNESS = float(os.getenv("JUDGE_STRICTNESS", "0.5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.3"))
COST_VRAM_THRESHOLD_GB = float(os.getenv("COST_VRAM_THRESHOLD_GB", "14.5"))
# Project render budget: cumulative VRAM-minutes (vram_gb * time_s / 60)
RENDER_BUDGET_CAP_VRAM_MIN = float(os.getenv("RENDER_BUDGET_CAP_VRAM_MIN", "80"))
RENDER_BUDGET_USED_VRAM_MIN = float(os.getenv("RENDER_BUDGET_USED_VRAM_MIN", "0"))
JUDGE_HEURISTIC_WEIGHT = float(os.getenv("JUDGE_HEURISTIC_WEIGHT", "0.45"))
JUDGE_LLM_WEIGHT = float(os.getenv("JUDGE_LLM_WEIGHT", "0.55"))

# Vision evaluator — same local VL as the text path (qwen3-vl-heretic)
VISION_ENABLED = os.getenv("VISION_ENABLED", "1").lower() in ("1", "true", "yes", "on")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl-heretic")
VISION_FRAMES = int(os.getenv("VISION_FRAMES", "4"))
VISION_TIMEOUT_S = int(os.getenv("VISION_TIMEOUT_S", "900"))
JUDGE_VISION_WEIGHT = float(os.getenv("JUDGE_VISION_WEIGHT", "0.30"))

# Director brain — LLM variant routing (falls back to keyword rules)
DIRECTOR_LLM = os.getenv("DIRECTOR_LLM", "1").lower() in ("1", "true", "yes", "on")

# Power mode — LLM proposes ComfyUI graph ops (set_widget/rewire/…) after
# heuristic patch, grounded in /object_info + workflow RAG. Validate-gated.
# Default off: enable with POWER_MODE=1 or --power-mode on `run` / CLI.
POWER_MODE = os.getenv("POWER_MODE", "0").lower() in ("1", "true", "yes", "on")
POWER_MODE_MAX_OPS = int(os.getenv("POWER_MODE_MAX_OPS", "12"))
POWER_MODE_REPAIR = os.getenv("POWER_MODE_REPAIR", "1").lower() in ("1", "true", "yes", "on")

# Knowledge base — ChromaDB RAG over workflows + run records
KB_ENABLED = os.getenv("KB_ENABLED", "1").lower() in ("1", "true", "yes", "on")
KB_EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "nomic-embed-text")
KB_RECALL_K = int(os.getenv("KB_RECALL_K", "3"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(STATE_DIR / "chroma"))).resolve()
# Outer full-video judge (multi-segment pipeline)
MAX_FULL_JUDGE_ROUNDS = int(os.getenv("MAX_FULL_JUDGE_ROUNDS", "2"))

# Storyboard: smart | always | multi_only | off
STORYBOARD_MODE = (os.getenv("STORYBOARD_MODE", "smart") or "smart").strip().lower()

# Per-run JSON records (orchestrator output; seed of the knowledge-base log)
RUNS_DIR = STATE_DIR / "runs"

# VRAM profile — RTX 5060 Ti 16GB defaults
VRAM_GB = float(os.getenv("VRAM_GB", "16"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "768"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "512"))
# Total requested length (multi-segment stitches clips up to this)
MAX_DURATION_S = float(os.getenv("MAX_DURATION_S", "30"))
# Per-clip cap on 16GB (single Comfy job); longer asks are split into segments
SEGMENT_MAX_S = float(os.getenv("SEGMENT_MAX_S", "6"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "12"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "1.0"))
DEFAULT_FPS = 24
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "balanced")  # draft | balanced | quality
# LTX frame law: valid counts are 8n+1 with a hard minimum of 9 (never 8, never 121-by-default).
DEFAULT_FRAMES = 9
# Diagnose / draft hull: short fire to measure sec/step before any scale.
DIAGNOSE_FRAMES = 9
DIAGNOSE_STEPS_MIN = 6
DIAGNOSE_STEPS_MAX = 8
DIAGNOSE_STEPS = 8
DIAGNOSE_SEED = 20260911
DIAGNOSE_WIDTH = 768
DIAGNOSE_HEIGHT = 512

# Per-variant generation profiles: fps + frame-count snapping (LTX=8n+1, Wan=4n+1)
VARIANT_GEN: dict[str, dict[str, int]] = {
    "wan22": {"fps": 16, "frame_snap": 4},
}
_DEFAULT_GEN = {"fps": DEFAULT_FPS, "frame_snap": 8}


def get_variant_gen(variant: str | None) -> dict[str, int]:
    return dict(VARIANT_GEN.get(variant or "") or _DEFAULT_GEN)

# Quality profiles (steps / max segment / max resolution)
QUALITY_PROFILES: dict[str, dict] = {
    "draft": {
        "steps": 8,
        "frames": 9,
        "segment_max_s": 5.0,
        "max_width": 640,
        "max_height": 384,
        "max_total_s": 15.0,
    },
    "balanced": {
        "steps": 12,
        "segment_max_s": 6.0,
        "max_width": 768,
        "max_height": 512,
        "max_total_s": 30.0,
    },
    "quality": {
        "steps": 15,
        "segment_max_s": 5.0,
        "max_width": 768,
        "max_height": 512,
        "max_total_s": 30.0,
    },
    # Flux stills (CCC sheets + LoRA validation): full 1024x1024, no video split
    "flux": {
        "steps": 20,
        "segment_max_s": 6.0,
        "max_width": 1024,
        "max_height": 1024,
        "max_total_s": 30.0,
    },
}

# OOM downscale ladder: (width, height, frames). First rungs are Rainey-safe
# 8n+1 minima (9/17/25/…) — never open with a 121-frame burn.
DOWNSCALE_LADDER: list[tuple[int, int, int]] = [
    (768, 512, 9),
    (640, 384, 17),
    (512, 384, 25),
    (512, 320, 33),
]

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "2"))
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "1800"))
COMFYUI_VRAM_FLAG = os.getenv("COMFYUI_VRAM_FLAG", "--normalvram")

# Expected model filenames (16GB tier) — used for preflight checks
# CheckpointLoaderSimple needs an all-in-one .safetensors. On this 16GB install we
# use the community EROS baked all-in-one until official Lightricks
# ltx-2.3-22b-dev.safetensors is placed in models/checkpoints/.
DEFAULT_ALL_IN_ONE_CKPT = (
    "LTX2.3_DISTILLED-1.1_BAKED_LTX_10Eros_v14_r768.safetensors"
)
# Optional upgrade target (full quality base, ~40GB+)
OFFICIAL_DEV_CKPT = "ltx-2.3-22b-dev.safetensors"

MODEL_FILES: dict[str, dict[str, str]] = {
    "base": {
        "checkpoint": DEFAULT_ALL_IN_ONE_CKPT,
        "diffusion": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors",
        "lora": "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
        "vae": "taeltx2_3.safetensors",
        "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    },
    "eros": {
        "checkpoint": DEFAULT_ALL_IN_ONE_CKPT,
        "vae": "taeltx2_3.safetensors",
        "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    },
    "directors": {
        "checkpoint": DEFAULT_ALL_IN_ONE_CKPT,
        "diffusion": "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
        "lora": "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
        "vae": "taeltx2_3.safetensors",
        "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    },
    "lipsync": {
        # Dev FP8 all-in-one (contains audio VAE + text projection); the official
        # ltx-2.3-22b-dev.safetensors is the optional upgrade when downloaded.
        "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
        "diffusion": "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
        "lora": "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
        "vae": "taeltx2_3.safetensors",
        "audio_vae": "LTX23_audio_vae_bf16.safetensors",
        "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    },
    "flux": {
        # CCC character sheets + Flux LoRA validation renders (fp8, 16GB-friendly)
        "diffusion": "flux1-dev-fp8.safetensors",
        "clip_l": "clip_l.safetensors",
        "t5xxl": "t5xxl_fp8_e4m3fn.safetensors",
        "vae": "ae.safetensors",
    },
    "wan22": {
        # Wan 2.2 two-stage T2V (Mickmumpitz graph): separate high/low-noise
        # UNETs, so no "checkpoint" key — the patcher must not spray an LTX
        # ckpt onto these UNETLoaders.
        "checkpoint_high": "wan\\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "checkpoint_low": "wan\\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
    },
    "krea2_img": {
        # Krea-2 turbo image graph: no "checkpoint" key (single UNET keeps the
        # template's unet_name; must not get an LTX ckpt sprayed onto it).
        "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
    },
}

# Soft requirements: missing these warn but do not fail preflight hard-count alone
MODEL_OPTIONAL_KEYS = frozenset({"diffusion", "checkpoint"})

# --- Flux (CCC character sheets + Flux LoRA validation renders) ---
# Non-gated fp8 set: Comfy-Org/flux1-dev + comfyanonymous/flux_text_encoders
FLUX_UNET = "flux1-dev-fp8.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_T5XXL = "t5xxl_fp8_e4m3fn.safetensors"
FLUX_VAE = "ae.safetensors"  # already in models/vae

# --- Characters & LoRA training (stages 3-4) ---
CHARACTERS_DIR = Path(os.getenv("CHARACTERS_DIR", str(STATE_DIR / "characters"))).resolve()
AI_TOOLKIT_DIR = Path(os.getenv("AI_TOOLKIT_DIR", str(PROJECT_ROOT / "ai-toolkit"))).resolve()
LORA_DEFAULTS: dict[str, Any] = {
    "rank": 16,
    "alpha": 16,
    "steps": 1500,
    "lr": 1e-4,
    "save_every": 250,
    "resolution": [512, 768],
}
LORA_MAX_ATTEMPTS = int(os.getenv("LORA_MAX_ATTEMPTS", "5"))
LORA_SCORE_THRESHOLD = float(os.getenv("LORA_SCORE_THRESHOLD", str(JUDGE_SCORE_THRESHOLD)))

# Persona & pre-generation intake interview
PERSONA = (os.getenv("PERSONA", "ara") or "ara").strip().lower()
PERSONA_DIR = STATE_DIR / "personas"  # user personas override bundled ones
SOUL = (os.getenv("SOUL", "studio") or "studio").strip().lower()
SOUL_DIR = STATE_DIR / "souls"  # user souls override bundled ones
INTAKE_ENABLED = os.getenv("INTAKE_ENABLED", "1").lower() in ("1", "true", "yes", "on")
INTAKE_MAX_ROUNDS = int(os.getenv("INTAKE_MAX_ROUNDS", "6"))

# Fractal renderer + music video defaults (CLI/pipeline fallbacks)
FRACTAL_DEFAULTS: dict[str, Any] = {
    "fps": 24,
    "width": 768,
    "height": 512,
    "palette": "fire",
    "target": "seahorse",
    "duration_s": 20.0,
}
MUSIC_DEFAULTS: dict[str, Any] = {
    "min_shot_s": 2.0,
    "high_energy_s": 2.0,
    "low_energy_s": 4.0,
}

WORKFLOW_FILES: dict[str, str] = {
    "base": "base_t2v_i2v.json",
    "eros": "eros_t2v_i2v.json",
    "directors": "directors.json",
    "lipsync": "lipsync_ia2v.json",
    "wan22": "260713_MICKMUMPITZ_WAN-2-2-VID_1-0_api.json",
}


def ensure_dirs() -> None:
    for d in (
        WORKFLOWS_DIR,
        OUTPUTS_DIR,
        STATE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def resolve_model_path(filename: str) -> Path | None:
    """Search common model subfolders for a weight file."""
    subdirs = (
        "checkpoints",
        "diffusion_models",
        "loras",
        "vae",
        "text_encoders",
        "unet",
    )
    for sub in subdirs:
        p = MODELS_DIR / sub / filename
        if p.is_file():
            return p
        # Also check ComfyUI default models tree
        p2 = COMFYUI_ROOT / "models" / sub / filename
        if p2.is_file():
            return p2
    return None


def get_quality_profile(name: str | None = None) -> dict:
    key = (name or DEFAULT_QUALITY or "balanced").strip().lower()
    return dict(QUALITY_PROFILES.get(key) or QUALITY_PROFILES["balanced"])


def is_valid_ltx_frames(n: int) -> bool:
    """True when n is a legal LTX count: 8k+1 and at least 9."""
    try:
        frames = int(n)
    except (TypeError, ValueError):
        return False
    return frames >= 9 and (frames - 1) % 8 == 0


def snap_ltx_frames(n: int) -> int:
    """Nearest valid LTX frame count (8n+1), minimum 9. Never returns 8."""
    try:
        raw = int(n)
    except (TypeError, ValueError):
        return DEFAULT_FRAMES
    if raw <= 9:
        return 9
    k = max(1, round((raw - 1) / 8))
    return k * 8 + 1


def frames_for_duration(
    duration_s: float,
    fps: int = DEFAULT_FPS,
    *,
    max_s: float | None = None,
    snap: int = 8,
) -> int:
    """Convert seconds to a frame count (snap*n+1), clamped by segment max."""
    cap = max_s if max_s is not None else SEGMENT_MAX_S
    d = min(max(duration_s, 1.0), cap)
    raw = int(round(d * fps))
    n = max(1, round((raw - 1) / snap))
    frames = n * snap + 1
    max_frames = int(cap * fps)
    max_n = max(1, (max_frames - 1) // snap)
    frames = min(frames, max_n * snap + 1)
    if snap == 8:
        return snap_ltx_frames(frames)
    return frames


def plan_segment_durations(
    total_s: float,
    *,
    quality: str | None = None,
) -> list[float]:
    """
    Split a long request into per-clip durations that fit VRAM.
    Example: 30s + segment_max 6 → [6,6,6,6,6]
    """
    profile = get_quality_profile(quality)
    total = min(max(float(total_s), 1.0), float(profile["max_total_s"]))
    seg_max = float(profile["segment_max_s"])
    import math

    n = max(1, int(math.ceil(total / seg_max)))
    # Prefer equal segments; last absorbs remainder
    base = total / n
    # Snap each segment to something that yields valid 8n+1 frames (~0.33s steps)
    segs = [round(base, 2) for _ in range(n)]
    # Fix float drift on last
    segs[-1] = round(total - sum(segs[:-1]), 2)
    return [max(1.0, min(s, seg_max)) for s in segs]


def clamp_resolution(
    width: int,
    height: int,
    *,
    quality: str | None = None,
) -> tuple[int, int]:
    profile = get_quality_profile(quality)
    max_w = int(profile.get("max_width") or MAX_WIDTH)
    max_h = int(profile.get("max_height") or MAX_HEIGHT)
    w = min(max(width, 256), max_w)
    h = min(max(height, 256), max_h)
    w = (w // 32) * 32
    h = (h // 32) * 32
    return max(w, 256), max(h, 256)
