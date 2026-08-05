"""Render an ai-toolkit Flux LoRA training config (YAML) for a character.

Modeled on ai-toolkit's ``config/examples/train_lora_flux_24gb.yaml``.
Paths are absolute with FORWARD SLASHES — Windows backslashes break YAML
parsing in ai-toolkit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from master_agent.config import (
    CHARACTERS_DIR,
    FLUX_UNET,
    LORA_DEFAULTS,
    MODELS_DIR,
    resolve_model_path,
)

_OVERRIDE_KEYS = ("rank", "alpha", "steps", "lr", "save_every", "resolution")


def _fwd(path: Path) -> str:
    """Absolute path with forward slashes (Windows-safe YAML)."""
    return str(Path(path).resolve()).replace("\\", "/")


def _flux_unet_path() -> str:
    resolved = resolve_model_path(FLUX_UNET)
    if resolved is None:
        resolved = MODELS_DIR / "diffusion_models" / FLUX_UNET
    return _fwd(resolved)


def render_train_config(character: dict, overrides: dict | None = None) -> str:
    """YAML training config string for the character's dataset.

    ``character`` is the character.json dict (name, trigger_word, appearance).
    ``overrides`` may replace rank/alpha/steps/lr/save_every/resolution.
    """
    params: dict[str, Any] = dict(LORA_DEFAULTS)
    for key in _OVERRIDE_KEYS:
        if overrides and key in overrides and overrides[key] is not None:
            params[key] = overrides[key]

    name = str(character.get("name") or "character")
    trigger = str(character.get("trigger_word") or f"zxc_{name}")
    appearance = str(character.get("appearance") or name)
    dataset_path = _fwd(CHARACTERS_DIR / name / "dataset")

    sample_prompts = [
        f"[trigger], {appearance}, front view, close-up portrait, simple background",
        f"[trigger], {appearance}, three-quarter view, smile, medium shot",
        f"[trigger], {appearance}, full body shot, standing, neutral expression",
        f"[trigger], {appearance}, profile view, serious expression",
    ]

    config = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "output",
                    "device": "cuda:0",
                    "trigger_word": trigger,
                    "network": {
                        "type": "lora",
                        "linear": int(params["rank"]),
                        "linear_alpha": int(params["alpha"]),
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": int(params["save_every"]),
                        "max_step_saves_to_keep": 4,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [int(r) for r in params["resolution"]],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": int(params["steps"]),
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": float(params["lr"]),
                        "ema_config": {"use_ema": True, "ema_decay": 0.99},
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": _flux_unet_path(),
                        "is_flux": True,
                        "quantize": True,
                        "low_vram": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": int(params["save_every"]),
                        "width": 1024,
                        "height": 1024,
                        "prompts": sample_prompts,
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {"name": name, "version": "1.0"},
    }
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
