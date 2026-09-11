"""LoRA A/B lock: encoder + seed family stay put; only name/strength may move.

Do not swap gemma_3_12B_it_fp8_scaled / the default TE for Heretic mid-A/B.
Windows folder-prefixed checkpoint names keep their backslash style.
"""

from __future__ import annotations

from typing import Any

# Default LTX text-encoder family. Do not replace with Heretic during A/B.
DEFAULT_TE_FAMILY = frozenset(
    {
        "gemma_3_12B_it_fp8_scaled",
        "gemma_3_12B_it_fp8_scaled.safetensors",
        "gemma_3_12B_it_fp4_mixed",
        "gemma_3_12B_it_fp4_mixed.safetensors",
    }
)
HERETIC_MARKERS = ("heretic",)
LOCKED_KEYS = (
    "text_encoder",
    "seed",
    "seed_family",
    "checkpoint",
    "checkpoint_high",
    "checkpoint_low",
    "clip_l",
    "t5xxl",
    "vae_name",
)
MUTABLE_KEYS = frozenset({"lora", "lora_name", "strength", "lora_strength", "strength_model"})


class AbLockError(ValueError):
    """Illegal mid-A/B swap (encoder, seed family, or Heretic TE)."""


def is_heretic_encoder(name: Any) -> bool:
    text = str(name or "").lower()
    return any(marker in text for marker in HERETIC_MARKERS)


def preserve_checkpoint_style(name: Any) -> Any:
    """Keep Windows folder-prefixed names (wan\\ckpt.safetensors) as backslashes."""
    if not isinstance(name, str):
        return name
    # Never pathlib-normalize. If a folder prefix already uses backslashes, keep them.
    return name


def seed_family(base_seed: int, index: int = 0) -> int:
    """A/B slots share one family: same base seed, optional +index."""
    return int(base_seed) + int(index)


def same_seed_family(left: Any, right: Any, *, span: int = 1) -> bool:
    try:
        return abs(int(left) - int(right)) <= int(span)
    except (TypeError, ValueError):
        return False


def lock_lora_ab(
    base: dict[str, Any],
    *,
    lora_name: str,
    strength: float | None = None,
    slot: int = 0,
) -> dict[str, Any]:
    """Build a challenger dict: only LoRA name/strength change."""
    locked = {key: base.get(key) for key in LOCKED_KEYS if key in base}
    if "checkpoint" in locked:
        locked["checkpoint"] = preserve_checkpoint_style(locked["checkpoint"])
    for key in ("checkpoint_high", "checkpoint_low"):
        if key in locked:
            locked[key] = preserve_checkpoint_style(locked[key])
    te = locked.get("text_encoder", base.get("text_encoder"))
    if is_heretic_encoder(te):
        raise AbLockError("do not swap default TE for Heretic mid-A/B")
    locked["text_encoder"] = te
    base_seed = base.get("seed")
    if base_seed is not None:
        locked["seed"] = seed_family(int(base_seed), slot)
        locked["seed_family"] = int(base_seed)
    locked["lora"] = lora_name
    locked["lora_name"] = lora_name
    if strength is not None:
        locked["strength"] = float(strength)
        locked["lora_strength"] = float(strength)
        locked["strength_model"] = float(strength)
    return locked


def validate_ab_pair(base: dict[str, Any], challenger: dict[str, Any]) -> list[str]:
    """Return human-readable violations (empty == safe to compare)."""
    errors: list[str] = []
    if is_heretic_encoder(challenger.get("text_encoder")):
        errors.append("do not swap default TE for Heretic mid-A/B")
    if challenger.get("text_encoder") != base.get("text_encoder"):
        errors.append("text encoder must stay locked during LoRA A/B")
    if base.get("seed") is not None and not same_seed_family(
        base.get("seed"), challenger.get("seed")
    ):
        errors.append("seed family must stay locked during LoRA A/B")
    for key in ("checkpoint", "checkpoint_high", "checkpoint_low"):
        left, right = base.get(key), challenger.get(key)
        if left and right and preserve_checkpoint_style(left) != preserve_checkpoint_style(right):
            errors.append(f"{key} must stay locked during LoRA A/B")
        if isinstance(right, str) and "\\" in str(left or "") and "/" in right and "\\" not in right:
            errors.append(f"{key} lost Windows backslash style")
    return errors


def apply_ab_lock_to_workflow(
    workflow: dict[str, Any],
    *,
    text_encoder: str | None,
    seed: int | None,
    lora_name: str | None = None,
    strength: float | None = None,
) -> dict[str, Any]:
    """Write locked encoder + seed; only touch LoRA name/strength when given."""
    if is_heretic_encoder(text_encoder):
        raise AbLockError("do not swap default TE for Heretic mid-A/B")
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type") or "")
        if text_encoder is not None and "text_encoder" in inputs:
            inputs["text_encoder"] = text_encoder
        if seed is not None:
            if "noise_seed" in inputs:
                inputs["noise_seed"] = int(seed)
            elif "seed" in inputs:
                inputs["seed"] = int(seed)
        if lora_name and class_type in {
            "LoraLoader",
            "LoraLoaderModelOnly",
            "LTXICLoRALoaderModelOnly",
        }:
            if "lora_name" in inputs or "lora" in inputs:
                key = "lora_name" if "lora_name" in inputs else "lora"
                inputs[key] = lora_name
            if strength is not None:
                for key in ("strength_model", "strength"):
                    if key in inputs:
                        inputs[key] = float(strength)
    return workflow
