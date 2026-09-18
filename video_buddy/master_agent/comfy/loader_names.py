"""Map weight filenames onto the exact combo strings Comfy's object_info lists.

UnetLoaderGGUF / CLIPLoaderGGUF (and folder-aware UNET loaders) list files
relative to the model folder. On Windows that is ``gguf\\name.gguf``; on
posix it is ``gguf/name.gguf``. Buddy used to write ``Path.name`` (bare
filename), which is not in the list and fails ``POST /prompt`` with 400.

Matching prefers the live combo list, then inventory / on-disk path. Already
prefixed names are never given a second folder prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

# Widgets that hold weight filenames (GGUF unet/clip plus the usual loaders).
LOADER_NAME_KEYS = (
    "unet_name",
    "clip_name",
    "clip_name1",
    "clip_name2",
    "clip_name3",
    "clip_name4",
    "gguf_name",
    "ckpt_name",
    "vae_name",
    "lora_name",
    "text_encoder",
    "text_encoder1",
    "text_encoder2",
    "model_name",
    "diffusion_model",
)

# First path component under Comfy's models/ tree (or Buddy MODELS_DIR).
ROLE_FOLDERS = frozenset(
    {
        "checkpoints",
        "diffusion_models",
        "unet",
        "loras",
        "vae",
        "text_encoders",
        "clip",
        "clip_vision",
        "controlnet",
        "upscale_models",
        "embeddings",
        "latent_upscale_models",
        "model_patches",
        "SEEDVR2",
        "seedvr2",
    }
)


def slash_key(name: str) -> str:
    return str(name).replace("\\", "/")


def basename_key(name: str) -> str:
    return slash_key(name).rsplit("/", 1)[-1]


def is_folder_prefixed(name: str) -> bool:
    return "/" in slash_key(name)


def combo_choices_from_spec(spec: Any) -> Optional[list[Any]]:
    """Return combo choices from a Comfy input spec, or None if not a combo."""
    if isinstance(spec, (list, tuple)) and spec:
        if isinstance(spec[0], (list, tuple)):
            return list(spec[0])
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            opts = spec[1].get("options") or spec[1].get("choices")
            if isinstance(opts, (list, tuple)):
                return list(opts)
    return None


def combo_choices_for(
    object_info: dict[str, Any] | None,
    class_type: str,
    input_name: str,
) -> Optional[list[Any]]:
    if not object_info or not class_type:
        return None
    info = object_info.get(class_type) or {}
    spec_in = info.get("input") or {}
    for section in ("required", "optional"):
        spec = (spec_in.get(section) or {}).get(input_name)
        if spec is not None:
            return combo_choices_from_spec(spec)
    return None


def match_combo_name(name: str, choices: Sequence[Any] | None) -> Optional[str]:
    """Return the exact list entry for ``name``, or None if nothing unique matches."""
    if not name or not choices:
        return None
    strings = [c for c in choices if isinstance(c, str)]
    if name in strings:
        return name
    norm = slash_key(name)
    for choice in strings:
        if slash_key(choice) == norm:
            return choice
    base = basename_key(name)
    matches = [c for c in strings if basename_key(c) == base]
    if not matches:
        lowered = base.lower()
        matches = [c for c in strings if basename_key(c).lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and is_folder_prefixed(name):
        for choice in matches:
            if slash_key(choice) == norm:
                return choice
        for choice in matches:
            cslash = slash_key(choice)
            if cslash.endswith(norm) or norm.endswith(cslash):
                return choice
    return None


def name_from_local_path(path: Path | None) -> Optional[str]:
    """Combo-style relative name under a Comfy role folder (posix separators)."""
    if path is None:
        return None
    parts = path.parts
    role_idx = None
    for i, part in enumerate(parts):
        if part in ROLE_FOLDERS:
            role_idx = i
    if role_idx is None or role_idx + 1 >= len(parts):
        return path.name
    rel = parts[role_idx + 1 :]
    if len(rel) == 1:
        return rel[0]
    return "/".join(rel)


def name_from_inventory(name: str, inventory: Any) -> Optional[str]:
    """Folder-prefixed name from inventory. Already-prefixed names are unchanged."""
    if not name or inventory is None:
        return None
    if is_folder_prefixed(name):
        return name
    resolve = getattr(inventory, "resolve", None)
    if not callable(resolve):
        return None
    entry = resolve(name)
    if entry is None:
        return None
    rel = slash_key(getattr(entry, "rel_path", "") or "")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 2 and parts[0] in ROLE_FOLDERS:
        parts = parts[1:]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return "/".join(parts)


def normalize_loader_name(
    name: str,
    *,
    choices: Sequence[Any] | None = None,
    inventory: Any = None,
    local_path: Path | None = None,
) -> str:
    """Return the string that should be written into a Comfy loader widget."""
    if not isinstance(name, str) or not name:
        return name
    matched = match_combo_name(name, choices)
    if matched is not None:
        return matched
    if is_folder_prefixed(name):
        return name
    derived = name_from_local_path(local_path)
    if derived and derived != name:
        matched = match_combo_name(derived, choices)
        return matched if matched is not None else derived
    inv_name = name_from_inventory(name, inventory)
    if inv_name:
        matched = match_combo_name(inv_name, choices)
        return matched if matched is not None else inv_name
    return name


def normalize_loader_widgets(
    workflow: dict[str, Any],
    *,
    object_info: dict[str, Any] | None = None,
    inventory: Any = None,
) -> list[tuple[str, str, str, str]]:
    """Rewrite loader widgets in-place. Returns (node_id, key, old, new) changes."""
    changed: list[tuple[str, str, str, str]] = []
    if not isinstance(workflow, dict):
        return changed
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type") or "")
        for key in LOADER_NAME_KEYS:
            value = inputs.get(key)
            if not isinstance(value, str) or not value:
                continue
            choices = combo_choices_for(object_info, class_type, key)
            new = normalize_loader_name(value, choices=choices, inventory=inventory)
            if new != value:
                inputs[key] = new
                changed.append((str(nid), key, value, new))
    return changed
