"""Readiness checks before an ingested graph is queued.

Missing custom nodes fail closed. Missing model filenames are reported
against a Comfy model list and Buddy ``extra_model_paths`` when that
inventory exists. Filenames are never rewritten to a substitute. Tiny
preview VAEs on tiled decode follow ``vae_guard``: a baked default is
swapped, an explicit request raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from master_agent.comfy.ingest.classify import missing_class_types
from master_agent.config import LTX23_FULL_VIDEO_VAE
from master_agent.comfy.vae_guard import (
    TinyVAETiledDecodeError,
    explicit_tiny_request,
    explicit_tiny_vae_message,
    graph_has_tiled_decode,
    reject_tiny_vae_on_tiled_decode,
    replace_tiny_vae_on_tiled_decode,
    tiny_vae_names_on_tiled_decode,
)

# Re-export so callers can catch the guard error without a second import path.
__all__ = [
    "DOCTOR_POINTER",
    "DOWNLOAD_POINTER",
    "LTX23_FULL_VIDEO_VAE",
    "MissingCustomNodeError",
    "MissingModelError",
    "TinyVAETiledDecodeError",
    "apply_vae_guard",
    "assert_known_nodes",
    "buddy_model_inventory",
    "dangers_for",
    "missing_model_filenames",
    "pointers_for",
    "referenced_model_filenames",
]

DOCTOR_POINTER = "python -m master_agent doctor"
DOWNLOAD_POINTER = "python -m master_agent download-models"

_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".sft",
)
_MODEL_INPUTS = frozenset(
    {
        "ckpt_name",
        "vae_name",
        "lora_name",
        "unet_name",
        "clip_name",
        "model_name",
        "text_encoder",
    }
)
_MODEL_FOLDERS = (
    "checkpoints",
    "loras",
    "vae",
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
)


class MissingModelError(RuntimeError):
    """A referenced weight is not in the Comfy list or extra_model_paths.

    The graph filename is left unchanged. Buddy does not substitute another file.
    """

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        listed = ", ".join(self.missing)
        super().__init__(
            f"missing model(s): {listed}. "
            "Buddy does not substitute another weight. "
            f"{DOCTOR_POINTER}. {DOWNLOAD_POINTER}."
        )


class MissingCustomNodeError(RuntimeError):
    """One or more class_types are not in object_info or the offline catalog."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        listed = ", ".join(self.missing)
        super().__init__(
            f"missing custom node(s): {listed}. "
            "Buddy does not auto-install custom nodes. "
            "Install the pack in Comfy yourself, then retry."
        )


def assert_known_nodes(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> None:
    missing = missing_class_types(workflow, object_info)
    if missing:
        raise MissingCustomNodeError(missing)


def _looks_like_weight(value: str) -> bool:
    lower = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(lower.endswith(suffix) for suffix in _WEIGHT_SUFFIXES)


def referenced_model_filenames(workflow: dict[str, Any]) -> list[str]:
    """Weight filenames written on loader widgets. Order is stable. No substitution."""
    found: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            if key not in _MODEL_INPUTS and not _looks_like_weight(text):
                continue
            if not _looks_like_weight(text):
                continue
            if text not in found:
                found.append(text)
    return found


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def missing_model_filenames(
    workflow: dict[str, Any],
    inventory: set[str] | None,
) -> list[str]:
    """Filenames absent from ``inventory``.

    ``inventory is None`` means the check did not run (no Comfy list and no
    extra_model_paths). An empty inventory means the check ran and nothing
    was found, so every referenced file is missing. Names are not replaced.
    """
    if inventory is None:
        return []
    known = set(inventory)
    known_base = {_basename(name) for name in known}
    missing: list[str] = []
    for name in referenced_model_filenames(workflow):
        if name in known or _basename(name) in known or _basename(name) in known_base:
            continue
        missing.append(name)
    return missing


def pointers_for(*, missing_nodes: list[Any], missing_models: list[Any]) -> list[str]:
    if not missing_nodes and not missing_models:
        return []
    pointers: list[str] = []
    if missing_nodes or missing_models:
        pointers.append(DOCTOR_POINTER)
    if missing_models:
        pointers.append(DOWNLOAD_POINTER)
    return pointers


def _names_under(root: Path) -> set[str]:
    names: set[str] = set()
    if not root.is_dir():
        return names
    for folder in _MODEL_FOLDERS:
        directory = root / folder
        if not directory.is_dir():
            continue
        try:
            children = list(directory.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_file():
                names.add(child.name)
            elif child.is_dir():
                try:
                    nested = list(child.iterdir())
                except OSError:
                    continue
                for item in nested:
                    if item.is_file():
                        names.add(item.name)
    return names


def buddy_model_inventory() -> set[str] | None:
    """Filenames from Buddy ``extra_model_paths.yaml`` base paths, if that file exists.

    Returns None when the YAML is absent so a machine with no path file is
    not treated as an empty model library. Does not download or rename files.
    """
    from master_agent.comfy.model_paths import buddy_yaml_path

    path = buddy_yaml_path()
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    roots: list[Path] = []
    for section in data.values():
        if isinstance(section, dict) and section.get("base_path"):
            roots.append(Path(str(section["base_path"])))
    if not roots:
        return set()
    names: set[str] = set()
    for root in roots:
        names.update(_names_under(root))
    return names


def union_inventories(*groups: set[str] | None) -> set[str] | None:
    present = [group for group in groups if group is not None]
    if not present:
        return None
    merged: set[str] = set()
    for group in present:
        merged.update(group)
    return merged


def dangers_for(workflow: dict[str, Any]) -> list[dict[str, str]]:
    names = tiny_vae_names_on_tiled_decode(workflow)
    if not names:
        return []
    return [
        {
            "code": "tiny_vae_tiled",
            "action": "swap_default",
            "detail": (
                f"{', '.join(names)} feeds tiled decode. "
                f"A baked default swaps to {LTX23_FULL_VIDEO_VAE} on run. "
                "An explicit tiny VAE request fails closed."
            ),
        }
    ]


def apply_vae_guard(
    workflow: dict[str, Any],
    *,
    vae: str | None = None,
    node_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Apply ``vae_guard`` rules in place. Return human notes. Never deletes files."""
    requested = explicit_tiny_request(vae, node_overrides)
    if requested and graph_has_tiled_decode(workflow):
        raise TinyVAETiledDecodeError(explicit_tiny_vae_message(requested))
    notes: list[str] = []
    swapped = replace_tiny_vae_on_tiled_decode(workflow)
    if swapped:
        notes.append(
            f"swapped {swapped} tiny preview VAE slot(s) off tiled decode "
            f"to {LTX23_FULL_VIDEO_VAE}"
        )
    reject_tiny_vae_on_tiled_decode(workflow)
    return notes
