"""Readiness checks before an ingested graph is queued.

Missing custom nodes fail closed. Tiny preview VAEs on tiled decode follow
``vae_guard``: a baked default is swapped, an explicit request raises.
"""

from __future__ import annotations

from typing import Any

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
    "LTX23_FULL_VIDEO_VAE",
    "MissingCustomNodeError",
    "TinyVAETiledDecodeError",
    "apply_vae_guard",
    "assert_known_nodes",
    "dangers_for",
]


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
