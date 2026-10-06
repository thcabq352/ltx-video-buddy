"""Keep tiny LTX preview VAEs off tiled decode nodes.

ComfyUI 0.38 reports every ``taeltx*`` / ``tae*`` decoder as 16x spatial and
4x temporal. Those files actually scale 32x / 8x, so a tiled decode builds
its blend mask for the wrong tile and crashes. Plain ``VAEDecode`` is fine.
The full decoder attested on the shipped movie-builder graph is
``LTX23_video_vae_bf16.safetensors``.
"""

from __future__ import annotations

from typing import Any, Iterator

from master_agent.config import LTX23_FULL_VIDEO_VAE

_TILED_DECODE_EXACT = frozenset({"VAEDecodeTiled", "LTXVTiledVAEDecode"})


class TinyVAETiledDecodeError(RuntimeError):
    """A tiny/preview VAE was paired with tiled decode."""


def is_tiny_preview_vae(name: str | None) -> bool:
    """True for ``taeltx*`` and ``tae*`` preview decoder filenames."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    if not base:
        return False
    return base.startswith("taeltx") or base.startswith("tae")


def is_tiled_decode_class(class_type: str | None) -> bool:
    name = str(class_type or "")
    if name in _TILED_DECODE_EXACT:
        return True
    return "Tiled" in name and "Decode" in name


def _node_map(workflow: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(workflow, dict):
        return out
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type"):
            out[str(nid)] = node
    return out


def graph_has_tiled_decode(workflow: dict[str, Any] | None) -> bool:
    return any(is_tiled_decode_class(node.get("class_type")) for node in _node_map(workflow).values())


def _link_id(value: Any) -> str | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[1], int)
        and value[0] is not None
    ):
        return str(value[0])
    return None


def iter_tiled_vae_name_slots(
    workflow: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], str, str]]:
    """Yield ``(node, input_key, filename)`` for VAE names that feed a tiled decode."""
    nodes = _node_map(workflow)
    for node in nodes.values():
        if not is_tiled_decode_class(node.get("class_type")):
            continue
        inputs = node.get("inputs") or {}
        direct = inputs.get("vae_name")
        if isinstance(direct, str):
            yield node, "vae_name", direct
        seen: set[str] = set()
        stack: list[Any] = [inputs.get("vae")]
        while stack:
            link = _link_id(stack.pop())
            if link is None or link in seen:
                continue
            seen.add(link)
            src = nodes.get(link)
            if not src:
                continue
            src_inputs = src.get("inputs") or {}
            name = src_inputs.get("vae_name")
            if isinstance(name, str):
                yield src, "vae_name", name
            if "vae" in src_inputs:
                stack.append(src_inputs.get("vae"))


def tiny_vae_names_on_tiled_decode(workflow: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for _node, _key, name in iter_tiled_vae_name_slots(workflow):
        if is_tiny_preview_vae(name) and name not in names:
            names.append(name)
    return names


def replace_tiny_vae_on_tiled_decode(
    workflow: dict[str, Any],
    replacement: str = LTX23_FULL_VIDEO_VAE,
) -> int:
    count = 0
    for node, key, name in iter_tiled_vae_name_slots(workflow):
        if not is_tiny_preview_vae(name):
            continue
        node.setdefault("inputs", {})[key] = replacement
        count += 1
    return count


def explicit_tiny_vae_message(name: str) -> str:
    return (
        f"Tiny preview VAE {name} is not valid with tiled decode "
        "(VAEDecodeTiled, LTXVTiledVAEDecode, or any class with Tiled and Decode). "
        f"Use {LTX23_FULL_VIDEO_VAE}, or plain VAEDecode for a fast preview."
    )


def vae_name_for_tiled_graph(
    vae_name: str | None,
    workflow: dict[str, Any],
    *,
    explicit: bool,
) -> str | None:
    """VAE filename to write onto loaders.

    A default tiny preview VAE on a tiled graph becomes the full LTX 2.3
    video VAE. An explicit tiny preview VAE on a tiled graph raises.
    """
    if vae_name and is_tiny_preview_vae(vae_name) and graph_has_tiled_decode(workflow):
        if explicit:
            raise TinyVAETiledDecodeError(explicit_tiny_vae_message(vae_name))
        replace_tiny_vae_on_tiled_decode(workflow)
        return LTX23_FULL_VIDEO_VAE
    return vae_name


def explicit_tiny_request(
    vae_name: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Return the tiny VAE filename when a CLI flag or field override asked for one."""
    if vae_name and is_tiny_preview_vae(vae_name):
        return str(vae_name)
    for fields in (overrides or {}).values():
        if not isinstance(fields, dict):
            continue
        value = fields.get("vae_name")
        if isinstance(value, str) and is_tiny_preview_vae(value):
            return value
    return None


def reject_tiny_vae_on_tiled_decode(workflow: dict[str, Any]) -> None:
    """Fail closed when a tiny preview VAE still feeds a tiled decode."""
    names = tiny_vae_names_on_tiled_decode(workflow)
    if not names:
        return
    raise TinyVAETiledDecodeError(explicit_tiny_vae_message(", ".join(names)))
