"""Sulphur LTX 2.3 studio workflows.

The public repo ships the graphs and the path convention. LoRA weight files
stay on the tower. This module does not list a LoRA filename inventory and
does not add download URLs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

# Queueable API graphs. UI sources sit beside them under workflows/sulphur/.
SULPHUR_FILES: dict[str, str] = {
    "ltx23_i2v_base": "sulphur/ltx23_i2v_base_api.json",
    "ltx23_i2v_distilled": "sulphur/ltx23_i2v_distilled.json",
    "ltx23_t2v_base": "sulphur/ltx23_t2v_base_api.json",
    "ltx23_t2v_distilled": "sulphur/ltx23_t2v_distilled_api.json",
}

SULPHUR_META: dict[str, dict[str, Any]] = {
    "ltx23_i2v_base": {
        "description": (
            "LTX 2.3 Sulphur image-to-video, base two-stage. "
            "Local LoRA weights stay on the tower."
        ),
        "modes": ["i2v"],
    },
    "ltx23_i2v_distilled": {
        "description": (
            "LTX 2.3 Sulphur image-to-video, distilled API graph. "
            "Local LoRA weights stay on the tower."
        ),
        "modes": ["i2v"],
    },
    "ltx23_t2v_base": {
        "description": (
            "LTX 2.3 Sulphur text-to-video, base two-stage. "
            "Local LoRA weights stay on the tower."
        ),
        "modes": ["t2v"],
    },
    "ltx23_t2v_distilled": {
        "description": (
            "LTX 2.3 Sulphur text-to-video, distilled two-stage. "
            "Local LoRA weights stay on the tower."
        ),
        "modes": ["t2v"],
    },
}

# Tower loras layout. Bare graph tokens resolve in either directory.
SULPHUR_LORA_RELATIVE_DIRS: tuple[tuple[str, ...], ...] = (
    ("loras", "sulphur"),
    ("loras",),
)


def is_sulphur_variant(variant: str | None) -> bool:
    key = (variant or "").strip()
    return key in SULPHUR_FILES


def sulphur_lora_dirs(roots: Iterable[Path]) -> list[Path]:
    """Candidate directories for one Sulphur LoRA token. Does not list files."""
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for parts in SULPHUR_LORA_RELATIVE_DIRS:
            path = Path(root).joinpath(*parts)
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return out


def resolve_sulphur_lora(
    filename: str,
    roots: Iterable[Path] | None = None,
) -> Path | None:
    """Resolve one graph filename token under the tower loras layout.

    Prefers ``models/loras/sulphur/<name>``, then ``models/loras/<name>``.
    A token that already contains a relative path is tried as-is. Does not
    return a directory listing.
    """
    raw = str(filename or "").strip()
    if not raw:
        return None
    slash = raw.replace("\\", "/")
    base = slash.rsplit("/", 1)[-1]
    if not base or base in {".", ".."}:
        return None
    if roots is None:
        from master_agent.models.weights import model_search_roots

        search = list(model_search_roots())
    else:
        search = [Path(root) for root in roots]
    for root in search:
        if not slash.startswith("/") and "/" in slash:
            direct = Path(root).joinpath(*[p for p in slash.split("/") if p])
            if _usable(direct):
                return direct
        for folder in sulphur_lora_dirs([root]):
            candidate = folder / base
            if _usable(candidate):
                return candidate
    try:
        from master_agent.models.weights import find_weight_file
    except Exception:
        return None
    return find_weight_file(base, search)


def patch_sulphur_graph(
    workflow: dict[str, Any],
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
) -> None:
    """Write Buddy run fields onto scalar widgets. Leave linked widgets alone.

    Positive text lives on ``PrimitiveStringMultiline`` titled Prompt.
    Negative text is the CLIP encode that feeds ``LTXVConditioning.negative``.
    Width, height, and length are ``PrimitiveInt`` widgets when those values
    are scalars. LoRA filename tokens are not rewritten.
    """
    if prompt:
        for node in _nodes(workflow, "PrimitiveStringMultiline"):
            title = _title(node)
            if "neg" in title or "prompt" not in title:
                continue
            _set_scalar(node, "value", prompt)
    if negative_prompt:
        for nid in _negative_clip_ids(workflow):
            node = workflow.get(nid)
            if isinstance(node, dict):
                _set_scalar(node, "text", negative_prompt)
    _set_primitive_int(workflow, "width", width)
    _set_primitive_int(workflow, "height", height)
    _set_primitive_int(workflow, "length", frames)


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _nodes(workflow: dict[str, Any], class_type: str):
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            yield node


def _title(node: dict[str, Any]) -> str:
    return str((node.get("_meta") or {}).get("title") or "").strip().lower()


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _set_scalar(node: dict[str, Any], key: str, value: Any) -> bool:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
        node["inputs"] = inputs
    if _is_link(inputs.get(key)):
        return False
    inputs[key] = value
    return True


def _set_primitive_int(workflow: dict[str, Any], title: str, value: int | None) -> None:
    if value is None:
        return
    want = title.strip().lower()
    for node in _nodes(workflow, "PrimitiveInt"):
        if _title(node) != want:
            continue
        _set_scalar(node, "value", int(value))


def _negative_clip_ids(workflow: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for node in _nodes(workflow, "LTXVConditioning"):
        negative = (node.get("inputs") or {}).get("negative")
        if _is_link(negative):
            nid = str(negative[0])
            if nid not in found:
                found.append(nid)
    return found
