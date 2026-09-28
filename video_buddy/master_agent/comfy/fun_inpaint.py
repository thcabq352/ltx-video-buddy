"""Supply ``fun_inpaint_mask.png`` before a Wan Fun Inpaint queue.

LoadImageMask reads ComfyUI's input folder. The API template names
``fun_inpaint_mask.png`` (red channel = the region to fill). A static
``/object_info`` pass fails that combo until the file exists. The live
pipeline writes a small RGB mask and uploads it under that name, the same
way other input images are uploaded, before lint and queue.
"""

from __future__ import annotations

import struct
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

FUN_INPAINT_MASK_NAME = "fun_inpaint_mask.png"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def default_inpaint_mask_png(size: int = 64) -> bytes:
    """RGB PNG. Center rectangle is white (red channel 255); edges are black."""
    side = max(8, int(size))
    x0, x1 = side // 4, side - side // 4
    y0, y1 = side // 4, side - side // 4
    raw = bytearray()
    for y in range(side):
        raw.append(0)
        for x in range(side):
            value = 255 if x0 <= x < x1 and y0 <= y < y1 else 0
            raw.extend((value, value, value))
    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def _mask_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "LoadImageMask":
            nodes.append(node)
    return nodes


def workflow_needs_default_mask(workflow: dict[str, Any]) -> bool:
    """True when a LoadImageMask still points at the template placeholder."""
    for node in _mask_nodes(workflow):
        image = str((node.get("inputs") or {}).get("image") or "")
        if image in ("", FUN_INPAINT_MASK_NAME):
            return True
    return False


def ensure_fun_inpaint_mask(
    workflow: dict[str, Any],
    upload: Callable[[Path], str],
) -> str | None:
    """Upload a generated mask and point placeholder LoadImageMask nodes at it.

    A caller-supplied name (anything other than the placeholder) is left alone.
    ``upload`` receives a temp PNG whose filename is ``fun_inpaint_mask.png``.
    """
    if not workflow_needs_default_mask(workflow):
        return None
    png = default_inpaint_mask_png()
    with tempfile.TemporaryDirectory(prefix="fun-inpaint-") as tmp:
        path = Path(tmp) / FUN_INPAINT_MASK_NAME
        path.write_bytes(png)
        uploaded = upload(path) or FUN_INPAINT_MASK_NAME
    name = str(uploaded)
    for node in _mask_nodes(workflow):
        inputs = node.setdefault("inputs", {})
        image = str(inputs.get("image") or "")
        if image in ("", FUN_INPAINT_MASK_NAME):
            inputs["image"] = name
    return name
