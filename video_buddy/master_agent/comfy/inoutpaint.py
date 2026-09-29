"""LTX 2.3 inpaint / outpaint layout, mask PNG, and queue-time upload.

The shipped graphs are ``workflows/ltx23_inoutpaint_api.json`` and
``workflows/ltx-2.5/LTX-2.5_ICLoRA_Inpaint_Outpaint_Two_Stage_Distilled_api.json``.
White mask pixels are the region the IC-LoRA regenerates. Black pixels are
held. Outpaint grows the canvas with ``ImagePadForOutpaint`` (pads are
multiples of 8, the node's step) and builds a matching mask: white on the new
border, black on the source. The 2.5 graph swaps its image-condition nodes to
``LTXVImgToVideoConditionOnly`` when the mode is outpaint.
"""

from __future__ import annotations

import base64
import math
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

VARIANT = "ltx23_inoutpaint"
LTX25_VARIANT = "ltx25_inoutpaint"
MASK_PLACEHOLDER = "example.png"
MASK_UPLOAD_NAME = "ltx23_inoutpaint_mask.png"
LTX25_MASK_UPLOAD_NAME = "ltx25_inoutpaint_mask.png"
PROVENANCE_VARIANTS = frozenset({VARIANT, LTX25_VARIANT})
OFFICIAL_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"

# 16:9-ish working size that stays on the latent grid (multiples of 64) and
# under the 768 long-edge cap used for a 16 GB two-stage pass.
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 448
DEFAULT_FRAMES = 25
DEFAULT_FPS = 24
MAX_LONG_EDGE = 768

# Model card / official ComfyUI-LTXVideo inpaint graph.
INPAINT_DILATE_STAGE1 = 15
INPAINT_DILATE_STAGE2 = 30
OUTPAINT_DILATE = 0
INPAINT_BLEND_DILATION = 6
OUTPAINT_BLEND_DILATION = 2

# Phrases that should win over lipsync when a source video is attached,
# and over Wan Fun when the request names LTX outpaint. Bare "inpaint"
# stays on wan_fun_inpaint. These stay on ltx23_inoutpaint.
ROUTE_KEYWORDS = (
    "ltx inpaint",
    "ltx outpaint",
    "ltx in-outpaint",
    "ltx inoutpaint",
    "in-outpaint",
    "inoutpaint",
    "outpaint",
    "outpainting",
    "extend the canvas",
    "extend the frame",
)


def requests_inoutpaint(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(key in lowered for key in ROUTE_KEYWORDS)


# Checked before ROUTE_KEYWORDS and before generic "ltx 2.5" / "ltx25".
# Bare "inpaint" and "ltx outpaint" do not match.
LTX25_ROUTE_KEYWORDS = (
    "ltx 2.5 inpaint",
    "ltx 2.5 outpaint",
    "ltx 2.5 inoutpaint",
    "ltx 2.5 in-outpaint",
    "ltx2.5 inpaint",
    "ltx2.5 outpaint",
    "ltx2.5 inoutpaint",
    "ltx25 inpaint",
    "ltx25 outpaint",
    "ltx25 inoutpaint",
    "ltx-2.5 inpaint",
    "ltx-2.5 outpaint",
    "ltx-2.5 inoutpaint",
    "ltx25_inoutpaint",
)


def requests_ltx25_inoutpaint(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(key in lowered for key in LTX25_ROUTE_KEYWORDS)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _rgb_png(width: int, height: int, rgb: bytes) -> bytes:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def paint_mask(
    width: int,
    height: int,
    *,
    fill: int = 0,
    rect: tuple[int, int, int, int] | None = None,
    rect_value: int = 255,
) -> bytes:
    """Packed RGB. ``rect`` is (x, y, w, h) painted ``rect_value``."""
    side_w = max(1, int(width))
    side_h = max(1, int(height))
    value = 255 if int(fill) else 0
    buf = bytearray((bytes((value, value, value)) * side_w) * side_h)
    if rect is None:
        return bytes(buf)
    x, y, w, h = (int(v) for v in rect)
    paint = 255 if int(rect_value) else 0
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(side_w, x + max(0, w))
    y1 = min(side_h, y + max(0, h))
    row = bytes((paint, paint, paint)) * (x1 - x0)
    for yy in range(y0, y1):
        start = (yy * side_w + x0) * 3
        buf[start : start + len(row)] = row
    return bytes(buf)


def mask_png_from_rgb(width: int, height: int, rgb: bytes) -> bytes:
    return _rgb_png(width, height, rgb)


def default_inpaint_mask_png(width: int = 64, height: int = 64) -> bytes:
    """Center rectangle white (regenerate). Edges black (keep)."""
    side_w = max(8, int(width))
    side_h = max(8, int(height))
    x0, x1 = side_w // 4, side_w - side_w // 4
    y0, y1 = side_h // 4, side_h - side_h // 4
    rgb = paint_mask(
        side_w,
        side_h,
        fill=0,
        rect=(x0, y0, x1 - x0, y1 - y0),
        rect_value=255,
    )
    return mask_png_from_rgb(side_w, side_h, rgb)


def outpaint_mask_png(
    canvas_w: int,
    canvas_h: int,
    left: int,
    top: int,
    src_w: int,
    src_h: int,
) -> bytes:
    """White canvas (generate). Black rectangle over the source (keep)."""
    rgb = paint_mask(
        canvas_w,
        canvas_h,
        fill=255,
        rect=(left, top, src_w, src_h),
        rect_value=0,
    )
    return mask_png_from_rgb(canvas_w, canvas_h, rgb)


def parse_aspect(text: str) -> float:
    """Width/height from ``9:16``, ``9/16``, or a float ratio."""
    raw = (text or "").strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("aspect is empty")
    for sep in (":", "/"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            w = float(left)
            h = float(right)
            if w <= 0 or h <= 0:
                raise ValueError(f"aspect {text!r} must be positive")
            return w / h
    ratio = float(raw)
    if ratio <= 0:
        raise ValueError(f"aspect {text!r} must be positive")
    return ratio


def _snap_up(n: int, step: int) -> int:
    n = int(n)
    if n <= 0:
        return step
    return ((n + step - 1) // step) * step


def _pad8(n: int) -> int:
    """ImagePadForOutpaint steps by 8. Keep a positive pad from collapsing to 0."""
    n = max(0, int(n))
    snapped = (n // 8) * 8
    if n > 0 and snapped == 0:
        return 8
    return snapped


def _target_canvas(
    src_w: int,
    src_h: int,
    *,
    aspect: float | None,
    target_w: int | None,
    target_h: int | None,
) -> tuple[int, int]:
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    if target_w is not None or target_h is not None:
        cw = int(target_w) if target_w else src_w
        ch = int(target_h) if target_h else src_h
        return max(cw, src_w), max(ch, src_h)
    if aspect is None or aspect <= 0:
        raise ValueError("outpaint needs an aspect ratio or a target width and height")
    if (src_w / src_h) > aspect:
        ch = int(math.ceil(src_w / aspect))
        cw = src_w
    else:
        cw = int(math.ceil(src_h * aspect))
        ch = src_h
    cw = max(_snap_up(cw, 32), src_w)
    ch = max(_snap_up(ch, 32), src_h)
    # Snapping one side can leave the other short of the source.
    return max(cw, src_w), max(ch, src_h)


def outpaint_layout(
    src_w: int,
    src_h: int,
    *,
    aspect: str | float | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
    max_long: int = MAX_LONG_EDGE,
) -> dict[str, Any]:
    """Pads, canvas, stage sizes, and the border mask PNG.

    The canvas contains the source. Pads are centered and multiples of 8.
    Stage 2 is the size the sampler actually runs (long edge <= ``max_long``,
    both sides multiples of 64) so stage 1 is exactly half for the x2 latent
    upscaler.
    """
    src_w = int(src_w)
    src_h = int(src_h)
    if src_w < 1 or src_h < 1:
        raise ValueError("source video size must be positive")
    ratio: float | None
    if isinstance(aspect, str):
        ratio = parse_aspect(aspect)
    else:
        ratio = float(aspect) if aspect else None
    desired_w, desired_h = _target_canvas(
        src_w,
        src_h,
        aspect=ratio,
        target_w=target_w,
        target_h=target_h,
    )
    extra_w = max(0, desired_w - src_w)
    extra_h = max(0, desired_h - src_h)
    left = _pad8(extra_w // 2)
    right = _pad8(extra_w - left)
    top = _pad8(extra_h // 2)
    bottom = _pad8(extra_h - top)
    canvas_w = src_w + left + right
    canvas_h = src_h + top + bottom
    stage_w, stage_h = fit_working_size(canvas_w, canvas_h, max_long=max_long)
    png = outpaint_mask_png(canvas_w, canvas_h, left, top, src_w, src_h)
    return {
        "mode": "outpaint",
        "src_w": src_w,
        "src_h": src_h,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "pad": {"left": left, "top": top, "right": right, "bottom": bottom},
        "width": stage_w,
        "height": stage_h,
        "stage1_w": stage_w // 2,
        "stage1_h": stage_h // 2,
        "mask_png": png,
    }


def fit_working_size(
    width: int,
    height: int,
    *,
    max_long: int = MAX_LONG_EDGE,
) -> tuple[int, int]:
    """Snap a desired output to multiples of 64 with a capped long edge.

    Multiples of 64 keep stage 1 (half) on the 32-pixel latent grid the x2
    spatial upscaler expects.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    cap = max(64, (int(max_long) // 64) * 64)
    scale = min(1.0, cap / float(max(width, height)))
    tw = width * scale
    th = height * scale
    target = tw / th if th else 1.0

    def _candidates(n: float) -> list[int]:
        lo = max(64, int(n // 64) * 64)
        opts = [lo]
        hi = lo + 64
        if hi <= cap:
            opts.append(hi)
        return opts

    best: tuple[tuple[float, float], int, int] | None = None
    for sw in _candidates(tw):
        for sh in _candidates(th):
            err = abs((sw / sh) - target)
            score = (err, abs(sw - tw) + abs(sh - th))
            if best is None or score < best[0]:
                best = (score, sw, sh)
    assert best is not None
    return best[1], best[2]


def probe_video_size(path: Path) -> tuple[int, int]:
    """Width and height of the first video stream. Requires ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not on PATH; cannot read the source video size") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed for {path.name}: {detail}")
    line = ""
    for row in (proc.stdout or "").splitlines():
        if "x" in row:
            line = row.strip()
            break
    if "x" not in line:
        raise RuntimeError(f"ffprobe returned no video size for {path.name}")
    w_s, h_s = line.split("x", 1)
    return int(w_s), int(h_s)


def _nodes(workflow: dict[str, Any], class_type: str) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            found.append((str(nid), node))
    return found


def _by_title(workflow: dict[str, Any], class_type: str, title: str) -> dict[str, Any] | None:
    want = title.lower()
    for _nid, node in _nodes(workflow, class_type):
        got = str((node.get("_meta") or {}).get("title") or "").lower()
        if got == want:
            return node
    return None


def _set(node: dict[str, Any] | None, key: str, value: Any) -> None:
    if node is None:
        return
    node.setdefault("inputs", {})[key] = value


def finalize_inoutpaint_graph(
    workflow: dict[str, Any],
    *,
    width: int,
    height: int,
    mode: str = "inpaint",
    pad: dict[str, int] | None = None,
    mask_png: bytes | None = None,
    ltx25: bool = False,
) -> None:
    """Write stage sizes, dilate/blend, and outpaint pads after the generic patcher.

    ``width`` / ``height`` are the stage-2 output. The empty latent is stage 1
    (half) so the x2 step (latent upscaler on 2.3, lanczos pixel resize on
    2.5) lands on stage 2. ``ltx25`` keeps official blend dilations and, for
    outpaint, swaps ``LTXVImgToVideoInplace`` to ``LTXVImgToVideoConditionOnly``.
    """
    stage_w, stage_h = fit_working_size(int(width), int(height))
    stage1_w, stage1_h = stage_w // 2, stage_h // 2
    outpaint = (mode or "inpaint").lower() == "outpaint"
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 1"), "width", stage1_w)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 1"), "height", stage1_h)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2"), "width", stage_w)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2"), "height", stage_h)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 1"), "width", stage1_w)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 1"), "height", stage1_h)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 2"), "width", stage_w)
    _set(_by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 2"), "height", stage_h)
    for _nid, node in _nodes(workflow, "EmptyLTXVLatentVideo"):
        inputs = node.setdefault("inputs", {})
        inputs["width"] = stage1_w
        inputs["height"] = stage1_h
    if outpaint:
        d1 = d2 = OUTPAINT_DILATE
        # 2.5 keeps the official blend dilations (5 then 6) in both modes.
        # 2.3 outpaint still uses the smaller blend from #41.
        blend = INPAINT_BLEND_DILATION if ltx25 else OUTPAINT_BLEND_DILATION
    else:
        d1, d2 = INPAINT_DILATE_STAGE1, INPAINT_DILATE_STAGE2
        blend = INPAINT_BLEND_DILATION
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 1"), "spatial_radius", d1)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 1"), "temporal_radius", 0)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 2"), "spatial_radius", d2)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 2"), "temporal_radius", 0)
    if ltx25:
        _set(
            _by_title(workflow, "LTXVLaplacianPyramidBlend", "Blend Stage 1"),
            "mask_low_res_dilation",
            5,
        )
    _set(
        _by_title(workflow, "LTXVLaplacianPyramidBlend", "Blend Stage 2"),
        "mask_low_res_dilation",
        blend,
    )
    if ltx25 and outpaint:
        # Official 2.5 outpaint uses ConditionOnly. Inpaint stays Inplace.
        for _nid, node in _nodes(workflow, "LTXVImgToVideoInplace"):
            node["class_type"] = "LTXVImgToVideoConditionOnly"
    pads = pad or {}
    pad_node = _by_title(workflow, "ImagePadForOutpaint", "Outpaint Pad")
    if pad_node is None:
        found = _nodes(workflow, "ImagePadForOutpaint")
        pad_node = found[0][1] if found else None
    for key in ("left", "top", "right", "bottom"):
        _set(pad_node, key, int(pads.get(key, 0) or 0))
    _set(pad_node, "feathering", 0)
    if mask_png:
        for _nid, node in _nodes(workflow, "LoadImageMask"):
            meta = node.setdefault("_meta", {})
            meta["inoutpaint_png_b64"] = base64.b64encode(mask_png).decode("ascii")


def _mask_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for _nid, node in _nodes(workflow, "LoadImageMask")]


def mask_upload_name(workflow: dict[str, Any]) -> str:
    """2.3 and 2.5 keep separate upload names so a shared queue does not collide."""
    for _nid, node in _nodes(workflow, "SaveVideo"):
        prefix = str((node.get("inputs") or {}).get("filename_prefix") or "")
        if "ltx25" in prefix:
            return LTX25_MASK_UPLOAD_NAME
    return MASK_UPLOAD_NAME


def _pop_mask_png(workflow: dict[str, Any]) -> bytes | None:
    raw: str | None = None
    for node in _mask_nodes(workflow):
        meta = node.get("_meta") or {}
        stashed = meta.pop("inoutpaint_png_b64", None)
        if stashed and raw is None:
            raw = str(stashed)
    if not raw:
        return None
    try:
        return base64.b64decode(raw)
    except (ValueError, TypeError):
        return None


def _pop_local_video(workflow: dict[str, Any]) -> Path | None:
    found: str | None = None
    for _nid, node in _nodes(workflow, "LoadVideo"):
        meta = node.get("_meta") or {}
        local = meta.pop("local_path", None)
        if local and found is None:
            found = str(local)
    if not found:
        return None
    path = Path(found)
    return path if path.is_file() else None


def workflow_needs_generated_mask(workflow: dict[str, Any]) -> bool:
    for node in _mask_nodes(workflow):
        image = str((node.get("inputs") or {}).get("image") or "")
        if image in ("", MASK_PLACEHOLDER):
            return True
    return False


def ensure_inoutpaint_mask(
    workflow: dict[str, Any],
    upload: Callable[[Path], str],
) -> str | None:
    """Upload a generated mask when LoadImageMask still names the placeholder.

    A caller-supplied filename is left alone. Stashed outpaint PNG bytes (set
    by ``finalize_inoutpaint_graph``) are consumed and removed before queue.
    """
    png = _pop_mask_png(workflow)
    if not workflow_needs_generated_mask(workflow):
        return None
    if png is None:
        stage2 = _by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2")
        inputs = (stage2 or {}).get("inputs") or {}
        try:
            width = int(inputs.get("width") or 64)
            height = int(inputs.get("height") or 64)
        except (TypeError, ValueError):
            width, height = 64, 64
        png = default_inpaint_mask_png(width, height)
    upload_name = mask_upload_name(workflow)
    with tempfile.TemporaryDirectory(prefix="ltx-inoutpaint-") as tmp:
        path = Path(tmp) / upload_name
        path.write_bytes(png)
        uploaded = upload(path) or upload_name
    name = str(uploaded)
    for node in _mask_nodes(workflow):
        inputs = node.setdefault("inputs", {})
        image = str(inputs.get("image") or "")
        if image in ("", MASK_PLACEHOLDER):
            inputs["image"] = name
    return name


def prepare_queue_inputs(
    workflow: dict[str, Any],
    upload: Callable[[Path], str],
) -> str | None:
    """Upload a stashed source video, then the in/outpaint mask. Before lint."""
    local = _pop_local_video(workflow)
    if local is not None:
        name = upload(local) or local.name
        for _nid, node in _nodes(workflow, "LoadVideo"):
            node.setdefault("inputs", {})["file"] = str(name)
    local_mask = _pop_local_mask(workflow)
    if local_mask is not None:
        _pop_mask_png(workflow)
        name = upload(local_mask) or local_mask.name
        for node in _mask_nodes(workflow):
            node.setdefault("inputs", {})["image"] = str(name)
        return str(name)
    return ensure_inoutpaint_mask(workflow, upload)


def stash_local_video(workflow: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        return
    for _nid, node in _nodes(workflow, "LoadVideo"):
        node.setdefault("_meta", {})["local_path"] = str(file_path)


def stash_local_mask(workflow: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        return
    for node in _mask_nodes(workflow):
        node.setdefault("_meta", {})["local_mask_path"] = str(file_path)


def _pop_local_mask(workflow: dict[str, Any]) -> Path | None:
    found: str | None = None
    for node in _mask_nodes(workflow):
        meta = node.get("_meta") or {}
        local = meta.pop("local_mask_path", None)
        if local and found is None:
            found = str(local)
    if not found:
        return None
    path = Path(found)
    return path if path.is_file() else None


def record_comfy_provenance(
    workflow: dict[str, Any],
    video_path: str | Path,
    *,
    variant: str | None,
) -> dict[str, Any] | None:
    """Write ``buddy.clip.provenance/v1`` next to a comfy-run render."""
    if (variant or "") not in PROVENANCE_VARIANTS:
        return None
    from types import SimpleNamespace

    from master_agent.provenance import build_clip_provenance, write_clip_provenance

    positive = ""
    negative = ""
    for _nid, node in _nodes(workflow, "CLIPTextEncode"):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        text = str((node.get("inputs") or {}).get("text") or "")
        if "neg" in title:
            negative = text
        elif not positive:
            positive = text
    seed = None
    noises = _nodes(workflow, "RandomNoise")
    if noises:
        raw_seed = (noises[0][1].get("inputs") or {}).get("noise_seed")
        if isinstance(raw_seed, int):
            seed = raw_seed
    stage2 = _by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2")
    stage_in = (stage2 or {}).get("inputs") or {}
    width = int(stage_in.get("width") or DEFAULT_WIDTH)
    height = int(stage_in.get("height") or DEFAULT_HEIGHT)
    frames = DEFAULT_FRAMES
    latents = _nodes(workflow, "EmptyLTXVLatentVideo")
    if latents:
        raw_len = (latents[0][1].get("inputs") or {}).get("length")
        if isinstance(raw_len, int):
            frames = raw_len
    fps = DEFAULT_FPS
    videos = _nodes(workflow, "LoadVideo")
    video_name = ""
    if videos:
        video_name = str((videos[0][1].get("inputs") or {}).get("file") or "")
    masks = _nodes(workflow, "LoadImageMask")
    mask_name = ""
    if masks:
        mask_name = str((masks[0][1].get("inputs") or {}).get("image") or "")
    state = SimpleNamespace(
        request=positive,
        prompt=positive,
        negative_prompt=negative,
        variant=variant,
        seed=seed,
        steps=8,
        cfg=1.0,
        width=width,
        height=height,
        fps=fps,
        duration_s=frames / float(fps),
        video_name=video_name or None,
        mask_name=mask_name or None,
        attempt=1,
        shot_index=1,
        judge_score=0.0,
        judge_issues=[],
        revise_history=[],
        shot=None,
    )
    payload = build_clip_provenance(state, path=video_path)
    write_clip_provenance(video_path, payload)
    return payload
