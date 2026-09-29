"""LTX 2.3 inpaint / outpaint layout, mask PNG, and queue-time upload.

The shipped graph is ``workflows/ltx23_inoutpaint_api.json``. White mask pixels
are the region the IC-LoRA regenerates. Black pixels are held. Outpaint grows
the canvas with ``ImagePadForOutpaint`` (pads are multiples of 8, the node's
step). The sampler mask for outpaint is that node's own mask so the green
plate lines up with the pad. Inpaint repeats the still mask to the latent
length before the blend, which keeps ``trim_to_shortest`` and still emits
every frame.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

VARIANT = "ltx23_inoutpaint"
MASK_PLACEHOLDER = "example.png"
MASK_UPLOAD_NAME = "ltx23_inoutpaint_mask.png"
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
# stays on wan_fun_inpaint.
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


def _nid_by_title(workflow: dict[str, Any], class_type: str, title: str) -> str | None:
    want = title.lower()
    for nid, node in _nodes(workflow, class_type):
        got = str((node.get("_meta") or {}).get("title") or "").lower()
        if got == want:
            return nid
    return None


def _widget_int(node: dict[str, Any] | None, key: str) -> int | None:
    if node is None:
        return None
    raw = (node.get("inputs") or {}).get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return int(raw)


def _resize_size(inputs: dict[str, Any], key: str, default: int) -> int:
    """Read ``resize_type.width`` / ``height`` (Comfy 0.35), then a plain widget."""
    raw = inputs.get(f"resize_type.{key}")
    if raw is None:
        raw = inputs.get(key)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class SourceVideoTrimError(RuntimeError):
    """The source clip could not be cut to the latent frame count."""


def _write_scale_dimensions(node: dict[str, Any] | None, width: int, height: int) -> None:
    """Exact stage size. ``crop=disabled`` stretches onto the latent grid.

    ComfyUI 0.35 names these ``resize_type.width`` / ``height`` / ``crop``.
    Plain ``width`` / ``height`` fail prompt validation.
    """
    if node is None:
        return
    inputs = node.setdefault("inputs", {})
    inputs.pop("width", None)
    inputs.pop("height", None)
    inputs["resize_type"] = "scale dimensions"
    inputs["resize_type.width"] = int(width)
    inputs["resize_type.height"] = int(height)
    inputs["resize_type.crop"] = "disabled"
    inputs["scale_method"] = "lanczos"


def _write_match_size(node: dict[str, Any] | None, match: list[Any]) -> None:
    """Fit the mask to the already-resized frames (official IC-LoRA graph)."""
    if node is None:
        return
    inputs = node.setdefault("inputs", {})
    inputs.pop("width", None)
    inputs.pop("height", None)
    inputs.pop("resize_type.width", None)
    inputs.pop("resize_type.height", None)
    inputs["resize_type"] = "match size"
    inputs["resize_type.match"] = match
    inputs["resize_type.crop"] = "center"
    inputs["scale_method"] = "area"


def finalize_inoutpaint_graph(
    workflow: dict[str, Any],
    *,
    width: int,
    height: int,
    mode: str = "inpaint",
    pad: dict[str, int] | None = None,
    mask_png: bytes | None = None,
) -> None:
    """Write stage sizes, dilate/blend, and outpaint pads after the generic patcher.

    ``width`` / ``height`` are the stage-2 output. The empty latent is stage 1
    (half) so ``LTXVLatentUpsampler`` lands on stage 2. Masks are match-sized
    to those frames so a still mask cannot drift off the plate. Spatial dilate
    stays the official 15 / 30 (outpaint 0) and the blend stays 6 / 2.
    """
    stage_w, stage_h = fit_working_size(int(width), int(height))
    stage1_w, stage1_h = stage_w // 2, stage_h // 2
    outpaint = (mode or "inpaint").lower() == "outpaint"
    _write_scale_dimensions(
        _by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 1"),
        stage1_w,
        stage1_h,
    )
    _write_scale_dimensions(
        _by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2"),
        stage_w,
        stage_h,
    )
    frames1 = _nid_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 1")
    frames2 = _nid_by_title(workflow, "ResizeImageMaskNode", "Resize Frames Stage 2")
    if frames1:
        _write_match_size(
            _by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 1"),
            [frames1, 0],
        )
    if frames2:
        _write_match_size(
            _by_title(workflow, "ResizeImageMaskNode", "Resize Mask Stage 2"),
            [frames2, 0],
        )
    for _nid, node in _nodes(workflow, "EmptyLTXVLatentVideo"):
        inputs = node.setdefault("inputs", {})
        inputs["width"] = stage1_w
        inputs["height"] = stage1_h
    if outpaint:
        d1 = d2 = OUTPAINT_DILATE
        blend = OUTPAINT_BLEND_DILATION
    else:
        d1, d2 = INPAINT_DILATE_STAGE1, INPAINT_DILATE_STAGE2
        blend = INPAINT_BLEND_DILATION
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 1"), "spatial_radius", d1)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 1"), "temporal_radius", 0)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 2"), "spatial_radius", d2)
    _set(_by_title(workflow, "LTXVDilateVideoMask", "Dilate Mask Stage 2"), "temporal_radius", 0)
    _set(
        _by_title(workflow, "LTXVLaplacianPyramidBlend", "Blend Stage 2"),
        "mask_low_res_dilation",
        blend,
    )
    pads = pad or {}
    pad_id = _nid_by_title(workflow, "ImagePadForOutpaint", "Outpaint Pad")
    pad_node = workflow.get(pad_id) if pad_id else None
    if not isinstance(pad_node, dict):
        found = _nodes(workflow, "ImagePadForOutpaint")
        if found:
            pad_id, pad_node = found[0]
        else:
            pad_id, pad_node = None, None
    for key in ("left", "top", "right", "bottom"):
        _set(pad_node, key, int(pads.get(key, 0) or 0))
    _set(pad_node, "feathering", 0)
    # Official guide leaves attention_mask disconnected and encodes the green
    # plate in one pass. Tiled encode was tinting the kept strip.
    for _nid, node in _nodes(workflow, "LTXAddVideoICLoRAGuideAdvanced"):
        inputs = node.setdefault("inputs", {})
        inputs.pop("attention_mask", None)
        inputs["use_tiled_encode"] = False
    for _nid, node in _nodes(workflow, "LTXVTiledVAEDecode"):
        _set(node, "overlap", 6)
    length = DEFAULT_FRAMES
    latents = _nodes(workflow, "EmptyLTXVLatentVideo")
    if latents:
        raw_len = _widget_int(latents[0][1], "length")
        if raw_len:
            length = raw_len
    _set(_by_title(workflow, "RepeatImageBatch", "Repeat Inpaint Mask"), "amount", length)
    if outpaint and pad_id:
        # The pad node mask is 1 on the new border and 0 on the source, one
        # entry per frame. A separately painted still can miss that edge.
        for title in ("Resize Mask Stage 1", "Resize Mask Stage 2"):
            _set(
                _by_title(workflow, "ResizeImageMaskNode", title),
                "input",
                [pad_id, 1],
            )
    if mask_png:
        for _nid, node in _nodes(workflow, "LoadImageMask"):
            meta = node.setdefault("_meta", {})
            meta["inoutpaint_png_b64"] = base64.b64encode(mask_png).decode("ascii")


def _mask_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for _nid, node in _nodes(workflow, "LoadImageMask")]


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
        width = _resize_size(inputs, "width", 64)
        height = _resize_size(inputs, "height", 64)
        png = default_inpaint_mask_png(width, height)
    with tempfile.TemporaryDirectory(prefix="ltx23-inoutpaint-") as tmp:
        path = Path(tmp) / MASK_UPLOAD_NAME
        path.write_bytes(png)
        uploaded = upload(path) or MASK_UPLOAD_NAME
    name = str(uploaded)
    for node in _mask_nodes(workflow):
        inputs = node.setdefault("inputs", {})
        image = str(inputs.get("image") or "")
        if image in ("", MASK_PLACEHOLDER):
            inputs["image"] = name
    return name


def _ffprobe_video_stream(path: Path, *, count_frames: bool) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        cmd.append("-count_frames")
    cmd += [
        "-show_entries",
        "stream=nb_frames,nb_read_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SourceVideoTrimError("ffprobe is not on PATH; cannot read the source frame count") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SourceVideoTrimError(f"ffprobe failed for {path.name}: {detail}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SourceVideoTrimError(f"ffprobe returned no JSON for {path.name}") from exc
    streams = data.get("streams") or []
    return streams[0] if streams and isinstance(streams[0], dict) else {}


def _stream_int(stream: dict[str, Any], key: str) -> int:
    raw = stream.get(key)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 0


def _frame_rate(text: Any) -> float:
    raw = str(text or "")
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            denom = float(den)
        except ValueError:
            return 0.0
        if denom == 0:
            return 0.0
        try:
            return float(num) / denom
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def probe_video_frame_count(path: Path) -> int:
    """Decoded frame count of the first video stream."""
    stream = _ffprobe_video_stream(path, count_frames=False)
    frames = _stream_int(stream, "nb_frames")
    if frames:
        return frames
    stream = _ffprobe_video_stream(path, count_frames=True)
    frames = _stream_int(stream, "nb_read_frames") or _stream_int(stream, "nb_frames")
    if frames:
        return frames
    try:
        duration = float(stream.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    rate = _frame_rate(stream.get("avg_frame_rate"))
    if duration > 0 and rate > 0:
        return max(int(round(duration * rate)), 1)
    return 0


def trim_video_command(src: Path, frames: int, dest: Path) -> list[str]:
    """First ``frames`` pictures, with audio when the source has any."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-frames:v",
        str(int(frames)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(dest),
    ]


def trim_video_to_frames(src: Path, frames: int, dest: Path) -> None:
    cmd = trim_video_command(src, frames, dest)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SourceVideoTrimError("ffmpeg is not on PATH; cannot trim the source video") from exc
    if proc.returncode != 0 or not dest.is_file():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise SourceVideoTrimError(f"ffmpeg failed to trim {src.name} to {frames} frames: {tail}")


def _latent_frame_count(workflow: dict[str, Any]) -> int | None:
    latents = _nodes(workflow, "EmptyLTXVLatentVideo")
    if not latents:
        return None
    return _widget_int(latents[0][1], "length")


def fit_source_video(path: Path, frames: int) -> Path:
    """Cut a longer source down to the requested 8n+1 latent length.

    ``LTXAddVideoICLoRAGuideAdvanced`` asserts when the encoded guide is
    longer than ``EmptyLTXVLatentVideo``. LoadVideo in this Comfy build has
    no frame cap, so the file itself has to be the right length.
    """
    target = int(frames)
    if target < 1:
        raise SourceVideoTrimError(f"latent length {frames} is not a frame count")
    count = probe_video_frame_count(path)
    if count <= 0:
        raise SourceVideoTrimError(f"could not read a frame count for {path.name}")
    if count <= target:
        return path
    dest_dir = Path(tempfile.mkdtemp(prefix="ltx23-trim-"))
    suffix = path.suffix if path.suffix else ".mp4"
    dest = dest_dir / f"{path.stem}_{target}f{suffix}"
    trim_video_to_frames(path, target, dest)
    got = probe_video_frame_count(dest)
    if got != target:
        raise SourceVideoTrimError(
            f"trimmed {path.name} has {got} frames; the latent length is {target}"
        )
    return dest


def _is_inoutpaint_graph(workflow: dict[str, Any]) -> bool:
    return bool(_nodes(workflow, "LTXVInpaintPreprocess"))


def prepare_queue_inputs(
    workflow: dict[str, Any],
    upload: Callable[[Path], str],
) -> str | None:
    """Upload a stashed source video, then the in/outpaint mask. Before lint."""
    inoutpaint = _is_inoutpaint_graph(workflow)
    target = _latent_frame_count(workflow) if inoutpaint else None
    if target:
        _set(_by_title(workflow, "RepeatImageBatch", "Repeat Inpaint Mask"), "amount", target)
    local = _pop_local_video(workflow)
    if local is not None:
        if target:
            local = fit_source_video(local, target)
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
    if (variant or "") != VARIANT:
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
    width = _resize_size(stage_in, "width", DEFAULT_WIDTH)
    height = _resize_size(stage_in, "height", DEFAULT_HEIGHT)
    frames = DEFAULT_FRAMES
    latents = _nodes(workflow, "EmptyLTXVLatentVideo")
    if latents:
        raw_len = _widget_int(latents[0][1], "length")
        if raw_len:
            frames = raw_len
    fps = DEFAULT_FPS
    duration_s = frames / float(fps)
    output_frames: int | None = None
    try:
        from master_agent.judge.probe import probe_video

        probed = probe_video(video_path)
    except Exception:
        probed = None
    if isinstance(probed, dict):
        probed_frames = probed.get("frames")
        probed_duration = probed.get("duration_s")
        if isinstance(probed_frames, int) and not isinstance(probed_frames, bool) and probed_frames > 0:
            output_frames = probed_frames
            frames = probed_frames
        if isinstance(probed_duration, (int, float)) and not isinstance(probed_duration, bool):
            if float(probed_duration) > 0:
                duration_s = float(probed_duration)
        elif output_frames:
            duration_s = output_frames / float(fps)
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
        variant=VARIANT,
        seed=seed,
        steps=8,
        cfg=1.0,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
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
    if output_frames is not None:
        params = payload.get("params")
        if isinstance(params, dict):
            params["frames"] = output_frames
            params["duration"] = duration_s
    write_clip_provenance(video_path, payload)
    return payload
