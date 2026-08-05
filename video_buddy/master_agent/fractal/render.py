"""Vectorized numpy Mandelbrot/Julia deep-zoom renderer -> ffmpeg H.264 pipe.

Deterministic, CPU-only, no ComfyUI/GPU needed:
  - smooth iteration counts (continuous coloring n + 1 - log2(log|z|))
  - 2x supersampled grid, box-downsampled for anti-aliasing
  - exponential zoom schedule toward curated targets (float64 depth cap ~1e10)
  - 5 named palettes as 256-entry LUTs, optional per-frame cycling
  - frames piped raw RGB24 into ffmpeg stdin -> H.264 mp4
  - optional beat-reactive mode driven by a music.beats.BeatMap
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

ESCAPE_RADIUS = 2.0
MAX_ZOOM_DEPTH = 1e10  # float64 stays crisp to about this magnification

# Curated deep-zoom targets: (center_re, center_im, start_half_width)
TARGETS: dict[str, tuple[float, float, float]] = {
    "seahorse": (-0.745, 0.113, 1.6),
    "elephant": (0.286, 0.011, 1.6),
    "minibrot": (-1.768778833, 0.001738996, 1.6),
    "spiral": (-0.77568377, 0.13646737, 1.6),
}

# Palette control points (position 0..1 -> RGB); LUTs are 256x3 uint8.
_PALETTE_STOPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "fire": [
        (0.00, (0, 0, 0)),
        (0.35, (120, 8, 0)),
        (0.65, (230, 90, 0)),
        (0.85, (255, 200, 40)),
        (1.00, (255, 255, 235)),
    ],
    "ocean": [
        (0.00, (0, 2, 12)),
        (0.40, (0, 40, 110)),
        (0.70, (0, 130, 180)),
        (0.90, (90, 220, 230)),
        (1.00, (240, 255, 255)),
    ],
    "monochrome": [
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ],
    "neon": [
        (0.00, (10, 0, 30)),
        (0.25, (190, 0, 255)),
        (0.50, (0, 255, 200)),
        (0.75, (255, 230, 0)),
        (1.00, (255, 0, 120)),
    ],
    "sunset": [
        (0.00, (20, 0, 40)),
        (0.35, (110, 20, 90)),
        (0.65, (230, 90, 60)),
        (0.85, (255, 180, 90)),
        (1.00, (255, 245, 220)),
    ],
}


def build_palettes() -> dict[str, np.ndarray]:
    """Interpolate the control points into 256x3 uint8 LUTs."""
    luts: dict[str, np.ndarray] = {}
    for name, stops in _PALETTE_STOPS.items():
        pos = np.array([s[0] for s in stops])
        cols = np.array([s[1] for s in stops], dtype=np.float64)
        grid = np.linspace(0.0, 1.0, 256)
        lut = np.stack(
            [np.interp(grid, pos, cols[:, ch]) for ch in range(3)], axis=1
        )
        luts[name] = np.clip(lut, 0, 255).astype(np.uint8)
    return luts


PALETTES = build_palettes()
INTERIOR_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def mandelbrot_smooth(
    cre: np.ndarray, cim: np.ndarray, max_iter: int = 256
) -> np.ndarray:
    """Smooth (continuous) iteration counts. Interior pixels == max_iter.

    Iterates only the not-yet-escaped subset — most pixels bail early, so the
    per-iteration cost collapses after the first few dozen steps.
    """
    cr = cre.ravel().astype(np.float64)
    ci = cim.ravel().astype(np.float64)
    counts = np.full(cr.shape, float(max_iter), dtype=np.float64)
    act = np.arange(cr.size)
    zr = np.zeros(cr.size, dtype=np.float64)
    zi = np.zeros(cr.size, dtype=np.float64)
    zr2 = np.zeros(cr.size, dtype=np.float64)
    zi2 = np.zeros(cr.size, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(max_iter):
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
            zr2 = zr * zr
            zi2 = zi * zi
            mag = zr2 + zi2
            esc = mag > ESCAPE_RADIUS * ESCAPE_RADIUS
            if esc.any():
                log_zn = 0.5 * np.log(mag[esc])
                counts[act[esc]] = n + 1 - np.log2(np.maximum(log_zn, 1e-12))
                keep = ~esc
                act = act[keep]
                if act.size == 0:
                    break
                cr, ci = cr[keep], ci[keep]
                zr, zi = zr[keep], zi[keep]
                zr2, zi2 = zr2[keep], zi2[keep]
    return counts.reshape(cre.shape)


def julia_smooth(
    zre: np.ndarray,
    zim: np.ndarray,
    c_re: float,
    c_im: float,
    max_iter: int = 256,
) -> np.ndarray:
    """Smooth iteration counts for the Julia set of c; z starts at the grid."""
    zr = zre.ravel().astype(np.float64).copy()
    zi = zim.ravel().astype(np.float64).copy()
    counts = np.full(zr.shape, float(max_iter), dtype=np.float64)
    act = np.arange(zr.size)
    zr2 = zr * zr
    zi2 = zi * zi
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(max_iter):
            zi = 2.0 * zr * zi + c_im
            zr = zr2 - zi2 + c_re
            zr2 = zr * zr
            zi2 = zi * zi
            mag = zr2 + zi2
            esc = mag > ESCAPE_RADIUS * ESCAPE_RADIUS
            if esc.any():
                log_zn = 0.5 * np.log(mag[esc])
                counts[act[esc]] = n + 1 - np.log2(np.maximum(log_zn, 1e-12))
                keep = ~esc
                act = act[keep]
                if act.size == 0:
                    break
                zr, zi = zr[keep], zi[keep]
                zr2, zi2 = zr2[keep], zi2[keep]
    return counts.reshape(zre.shape)


def colorize(
    counts: np.ndarray,
    lut: np.ndarray,
    *,
    max_iter: int,
    cycle: float = 0.0,
    scale: float = 8.0,
    interior: np.ndarray = INTERIOR_COLOR,
) -> np.ndarray:
    """Map smooth counts through a palette LUT -> HxWx3 uint8."""
    inside = counts >= max_iter  # interior pixels were left at float(max_iter)
    idx = np.floor(counts * scale + cycle).astype(np.int64) % lut.shape[0]
    rgb = lut[idx]
    rgb[inside] = interior
    return rgb


def render_frame(
    center: tuple[float, float],
    half_w: float,
    *,
    width: int,
    height: int,
    max_iter: int = 256,
    lut: np.ndarray | None = None,
    cycle: float = 0.0,
    julia_c: tuple[float, float] | None = None,
    ss: int = 2,
) -> np.ndarray:
    """One anti-aliased RGB frame (HxWx3 uint8), ss x supersampled."""
    lut = lut if lut is not None else PALETTES["fire"]
    half_h = half_w * height / width
    sw, sh = width * ss, height * ss
    xs = np.linspace(center[0] - half_w, center[0] + half_w, sw)
    ys = np.linspace(center[1] - half_h, center[1] + half_h, sh)
    cre, cim = np.meshgrid(xs, ys)
    if julia_c is None:
        counts = mandelbrot_smooth(cre, cim, max_iter)
    else:
        counts = julia_smooth(cre, cim, julia_c[0], julia_c[1], max_iter)
    rgb = colorize(counts, lut, max_iter=max_iter, cycle=cycle)
    if ss > 1:
        rgb = (
            rgb.reshape(sh // ss, ss, sw // ss, ss, 3).mean(axis=(1, 3)).astype(np.uint8)
        )
    return rgb


class FfmpegEncoder:
    """Pipe raw RGB24 frames into ffmpeg -> H.264 mp4."""

    def __init__(self, dest: Path, width: int, height: int, fps: int, crf: int = 18):
        from master_agent.video_concat import find_ffmpeg

        ff = find_ffmpeg()
        if not ff:
            raise RuntimeError("ffmpeg not found on PATH; cannot encode fractal video")
        self.dest = Path(dest)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ff, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(self.dest),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> Path:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        rc = self.proc.wait()
        if rc != 0 or not self.dest.is_file():
            raise RuntimeError(f"ffmpeg encode failed (rc={rc}): {err[-500:]}")
        return self.dest

    def __enter__(self) -> "FfmpegEncoder":
        return self

    def __exit__(self, *exc) -> None:
        if self.proc.poll() is None:
            self.close()


def _beat_pulse(t: float, beats: list[float], decay: float = 6.0) -> float:
    """0..1 pulse that spikes at each beat and decays exponentially."""
    if not beats:
        return 0.0
    # last beat at or before t
    import bisect

    i = bisect.bisect_right(beats, t)
    if i == 0:
        return 0.0
    dt = t - beats[i - 1]
    return math.exp(-decay * dt)


def zoom_frames(
    *,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    target: str = "seahorse",
    palette: str = "fire",
    max_iter: int = 256,
    zoom_secs_per_double: float = 4.0,
    beat_map=None,
    julia: bool = False,
    seed: int | None = None,
):
    """Yield (frame_index, HxWx3 uint8). Exponential zoom; optional beat-reactive.

    Beat-reactive mode: zoom speed eases into each beat (pulse), the palette
    flashes on onsets (cycle jump), and Julia's c wobbles with the pulse.
    """
    rng = np.random.default_rng(seed)
    if julia:
        center = (0.0, 0.0)
        half_w0 = 1.6
        base_c = TARGETS.get(target, TARGETS["seahorse"])[:2]
    else:
        c_re, c_im, half_w0 = TARGETS.get(target, TARGETS["seahorse"])
        center = (c_re, c_im)
        base_c = None
    lut = PALETTES.get(palette, PALETTES["fire"])
    n_frames = max(1, int(round(duration_s * fps)))
    beats = list(getattr(beat_map, "beats", []) or [])
    env = list(getattr(beat_map, "onset_env", []) or [])
    env_fps = float(getattr(beat_map, "env_fps", 43.0))

    half_w = half_w0
    rate = 0.5 ** (1.0 / (zoom_secs_per_double * fps))  # per-frame multiplier
    cycle = 0.0
    min_half_w = half_w0 / MAX_ZOOM_DEPTH
    wobble_phase = float(rng.uniform(0, 2 * math.pi))

    for f in range(n_frames):
        t = f / fps
        pulse = _beat_pulse(t, beats) if beats else 0.0
        # ease zoom into beats: momentary boost then return to base rate
        frame_rate = rate ** (1.0 + 2.0 * pulse)
        half_w = max(min_half_w, half_w * frame_rate)
        cycle += 1.5 + 14.0 * pulse  # palette flash on onsets
        jc = None
        if julia and base_c is not None:
            energy = 0.0
            if env:
                ei = min(int(t * env_fps / 8), len(env) - 1)  # env decimated x8
                energy = min(env[ei] / 3.0, 1.0)
            wob = 0.02 + 0.10 * pulse + 0.05 * energy
            ang = wobble_phase + t * 0.35
            jc = (base_c[0] + wob * math.cos(ang), base_c[1] + wob * math.sin(ang))
        yield f, render_frame(
            center,
            half_w,
            width=width,
            height=height,
            max_iter=max_iter,
            lut=lut,
            cycle=cycle,
            julia_c=jc if julia else None,
        )


def render_zoom_video(
    dest: Path,
    *,
    duration_s: float,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    target: str = "seahorse",
    palette: str = "fire",
    max_iter: int = 256,
    beat_map=None,
    julia: bool = False,
    seed: int | None = None,
    log=print,
) -> Path:
    """Render the full zoom to an H.264 mp4."""
    width = max(2, int(width) & ~1)
    height = max(2, int(height) & ~1)
    with FfmpegEncoder(dest, width, height, fps) as enc:
        for f, frame in zoom_frames(
            duration_s=duration_s, fps=fps, width=width, height=height,
            target=target, palette=palette, max_iter=max_iter,
            beat_map=beat_map, julia=julia, seed=seed,
        ):
            enc.write(frame)
            if f % max(1, fps * 2) == 0:
                log(f"fractal frame {f}/{int(duration_s * fps)}")
    log(f"fractal video: {enc.dest}")
    return enc.dest


# ── inpaint / outpaint (image + fractal fill) ─────────────

MODES = ("zoom", "inpaint", "outpaint")


def ensure_even_size(arr: np.ndarray) -> np.ndarray:
    """libx264 / yuv420p needs even width and height — crop 1px if odd.

    Works for HxW masks and HxWxC images.
    """
    h, w = arr.shape[:2]
    nh, nw = max(h - (h % 2), 2), max(w - (w % 2), 2)
    if nh == h and nw == w:
        return arr
    return arr[:nh, :nw]


def load_rgb(path: str | Path, *, max_side: int = 1280) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB, optionally downscaling long side."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    # even dims for H.264 later
    w, h = img.size
    if w % 2 or h % 2:
        img = img.crop((0, 0, w - (w % 2), h - (h % 2)))
    return np.asarray(img, dtype=np.uint8)


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a mask image (white/light = fill) resized to (H, W), float 0..1."""
    from PIL import Image

    h, w = shape
    m = Image.open(path).convert("L").resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def feather_binary_mask(hard: np.ndarray, radius: int) -> np.ndarray:
    """Soft-edge a binary mask (1=fill) with a cheap box-blur ramp.

    Pure numpy — no scipy. ``radius`` is in pixels; 0 returns hard edges.
    Keeps strong fill in the interior while ramping only near the boundary.
    """
    hard = np.clip(hard.astype(np.float32), 0.0, 1.0)
    if radius <= 0:
        return hard
    out = hard.copy()
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k /= k.sum()
    passes = max(1, int(radius))
    for _ in range(passes):
        pad = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = k[0] * pad[:, 0:-2] + k[1] * pad[:, 1:-1] + k[2] * pad[:, 2:]
        pad = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = k[0] * pad[0:-2, :] + k[1] * pad[1:-1, :] + k[2] * pad[2:, :]
    # Interior of the hard mask stays high; edge ramps via blur
    return np.clip(np.where(hard > 0.5, np.maximum(out, hard), out), 0.0, 1.0)


def soft_mask_from_gray(gray: np.ndarray, *, feather: int = 0) -> np.ndarray:
    """Turn a continuous mask (0..1, white=fill) into a soft alpha.

    If feather > 0, slightly dilate/blur the mid-tones so hard paint masks
    don't leave a jagged cut.
    """
    m = np.clip(gray.astype(np.float32), 0.0, 1.0)
    if feather <= 0:
        return m
    hard = (m > 0.5).astype(np.float32)
    soft = feather_binary_mask(hard, feather)
    # Prefer the smoother of continuous gray vs feathered hard edge
    return np.clip(np.maximum(m * 0.55 + soft * 0.45, m * soft), 0.0, 1.0)


def center_hole_mask(
    height: int,
    width: int,
    *,
    cover: float = 0.4,
    feather: int = 24,
    shape: str = "ellipse",
) -> np.ndarray:
    """Soft hole in the center for inpaint (1=fill with fractal).

    Uses a smooth radial (or rect) falloff so the seam doesn't look cut out.
    """
    cover = float(np.clip(cover, 0.05, 0.95))
    feather = max(0, int(feather))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    if shape == "rect":
        half_h = max(height * cover / 2.0, 1.0)
        half_w = max(width * cover / 2.0, 1.0)
        # chebyshev distance outside the rect, 0 inside
        dy = np.maximum(np.abs(yy - cy) - half_h, 0.0)
        dx = np.maximum(np.abs(xx - cx) - half_w, 0.0)
        dist = np.maximum(dx, dy)
        # inside = full fill; ramp out over feather px
        if feather <= 0:
            return (dist <= 0).astype(np.float32)
        return np.clip(1.0 - dist / float(feather), 0.0, 1.0).astype(np.float32)

    ry = max(height * cover / 2.0, 1.0)
    rx = max(width * cover / 2.0, 1.0)
    # normalized radius (1.0 = hole edge)
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    if feather <= 0:
        return (r <= 1.0).astype(np.float32)
    # soft band: full fill until ~inner, ramp across feather as fraction of radius
    band = max(feather / max(min(rx, ry), 1.0), 0.05)
    inner, outer = 1.0 - band, 1.0 + band * 0.35
    return np.clip((outer - r) / max(outer - inner, 1e-6), 0.0, 1.0).astype(np.float32)


def _edge_extend_canvas(image: np.ndarray, expand: int) -> np.ndarray:
    """Pad with edge-replicated pixels so soft outpaint blends into content."""
    h, w = image.shape[:2]
    canvas = np.zeros((h + 2 * expand, w + 2 * expand, 3), dtype=np.uint8)
    canvas[expand : expand + h, expand : expand + w] = image
    # top / bottom strips
    canvas[:expand, expand : expand + w] = image[0:1, :, :]
    canvas[expand + h :, expand : expand + w] = image[-1:, :, :]
    # left / right strips (including corners from already-filled edges)
    canvas[:, :expand] = canvas[:, expand : expand + 1]
    canvas[:, expand + w :] = canvas[:, expand + w - 1 : expand + w]
    return canvas


def prepare_outpaint(
    image: np.ndarray,
    *,
    expand: int = 128,
    feather: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad image on all sides; return (canvas RGB, fill-mask float 0..1).

    Canvas is edge-extended (not black) so the feathered fractal seam looks
    natural. Mask ramps from 0 at the photo edge to 1 over ``feather`` px.
    """
    expand = max(0, int(expand))
    feather = max(0, int(feather))
    if expand == 0:
        h, w = image.shape[:2]
        return image.copy(), np.zeros((h, w), dtype=np.float32)

    h, w = image.shape[:2]
    H, W = h + 2 * expand, w + 2 * expand
    canvas = _edge_extend_canvas(image, expand)

    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    # Euclidean distance outside the original photo rect (0 on/inside edge)
    top, left = float(expand), float(expand)
    bot, right = float(expand + h - 1), float(expand + w - 1)
    dy = np.where(yy < top, top - yy, np.where(yy > bot, yy - bot, 0.0))
    dx = np.where(xx < left, left - xx, np.where(xx > right, xx - right, 0.0))
    dist = np.sqrt(dx * dx + dy * dy)
    inside = (yy >= top) & (yy <= bot) & (xx >= left) & (xx <= right)
    if feather <= 0:
        mask = np.where(inside, 0.0, 1.0).astype(np.float32)
    else:
        # 0 at photo edge → 1 at feather distance (and beyond)
        mask = np.where(
            inside,
            0.0,
            np.clip(dist / float(feather), 0.0, 1.0),
        ).astype(np.float32)
    return canvas, mask


def composite_rgb(base: np.ndarray, fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-composite fill over base using mask (1 = all fill)."""
    if base.shape[:2] != fill.shape[:2]:
        raise ValueError(f"base {base.shape[:2]} vs fill {fill.shape[:2]}")
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    out = base.astype(np.float32) * (1.0 - m) + fill.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def paint_frames(
    *,
    base_rgb: np.ndarray,
    mask: np.ndarray,
    duration_s: float,
    fps: int,
    target: str = "seahorse",
    palette: str = "fire",
    max_iter: int = 256,
    zoom_secs_per_double: float = 4.0,
    beat_map=None,
    julia: bool = False,
    seed: int | None = None,
):
    """Yield (frame_index, composited HxWx3) — fractal animates under the mask."""
    height, width = base_rgb.shape[:2]
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if mask.shape[:2] != (height, width):
        raise ValueError(f"mask shape {mask.shape[:2]} != image {(height, width)}")
    for f, frac in zoom_frames(
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        target=target,
        palette=palette,
        max_iter=max_iter,
        zoom_secs_per_double=zoom_secs_per_double,
        beat_map=beat_map,
        julia=julia,
        seed=seed,
    ):
        yield f, composite_rgb(base_rgb, frac, mask)


def render_paint_video(
    dest: Path,
    *,
    base_rgb: np.ndarray,
    mask: np.ndarray,
    duration_s: float,
    fps: int = 24,
    target: str = "seahorse",
    palette: str = "fire",
    max_iter: int = 256,
    beat_map=None,
    julia: bool = False,
    seed: int | None = None,
    log=print,
) -> Path:
    """Encode an inpaint/outpaint fractal composite video."""
    base_rgb = ensure_even_size(base_rgb)
    mask = ensure_even_size(mask)
    height, width = base_rgb.shape[:2]
    n = max(1, int(round(duration_s * fps)))
    with FfmpegEncoder(dest, width, height, fps) as enc:
        for f, frame in paint_frames(
            base_rgb=base_rgb,
            mask=mask,
            duration_s=duration_s,
            fps=fps,
            target=target,
            palette=palette,
            max_iter=max_iter,
            beat_map=beat_map,
            julia=julia,
            seed=seed,
        ):
            enc.write(frame)
            if f % max(1, fps * 2) == 0:
                log(f"fractal paint frame {f}/{n}")
    log(f"fractal paint video: {enc.dest}")
    return enc.dest
