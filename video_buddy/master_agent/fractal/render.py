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
