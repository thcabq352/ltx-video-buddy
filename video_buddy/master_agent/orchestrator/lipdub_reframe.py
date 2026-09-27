"""Source-still anchoring helpers for segmented lipdub. No new models.

The shipped ``ltx25_a2v`` graph has one ``LoadImage`` into both
``LTXVImgToVideoInplace`` stages. It has no second keyframe and no
``LTXVAddGuide`` input, so identity is the original still on that single
image slot. The previous piece's last frame is a continuity guide only:
a low-weight blend into the hybrid conditioning image, plus an overlap
crossfade so the join is not a hard cut back to the still.

Push-in that still leaks through is measured as a similarity (zoom and
shift) against the source still and undone per frame. OpenCV ECC is used
when ``cv2`` is already installed. Otherwise a coarse normalized
cross-correlation search does the same job. Neither path downloads weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

HYBRID_PREV_WEIGHT = 0.25
# Zoom of the generated frame relative to the source. 1.0 is the same framing.
# Values above 1 are a push-in (subject larger than the still).
ZOOM_MIN = 0.92
ZOOM_MAX = 1.22
MATCH_MIN_SCORE = 0.25


@dataclass
class Similarity:
    """Push-in of a frame against the source still, in source pixels."""

    zoom: float
    dx: float
    dy: float
    score: float

    def as_dict(self) -> dict:
        return {
            "scale": round(float(self.zoom), 4),
            "dx": round(float(self.dx), 2),
            "dy": round(float(self.dy), 2),
            "score": round(float(self.score), 4),
        }

    @property
    def worth_applying(self) -> bool:
        if self.score < MATCH_MIN_SCORE:
            return False
        if not (ZOOM_MIN <= self.zoom <= ZOOM_MAX):
            return False
        return abs(self.zoom - 1.0) >= 0.012 or abs(self.dx) >= 0.75 or abs(self.dy) >= 0.75


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.dtype == np.uint8:
        return data
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def _gray(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        return data
    rgb = data[..., :3]
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _resize_gray(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape
    if w <= width:
        return gray.astype(np.float32, copy=False)
    nh = max(8, int(round(h * (width / float(w)))))
    im = Image.fromarray(_to_uint8(gray), "L")
    im = im.resize((width, nh), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.asarray(a, dtype=np.float64).reshape(-1)
    fb = np.asarray(b, dtype=np.float64).reshape(-1)
    if fa.size < 16 or fa.size != fb.size:
        return -1.0
    fa = fa - fa.mean()
    fb = fb - fb.mean()
    denom = float(np.sqrt(np.dot(fa, fa) * np.dot(fb, fb)))
    if denom < 1e-6:
        return 1.0 if float(np.mean(np.abs(a - b))) < 1.0 else 0.0
    return float(np.dot(fa, fb) / denom)


def push_in(arr: np.ndarray, zoom: float, dx: float, dy: float) -> np.ndarray:
    """Zoom into ``arr`` and shift the window. Output shape matches ``arr``.

    ``dx > 0`` slides the crop window to the right. ``zoom > 1`` enlarges
    the subject (a push-in).
    """
    data = _to_uint8(arr)
    if data.ndim == 2:
        mode = "L"
        h, w = data.shape
    else:
        mode = "RGB"
        data = data[..., :3]
        h, w = data.shape[:2]
    z = max(float(zoom), 1.001)
    pil = Image.fromarray(data, mode)
    nw = max(w + 2, int(round(w * z)))
    nh = max(h + 2, int(round(h * z)))
    big = pil.resize((nw, nh), Image.Resampling.BICUBIC)
    left = int(round((nw - w) / 2.0 + float(dx) * z))
    top = int(round((nh - h) / 2.0 + float(dy) * z))
    left = max(0, min(nw - w, left))
    top = max(0, min(nh - h, top))
    crop = big.crop((left, top, left + w, top + h))
    return np.asarray(crop)


def undo_push_in(
    frame: np.ndarray,
    source: np.ndarray,
    zoom: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Invert ``push_in`` so a zoomed frame lands back on the source framing.

    Pixels the shrunk frame does not cover are filled from the source still.
    ``zoom`` below 1 is recorded by the estimator but not warped: the tower
    failure is a push-in, and a zoom-out inverse would enlarge a bad match.
    """
    frame_u = _to_uint8(frame)
    source_u = _to_uint8(source)
    if frame_u.ndim == 2:
        frame_u = np.stack([frame_u] * 3, axis=-1)
    else:
        frame_u = frame_u[..., :3]
    if source_u.ndim == 2:
        source_u = np.stack([source_u] * 3, axis=-1)
    else:
        source_u = source_u[..., :3]
    fh, fw = frame_u.shape[:2]
    src = Image.fromarray(source_u, "RGB")
    if src.size != (fw, fh):
        src = src.resize((fw, fh), Image.Resampling.LANCZOS)
        source_u = np.asarray(src)
    z = float(zoom)
    # A measured zoom under this is not a push-in. Leave the frame alone.
    if z < 1.012:
        return frame_u
    z = min(max(z, 1.012), ZOOM_MAX)
    # Same window ``push_in`` crops out of the scaled still, including its clamp.
    nw = max(fw + 2, int(round(fw * z)))
    nh = max(fh + 2, int(round(fh * z)))
    left = int(round((nw - fw) / 2.0 + float(dx) * z))
    top = int(round((nh - fh) / 2.0 + float(dy) * z))
    left = max(0, min(nw - fw, left))
    top = max(0, min(nh - fh, top))
    ys, xs = np.mgrid[0:fh, 0:fw]
    fx = xs * (nw / float(fw)) - left
    fy = ys * (nh / float(fh)) - top
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (fx - x0)[..., None]
    wy = (fy - y0)[..., None]
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < fw) & (y1 < fh)
    out = source_u.astype(np.float32)
    fr = frame_u.astype(np.float32)
    x0c = np.clip(x0, 0, fw - 1)
    x1c = np.clip(x1, 0, fw - 1)
    y0c = np.clip(y0, 0, fh - 1)
    y1c = np.clip(y1, 0, fh - 1)
    samp = (
        fr[y0c, x0c] * (1.0 - wx) * (1.0 - wy)
        + fr[y0c, x1c] * wx * (1.0 - wy)
        + fr[y1c, x0c] * (1.0 - wx) * wy
        + fr[y1c, x1c] * wx * wy
    )
    out = out.copy()
    out[valid] = samp[valid]
    return _to_uint8(out)


def estimate_similarity(
    frame,
    source,
    *,
    method: str = "ncc",
) -> Similarity:
    """Estimate how far ``frame`` has pushed in relative to ``source``.

    ``method="ncc"`` is the normalized cross-correlation search (tests use
    this). ``method="auto"`` prefers an OpenCV ECC search over a few zooms
    when ``cv2`` is already installed, and falls back to NCC.
    """
    if method == "auto":
        ecc = _estimate_ecc(frame, source)
        if ecc is not None and ecc.score >= MATCH_MIN_SCORE:
            return ecc
    return _estimate_ncc(frame, source)


def _estimate_ecc(frame, source) -> Optional[Similarity]:
    """Zoom search + ECC translation. No extra weights; ``cv2`` is optional."""
    try:
        import cv2
    except Exception:
        return None
    try:
        src_full = _gray(np.asarray(source))
        fr_full = _gray(np.asarray(frame))
        src = _to_uint8(_resize_gray(src_full, 160))
        fr = _to_uint8(_resize_gray(fr_full, 160))
        if fr.shape != src.shape:
            fr = cv2.resize(fr, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        ratio = src_full.shape[1] / float(src.shape[1])
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-4)
        best: Optional[Similarity] = None
        for zoom in np.linspace(0.96, 1.18, 8):
            inv = 1.0 / float(zoom)
            scaled = cv2.resize(fr, None, fx=inv, fy=inv, interpolation=cv2.INTER_LINEAR)
            canvas = src.copy()
            sh, sw = scaled.shape[:2]
            ch, cw = canvas.shape[:2]
            y0 = max(0, (ch - sh) // 2)
            x0 = max(0, (cw - sw) // 2)
            y1 = min(ch, y0 + sh)
            x1 = min(cw, x0 + sw)
            canvas[y0:y1, x0:x1] = scaled[: y1 - y0, : x1 - x0]
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                score, warp = cv2.findTransformECC(
                    src, canvas, warp, cv2.MOTION_TRANSLATION, criteria, None, 1
                )
            except cv2.error:
                continue
            sim = Similarity(
                zoom=float(zoom),
                dx=float(warp[0, 2]) * ratio,
                dy=float(warp[1, 2]) * ratio,
                score=float(score),
            )
            if best is None or sim.score > best.score:
                best = sim
        return best
    except Exception:
        return None


def _estimate_ncc(frame, source) -> Similarity:
    src = _gray(np.asarray(source))
    fr = _gray(np.asarray(frame))
    if fr.shape != src.shape:
        fr_im = Image.fromarray(_to_uint8(fr), "L").resize(
            (src.shape[1], src.shape[0]), Image.Resampling.BILINEAR
        )
        fr = np.asarray(fr_im, dtype=np.float32)
    if float(src.std()) < 1.0 and float(fr.std()) < 1.0:
        return Similarity(1.0, 0.0, 0.0, 1.0)
    src_s = _resize_gray(src, 72)
    fr_s = _resize_gray(fr, 72)
    if fr_s.shape != src_s.shape:
        fr_s = np.asarray(
            Image.fromarray(_to_uint8(fr_s), "L").resize(
                (src_s.shape[1], src_s.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    ratio = src.shape[1] / float(src_s.shape[1])
    best_z, best_dx, best_dy, best_score = 1.0, 0.0, 0.0, -1.0
    for zoom in np.linspace(0.96, 1.18, 12):
        for dx in range(-8, 9, 2):
            for dy in range(-6, 7, 2):
                cand = push_in(src_s, float(zoom), float(dx), float(dy))
                score = _ncc(cand, fr_s)
                if score > best_score:
                    best_z, best_dx, best_dy, best_score = float(zoom), float(dx), float(dy), score
    # Local refine around the coarse peak.
    z0, x0, y0 = best_z, best_dx, best_dy
    for zoom in np.linspace(z0 - 0.03, z0 + 0.03, 7):
        for dx in (x0 - 1, x0, x0 + 1):
            for dy in (y0 - 1, y0, y0 + 1):
                cand = push_in(src_s, float(zoom), float(dx), float(dy))
                score = _ncc(cand, fr_s)
                if score > best_score:
                    best_z, best_dx, best_dy, best_score = float(zoom), float(dx), float(dy), score
    return Similarity(
        zoom=best_z,
        dx=best_dx * ratio,
        dy=best_dy * ratio,
        score=best_score,
    )


def smooth_track(
    samples: Sequence[tuple[int, Similarity]],
    n_frames: int,
) -> list[Similarity]:
    """Interpolate sampled similarities across ``n_frames`` and smooth them."""
    n = max(int(n_frames), 1)
    if n == 1 or not samples:
        sim = samples[0][1] if samples else Similarity(1.0, 0.0, 0.0, 0.0)
        return [sim for _ in range(n)]
    ordered = sorted(samples, key=lambda item: item[0])
    idx = np.array([min(max(i, 0), n - 1) for i, _sim in ordered], dtype=np.float64)
    vals = np.array(
        [[sim.zoom, sim.dx, sim.dy, sim.score] for _i, sim in ordered],
        dtype=np.float64,
    )
    xs = np.arange(n, dtype=np.float64)
    out = np.stack([np.interp(xs, idx, vals[:, k]) for k in range(4)], axis=1)
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = out.copy()
    for k in range(3):
        padded = np.pad(out[:, k], 1, mode="edge")
        smoothed[:, k] = np.convolve(padded, kernel, mode="valid")
    return [
        Similarity(
            zoom=float(smoothed[i, 0]),
            dx=float(smoothed[i, 1]),
            dy=float(smoothed[i, 2]),
            score=float(out[i, 3]),
        )
        for i in range(n)
    ]


def crossfade_alphas(n: int) -> list[float]:
    """Blend weights for an overlap. 0 keeps the previous frame, 1 takes the new one."""
    count = int(n)
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [i / float(count - 1) for i in range(count)]


def blend_rgb(previous, incoming, t: float) -> np.ndarray:
    a = _to_uint8(np.asarray(previous))
    b = _to_uint8(np.asarray(incoming))
    if a.shape != b.shape:
        b_im = Image.fromarray(b[..., :3] if b.ndim == 3 else b, "RGB" if b.ndim == 3 else "L")
        size = (a.shape[1], a.shape[0])
        b_im = b_im.resize(size, Image.Resampling.BILINEAR)
        b = np.asarray(b_im.convert("RGB"))
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
    weight = min(max(float(t), 0.0), 1.0)
    mixed = (1.0 - weight) * a.astype(np.float32) + weight * b.astype(np.float32)
    return _to_uint8(mixed)


def compose_hybrid_guide(
    source,
    previous,
    *,
    weight: float = HYBRID_PREV_WEIGHT,
    method: str = "ncc",
) -> np.ndarray:
    """Source still, with a little of the previous frame after undoing its zoom.

    The previous frame is the pose hint. The still stays the majority of the
    pixels so face paint and framing remain the identity reference. Stage 2
    of the a2v graph copies this image onto frame 0; the overlap crossfade
    removes that frame from the visible join.
    """
    src = _to_uint8(np.asarray(source.convert("RGB") if isinstance(source, Image.Image) else source))
    prev = _to_uint8(np.asarray(previous.convert("RGB") if isinstance(previous, Image.Image) else previous))
    if src.ndim == 2:
        src = np.stack([src] * 3, axis=-1)
    else:
        src = src[..., :3]
    if prev.ndim == 2:
        prev = np.stack([prev] * 3, axis=-1)
    else:
        prev = prev[..., :3]
    if prev.shape[:2] != src.shape[:2]:
        prev = np.asarray(
            Image.fromarray(prev, "RGB").resize((src.shape[1], src.shape[0]), Image.Resampling.LANCZOS)
        )
    sim = estimate_similarity(prev, src, method=method)
    corrected = undo_push_in(prev, src, sim.zoom, sim.dx, sim.dy) if sim.worth_applying else prev
    return blend_rgb(src, corrected, weight)


def compose_hybrid_guide_file(
    source_path: Path,
    previous_path: Path,
    dest: Path,
    *,
    weight: float = HYBRID_PREV_WEIGHT,
) -> Path:
    with Image.open(source_path) as src, Image.open(previous_path) as prev:
        mixed = compose_hybrid_guide(src.convert("RGB"), prev.convert("RGB"), weight=weight, method="auto")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mixed, "RGB").save(dest)
    return dest


def _sample_indices(n: int, k: int = 5) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    raw = [int(round(i * (n - 1) / float(k - 1))) for i in range(k)]
    out: list[int] = []
    for idx in raw:
        if idx not in out:
            out.append(idx)
    return out


def median_similarity(track: Sequence[Similarity]) -> Similarity:
    if not track:
        return Similarity(1.0, 0.0, 0.0, 0.0)
    zoom = float(np.median([s.zoom for s in track]))
    dx = float(np.median([s.dx for s in track]))
    dy = float(np.median([s.dy for s in track]))
    score = float(np.median([s.score for s in track]))
    return Similarity(zoom=zoom, dx=dx, dy=dy, score=score)


def _ffmpeg() -> str:
    from master_agent.video_concat import find_ffmpeg

    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    return ff


def _run(cmd: list[str]) -> None:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-600:]
        raise RuntimeError(f"ffmpeg failed: {err}")


def _decode_frames(video: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pattern = dest_dir / "f_%04d.png"
    _run([_ffmpeg(), "-y", "-i", str(video), str(pattern)])
    frames = sorted(dest_dir.glob("f_*.png"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    return frames


def _encode_frames(frames: Sequence[Path], dest: Path, *, fps: int) -> Path:
    if not frames:
        raise RuntimeError("no frames to encode")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / f"{dest.stem}_enc"
    if work.exists():
        for old in work.glob("f_*.png"):
            old.unlink()
    work.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames):
        target = work / f"f_{i:04d}.png"
        if Path(src) != target:
            Image.open(src).save(target)
    _run(
        [
            _ffmpeg(), "-y",
            "-framerate", str(int(fps)),
            "-i", str(work / "f_%04d.png"),
            "-an",
            "-r", str(int(fps)),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"encode produced no file: {dest}")
    return dest


def reframe_video(
    video: Path,
    source_image: Path,
    dest: Path,
    *,
    fps: int,
    method: str = "auto",
) -> tuple[Path, Similarity]:
    """Warp ``video`` back toward ``source_image`` framing. Returns the median drift."""
    video = Path(video)
    dest = Path(dest)
    work = dest.parent / f"{dest.stem}_dec"
    frames = _decode_frames(video, work)
    with Image.open(source_image) as src_im:
        source = np.asarray(src_im.convert("RGB"))
    indices = _sample_indices(len(frames), 5)
    samples: list[tuple[int, Similarity]] = []
    for idx in indices:
        with Image.open(frames[idx]) as fr_im:
            frame = np.asarray(fr_im.convert("RGB"))
        samples.append((idx, estimate_similarity(frame, source, method=method)))
    track = smooth_track(samples, len(frames))
    summary = median_similarity([sim for _i, sim in samples])
    corrected: list[Path] = []
    out_dir = dest.parent / f"{dest.stem}_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (path, sim) in enumerate(zip(frames, track)):
        with Image.open(path) as fr_im:
            frame = np.asarray(fr_im.convert("RGB"))
        if sim.worth_applying:
            fixed = undo_push_in(frame, source, sim.zoom, sim.dx, sim.dy)
        else:
            fixed = frame
        out = out_dir / f"f_{i:04d}.png"
        Image.fromarray(_to_uint8(fixed), "RGB").save(out)
        corrected.append(out)
    return _encode_frames(corrected, dest, fps=fps), summary


def crossfade_tail(
    previous: Path,
    overlap: Path,
    dest: Path,
    n: int,
    *,
    fps: int,
) -> Path:
    """Replace the last ``n`` frames of ``previous`` with a blend into ``overlap``.

    Frame count stays the same. The last blended frame is the incoming overlap
    frame, which is the same timestamp the next piece continues from.
    """
    count = int(n)
    if count <= 0:
        raise RuntimeError("crossfade length must be positive")
    prev_dir = Path(dest).parent / f"{Path(dest).stem}_prev"
    ov_dir = Path(dest).parent / f"{Path(dest).stem}_ov"
    prev_frames = _decode_frames(Path(previous), prev_dir)
    ov_frames = _decode_frames(Path(overlap), ov_dir)
    count = min(count, len(prev_frames), len(ov_frames))
    if count <= 0:
        raise RuntimeError("crossfade has no overlapping frames")
    alphas = crossfade_alphas(count)
    blended_paths: list[Path] = []
    head = prev_frames[:-count]
    mix_dir = Path(dest).parent / f"{Path(dest).stem}_mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    for i, (alpha, prev_path, ov_path) in enumerate(
        zip(alphas, prev_frames[-count:], ov_frames[:count])
    ):
        with Image.open(prev_path) as a_im, Image.open(ov_path) as b_im:
            mixed = blend_rgb(np.asarray(a_im.convert("RGB")), np.asarray(b_im.convert("RGB")), alpha)
        out = mix_dir / f"m_{i:04d}.png"
        Image.fromarray(mixed, "RGB").save(out)
        blended_paths.append(out)
    return _encode_frames([*head, *blended_paths], dest, fps=fps)
