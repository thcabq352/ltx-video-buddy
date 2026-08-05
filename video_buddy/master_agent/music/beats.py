"""Dependency-free beat detection: ffmpeg decode + numpy spectral flux.

No librosa/scipy — ffmpeg decodes any audio file to mono float32 PCM, then:
  onset envelope  = log-magnitude spectral flux (numpy rfft STFT)
  tempo           = comb-filter scoring over a 60-200 BPM period grid
  beat grid       = phase alignment of a pulse train with the envelope
  sections        = per-beat RMS energy runs (intro/verse/chorus/drop/outro)

Everything is unit-testable on synthetic click tracks (no audio files needed).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
FFT_SIZE = 1024
HOP_SIZE = 512
MIN_BPM = 60.0
MAX_BPM = 200.0


@dataclass
class BeatMap:
    bpm: float
    beats: list[float]  # beat timestamps (s)
    downbeats: list[float]  # every 4th beat
    sections: list[dict]  # {start_s, end_s, label, energy}
    duration_s: float
    onset_env: list[float] = field(default_factory=list)
    env_fps: float = SAMPLE_RATE / HOP_SIZE

    def to_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 2),
            "beats": [round(b, 4) for b in self.beats],
            "downbeats": [round(b, 4) for b in self.downbeats],
            "sections": self.sections,
            "duration_s": round(self.duration_s, 3),
        }


def decode_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode any audio/video file to mono float32 PCM via ffmpeg."""
    from master_agent.video_concat import find_ffmpeg

    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ff, "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"ffmpeg audio decode failed: {err}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if pcm.size == 0:
        raise RuntimeError(f"no audio decoded from {path}")
    return pcm, sr


def audio_duration(path: str | Path) -> float:
    """Duration in seconds via ffprobe (0.0 if unavailable)."""
    probe = shutil.which("ffprobe")
    if not probe:
        return 0.0
    cmd = [
        probe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return float(proc.stdout.strip())
    except (OSError, ValueError):
        return 0.0


def _stft_mag(pcm: np.ndarray, n_fft: int = FFT_SIZE, hop: int = HOP_SIZE) -> np.ndarray:
    """Magnitude STFT via stride tricks + rfft. Frames x (n_fft//2+1)."""
    if pcm.size < n_fft:
        pcm = np.pad(pcm, (0, n_fft - pcm.size))
    n_frames = 1 + (pcm.size - n_fft) // hop
    shape = (n_frames, n_fft)
    strides = (pcm.strides[0] * hop, pcm.strides[0])
    frames = np.lib.stride_tricks.as_strided(pcm, shape=shape, strides=strides)
    window = np.hanning(n_fft).astype(np.float32)
    return np.abs(np.fft.rfft(frames * window, axis=1))


def onset_envelope(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Log-magnitude spectral flux, median-normalized. env_fps = sr / HOP_SIZE."""
    mag = _stft_mag(pcm)
    log_mag = np.log1p(mag * 1000.0)
    flux = np.maximum(np.diff(log_mag, axis=0), 0.0).sum(axis=1)
    env = np.concatenate([[0.0], flux])
    med = np.median(env)
    if med > 0:
        env = env / med
    return env


def estimate_tempo(
    env: np.ndarray,
    env_fps: float,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
) -> float:
    """Comb-filter tempo: score each candidate period by the (fractionally
    sampled) envelope energy under its best-aligned pulse train.

    Raw-sum scoring rewards covering every onset, so half-tempo loses; among
    near-max scores the largest period wins, so double-tempo loses. Robust to
    non-integer frame periods where plain autocorrelation picks the 2x lag.
    """
    e = env - env.mean()
    n = e.size
    if n < 8 or not np.any(e):
        return 120.0
    xs = np.arange(n)
    scores: list[tuple[float, float]] = []
    p_lo = env_fps * 60.0 / max_bpm
    p_hi = env_fps * 60.0 / min_bpm
    for p in np.arange(p_lo, p_hi, 0.02):
        k_max = int(np.ceil(n / p))
        best = -np.inf
        for ph in np.arange(0.0, p, 1.0):
            pos = ph + np.arange(k_max) * p
            s = float(np.interp(pos, xs, e).sum())
            if s > best:
                best = s
        scores.append((best, p))
    top = max(s for s, _ in scores)
    if top <= 0:
        return 120.0
    cands = [(s, p) for s, p in scores if s >= 0.995 * top]
    _, period = max(cands, key=lambda sp: sp[1])
    return 60.0 * env_fps / period


def beat_grid(env: np.ndarray, bpm: float, env_fps: float) -> np.ndarray:
    """Phase-align a pulse train with the onset envelope -> beat timestamps (s)."""
    period = env_fps * 60.0 / bpm
    if period < 1.0 or env.size < period:
        return np.array([])
    e = env - env.mean()
    n = e.size
    xs = np.arange(n)
    k_max = int(np.ceil(n / period))
    ks = np.arange(k_max)
    best_phase, best_score = 0.0, -np.inf
    for ph in np.arange(0.0, period, 0.5):
        s = float(np.interp(ph + ks * period, xs, e).sum())
        if s > best_score:
            best_score, best_phase = s, ph
    frames = np.arange(best_phase, n, period)
    return frames / env_fps


def energy_sections(
    pcm: np.ndarray,
    sr: int,
    beats: np.ndarray,
    duration_s: float,
) -> list[dict]:
    """Coarse sections from per-beat RMS energy runs.

    Beats are grouped into runs of above/below-mean energy; runs shorter than
    2 beats merge into the previous run. Labels: first low run -> intro, last
    low run -> outro, the hottest high run -> drop, other high runs -> chorus,
    remaining low runs -> verse.
    """
    if beats.size < 2:
        return [{"start_s": 0.0, "end_s": duration_s, "label": "verse", "energy": 0.5}]

    per_beat = []
    for i in range(beats.size):
        start = float(beats[i])
        end = float(beats[i + 1]) if i + 1 < beats.size else duration_s
        seg = pcm[int(start * sr):int(end * sr)]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        per_beat.append(rms)
    energies = np.array(per_beat)
    thr = float(energies.mean()) if energies.size else 0.0
    high = energies > thr

    # Group consecutive same-level beats into runs: (start_beat, end_beat, is_high)
    runs: list[list] = []
    for i in range(beats.size):
        if runs and runs[-1][2] == bool(high[i]):
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, bool(high[i])])
    # Merge runs shorter than 2 beats into the previous run
    merged: list[list] = []
    for run in runs:
        if merged and (run[1] - run[0]) < 2:
            merged[-1][1] = run[1]
            # adopt the higher-energy level
            if run[2]:
                merged[-1][2] = True
        else:
            merged.append(list(run))

    def _run_energy(run) -> float:
        return float(energies[run[0]:run[1]].mean())

    hottest = max(
        (r for r in merged if r[2]), key=_run_energy, default=None
    )
    sections = []
    for idx, run in enumerate(merged):
        start_s = float(beats[run[0]])
        end_s = float(beats[run[1]]) if run[1] < beats.size else duration_s
        if idx == 0 and not run[2]:
            label = "intro"
        elif idx == len(merged) - 1 and not run[2] and len(merged) > 1:
            label = "outro"
        elif run[2]:
            label = "drop" if run is hottest else "chorus"
        else:
            label = "verse"
        sections.append(
            {
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "label": label,
                "energy": round(_run_energy(run), 4),
            }
        )
    return sections


def analyze_audio(path: str | Path) -> BeatMap:
    """Full beat map for an audio file."""
    pcm, sr = decode_audio(path)
    duration_s = pcm.size / sr
    env_fps = sr / HOP_SIZE
    env = onset_envelope(pcm, sr)
    bpm = estimate_tempo(env, env_fps)
    beats = beat_grid(env, bpm, env_fps)
    downbeats = [float(b) for b in beats[::4]]
    sections = energy_sections(pcm, sr, beats, duration_s)
    return BeatMap(
        bpm=bpm,
        beats=[float(b) for b in beats],
        downbeats=downbeats,
        sections=sections,
        duration_s=duration_s,
        onset_env=[round(float(v), 4) for v in env[::8]],  # decimated, for records
        env_fps=env_fps,
    )


def section_at(sections: list[dict], t: float) -> dict:
    for sec in sections:
        if sec["start_s"] <= t < sec["end_s"]:
            return sec
    return sections[-1] if sections else {"label": "verse", "energy": 0.5}


def plan_shot_windows(
    bmap: BeatMap,
    *,
    min_s: float = 2.0,
    max_s: float = 6.0,
    high_energy_s: float = 2.0,
    low_energy_s: float = 4.0,
) -> list[tuple[float, float]]:
    """Split the track into shot windows snapped to beat boundaries.

    High-energy sections (chorus/drop) get shorter shots, low-energy sections
    longer ones. Every window starts/ends on a beat (or 0.0 / duration_s),
    windows are contiguous and cover the full track, and none exceeds max_s.
    """
    beat_period = 60.0 / bmap.bpm if bmap.bpm > 0 else 0.5
    bounds = sorted({0.0, *[round(b, 6) for b in bmap.beats], round(bmap.duration_s, 6)})
    bounds = [b for b in bounds if b >= 0.0]
    if len(bounds) < 2:
        return [(0.0, bmap.duration_s)]

    windows: list[tuple[float, float]] = []
    i = 0
    while i < len(bounds) - 1:
        start = bounds[i]
        sec = section_at(bmap.sections, start)
        target = high_energy_s if sec["label"] in ("chorus", "drop") else low_energy_s
        target = max(beat_period, min(target, max_s))
        # Pick the next boundary whose length is closest to target
        best_j = i + 1
        best_err = float("inf")
        for j in range(i + 1, len(bounds)):
            length = bounds[j] - start
            err = abs(length - target)
            if err < best_err - 1e-9:
                best_err, best_j = err, j
            if length > max_s + beat_period:
                break
        windows.append((start, bounds[best_j]))
        i = best_j
    return windows
