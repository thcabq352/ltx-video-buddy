"""Tests for beat detection + music-video assembly helpers (no GPU/network).

Run: .venv/Scripts/python.exe -m pytest tests/test_music.py -q
"""

from __future__ import annotations

import shutil
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from master_agent.music.beats import (
    HOP_SIZE,
    SAMPLE_RATE,
    BeatMap,
    analyze_audio,
    beat_grid,
    decode_audio,
    energy_sections,
    estimate_tempo,
    onset_envelope,
    plan_shot_windows,
)

ENV_FPS = SAMPLE_RATE / HOP_SIZE


def _click_track(bpm: float, seconds: float, *, phase_s: float = 0.0) -> np.ndarray:
    """Synthetic click track: 5ms noise bursts on every beat."""
    pcm = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    rng = np.random.default_rng(7)
    period = 60.0 / bpm
    t = phase_s
    click = (rng.standard_normal(int(0.005 * SAMPLE_RATE)) * 0.9).astype(np.float32)
    while t < seconds - 0.01:
        i = int(t * SAMPLE_RATE)
        pcm[i:i + click.size] += click
        t += period
    return pcm


class TestTempo(unittest.TestCase):
    def test_120_bpm_pulse_train(self):
        pcm = _click_track(120.0, 12.0)
        env = onset_envelope(pcm, SAMPLE_RATE)
        bpm = estimate_tempo(env, ENV_FPS)
        self.assertAlmostEqual(120.0, bpm, delta=2.0)

    def test_90_bpm_pulse_train(self):
        pcm = _click_track(90.0, 14.0)
        env = onset_envelope(pcm, SAMPLE_RATE)
        bpm = estimate_tempo(env, ENV_FPS)
        self.assertAlmostEqual(90.0, bpm, delta=3.0)

    def test_silence_returns_default(self):
        env = np.zeros(500)
        self.assertEqual(120.0, estimate_tempo(env, ENV_FPS))


class TestBeatGrid(unittest.TestCase):
    def test_phase_alignment(self):
        # clicks every 0.5s starting at 0.25s -> first beat near 0.25
        pcm = _click_track(120.0, 10.0, phase_s=0.25)
        env = onset_envelope(pcm, SAMPLE_RATE)
        beats = beat_grid(env, 120.0, ENV_FPS)
        self.assertGreater(beats.size, 10)
        self.assertAlmostEqual(0.25, float(beats[0]), delta=0.05)
        spacing = np.diff(beats)
        self.assertAlmostEqual(0.5, float(spacing.mean()), delta=0.02)


class TestSections(unittest.TestCase):
    def test_loud_quiet_split(self):
        seconds = 16.0
        pcm = _click_track(120.0, seconds)
        t = np.arange(pcm.size) / SAMPLE_RATE
        carrier = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        amp = np.where(t < 8.0, 0.02, 0.5)  # quiet first half, loud second half
        pcm = pcm + carrier * amp.astype(np.float32)

        env = onset_envelope(pcm, SAMPLE_RATE)
        beats = beat_grid(env, 120.0, ENV_FPS)
        sections = energy_sections(pcm, SAMPLE_RATE, beats, seconds)
        self.assertGreaterEqual(len(sections), 2)
        # boundary should land within a couple beats of the 8s change point
        boundaries = [s["start_s"] for s in sections[1:]]
        self.assertTrue(
            any(abs(b - 8.0) < 1.5 for b in boundaries),
            f"no boundary near 8s: {sections}",
        )
        first_half = [s for s in sections if s["end_s"] <= 8.0]
        second_half = [s for s in sections if s["start_s"] >= 7.0]
        self.assertLess(
            max(s["energy"] for s in first_half),
            max(s["energy"] for s in second_half),
        )


class TestShotWindows(unittest.TestCase):
    def _beatmap(self, seconds=20.0, bpm=120.0) -> BeatMap:
        period = 60.0 / bpm
        beats = list(np.arange(0.0, seconds, period))
        sections = [
            {"start_s": 0.0, "end_s": seconds / 2, "label": "verse", "energy": 0.2},
            {"start_s": seconds / 2, "end_s": seconds, "label": "drop", "energy": 0.9},
        ]
        return BeatMap(
            bpm=bpm,
            beats=[float(b) for b in beats],
            downbeats=[float(b) for b in beats[::4]],
            sections=sections,
            duration_s=seconds,
        )

    def test_windows_align_cover_and_respect_cap(self):
        bmap = self._beatmap()
        max_s = 6.0
        windows = plan_shot_windows(bmap, max_s=max_s)
        bounds = {0.0, *[round(b, 6) for b in bmap.beats], round(bmap.duration_s, 6)}
        for start, end in windows:
            self.assertIn(round(start, 6), bounds)
            self.assertIn(round(end, 6), bounds)
            self.assertLessEqual(end - start, max_s + 1e-6)
        # contiguous full coverage
        self.assertEqual(0.0, windows[0][0])
        self.assertAlmostEqual(bmap.duration_s, windows[-1][1], places=6)
        for (_, end), (next_start, _) in zip(windows, windows[1:]):
            self.assertAlmostEqual(end, next_start, places=6)

    def test_high_energy_sections_get_shorter_shots(self):
        bmap = self._beatmap()
        windows = plan_shot_windows(bmap, high_energy_s=2.0, low_energy_s=4.0)
        verse = [e - s for s, e in windows if s < 10.0]
        drop = [e - s for s, e in windows if s >= 10.0]
        self.assertLess(np.mean(drop), np.mean(verse))


class TestMuxAndDecode(unittest.TestCase):
    def test_mux_command_construction(self):
        from master_agent import video_concat

        with TemporaryDirectory() as td:
            dest = Path(td) / "out.mp4"
            dest.write_bytes(b"x")  # dest exists so the rc==0 path passes
            with patch.object(video_concat, "find_ffmpeg", return_value="ffmpeg"), \
                 patch.object(video_concat.subprocess, "run") as run:
                run.return_value.returncode = 0
                video_concat.mux_audio(Path("v.mp4"), Path("a.mp3"), dest)
            cmd = run.call_args[0][0]
            self.assertEqual("ffmpeg", cmd[0])
            joined = " ".join(cmd)
            self.assertIn("-c:v copy", joined)
            self.assertIn("-c:a aac", joined)
            self.assertIn("-shortest", joined)
            self.assertIn("v.mp4", cmd)
            self.assertIn("a.mp3", cmd)

    def test_decode_audio_roundtrip(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not on PATH")
        with TemporaryDirectory() as td:
            wav_path = Path(td) / "tone.wav"
            sr = SAMPLE_RATE
            t = np.arange(sr) / sr
            pcm16 = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm16.tobytes())
            pcm, out_sr = decode_audio(wav_path)
            self.assertEqual(sr, out_sr)
            self.assertGreater(pcm.size, sr // 2)
            self.assertEqual(np.float32, pcm.dtype)

    def test_analyze_audio_end_to_end(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not on PATH")
        with TemporaryDirectory() as td:
            wav_path = Path(td) / "clicks.wav"
            pcm = _click_track(120.0, 8.0)
            pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm16.tobytes())
            bmap = analyze_audio(wav_path)
            self.assertAlmostEqual(120.0, bmap.bpm, delta=2.0)
            self.assertAlmostEqual(8.0, bmap.duration_s, delta=0.2)
            self.assertGreater(len(bmap.beats), 10)
            self.assertTrue(bmap.sections)


if __name__ == "__main__":
    unittest.main()
