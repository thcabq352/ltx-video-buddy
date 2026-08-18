"""Tests for the numpy fractal renderer (no GPU, no ffmpeg needed).

Run: .venv/Scripts/python.exe -m pytest tests/test_fractal.py -q
"""

from __future__ import annotations

import unittest

import numpy as np

from master_agent.fractal.render import (
    MODES,
    PALETTES,
    TARGETS,
    center_hole_mask,
    colorize,
    composite_rgb,
    ensure_even_size,
    julia_smooth,
    mandelbrot_smooth,
    paint_frames,
    prepare_outpaint,
    render_frame,
    soft_mask_from_gray,
    zoom_frames,
)


def _counts_at(points, max_iter=256):
    pts = np.array(points, dtype=np.float64)
    return mandelbrot_smooth(pts[:, 0], pts[:, 1], max_iter)


class TestMandelbrotMath(unittest.TestCase):
    def test_origin_is_interior(self):
        (count,) = _counts_at([(0.0, 0.0)], max_iter=200)
        self.assertEqual(count, 200.0)

    def test_pure_i_is_bounded(self):
        # 0 + i stays bounded under z -> z^2 + c
        (count,) = _counts_at([(0.0, 1.0)], max_iter=300)
        self.assertEqual(count, 300.0)

    def test_far_point_escapes_fast(self):
        (count,) = _counts_at([(10.0, 10.0)], max_iter=256)
        self.assertLess(count, 5.0)

    def test_minibrot_center_is_deep(self):
        # The named minibrot target sits deep in the set; in float64 with
        # approximate coordinates it survives hundreds of iterations.
        (count,) = _counts_at([TARGETS["minibrot"][:2]], max_iter=400)
        self.assertGreaterEqual(count, 200.0)

    def test_smooth_counts_are_continuous(self):
        # Points just outside the set should give non-integer smooth counts
        xs = np.linspace(0.2501, 0.26, 8)
        counts = mandelbrot_smooth(xs, np.zeros_like(xs), 512)
        frac = counts - np.floor(counts)
        self.assertTrue(np.any(frac > 0.01), "smooth coloring lost its fraction")


class TestJuliaMath(unittest.TestCase):
    def test_julia_origin_bounded_for_period2_c(self):
        # c = -1 is the period-2 bulb center: orbit of 0 is 0 -> -1 -> 0 (bounded)
        grid = np.array([[0.0]])
        counts = julia_smooth(grid, grid, -1.0, 0.0, 256)
        self.assertEqual(counts[0, 0], 256.0)

    def test_julia_far_point_escapes(self):
        zre = np.array([[5.0]])
        counts = julia_smooth(zre, zre, -0.8, 0.156, 256)
        self.assertLess(counts[0, 0], 5.0)


class TestPalettes(unittest.TestCase):
    def test_five_palettes_as_256_luts(self):
        self.assertEqual(
            {"fire", "ocean", "monochrome", "neon", "sunset"}, set(PALETTES)
        )
        for name, lut in PALETTES.items():
            self.assertEqual((256, 3), lut.shape, name)
            self.assertEqual(np.uint8, lut.dtype, name)

    def test_colorize_marks_interior(self):
        counts = np.array([[256.0, 3.5]])
        rgb = colorize(counts, PALETTES["fire"], max_iter=256)
        self.assertEqual((1, 2, 3), rgb.shape)
        self.assertTrue((rgb[0, 0] == [0, 0, 0]).all(), "interior must be black")
        self.assertFalse((rgb[0, 1] == [0, 0, 0]).all())


class TestFramesAndZoom(unittest.TestCase):
    def test_render_frame_shape_and_bytes(self):
        frame = render_frame(
            TARGETS["seahorse"][:2],
            1.6,
            width=64,
            height=48,
            max_iter=64,
            lut=PALETTES["ocean"],
        )
        self.assertEqual((48, 64, 3), frame.shape)
        self.assertEqual(np.uint8, frame.dtype)

    def test_render_frame_julia_mode(self):
        frame = render_frame(
            (0.0, 0.0),
            1.6,
            width=64,
            height=48,
            max_iter=64,
            julia_c=(-0.8, 0.156),
        )
        self.assertEqual((48, 64, 3), frame.shape)

    def test_zoom_schedule_is_monotonic(self):
        # Patch render_frame to record half_w so the schedule is inspectable
        seen = []

        def spy(center, half_w, **kw):
            seen.append(half_w)
            return np.zeros((kw["height"], kw["width"], 3), dtype=np.uint8)

        import master_agent.fractal.render as render_mod

        orig = render_mod.render_frame
        render_mod.render_frame = spy
        try:
            frames = list(
                zoom_frames(duration_s=1.0, fps=8, width=16, height=16, target="seahorse")
            )
        finally:
            render_mod.render_frame = orig
        self.assertEqual(8, len(frames))
        self.assertEqual(8, len(seen))
        for a, b in zip(seen, seen[1:]):
            self.assertLess(b, a, "zoom must shrink half_w every frame")

    def test_zoom_respects_float64_depth_cap(self):
        import master_agent.fractal.render as render_mod

        seen = []

        def spy(center, half_w, **kw):
            seen.append(half_w)
            return np.zeros((kw["height"], kw["width"], 3), dtype=np.uint8)

        orig = render_mod.render_frame
        render_mod.render_frame = spy
        try:
            # tiny doubling interval drives the zoom to the cap quickly
            list(
                zoom_frames(
                    duration_s=4.0,
                    fps=24,
                    width=16,
                    height=16,
                    target="seahorse",
                    zoom_secs_per_double=0.01,
                )
            )
        finally:
            render_mod.render_frame = orig
        floor = TARGETS["seahorse"][2] / render_mod.MAX_ZOOM_DEPTH
        self.assertGreaterEqual(min(seen), floor * 0.999)


class TestInpaintOutpaint(unittest.TestCase):
    def test_modes_tuple(self):
        self.assertEqual({"zoom", "inpaint", "outpaint"}, set(MODES))

    def test_center_hole_mask_soft_edges(self):
        m = center_hole_mask(64, 96, cover=0.5, feather=8)
        self.assertEqual(m.shape, (64, 96))
        self.assertGreater(m[32, 48], 0.9)  # center filled
        self.assertLess(m[0, 0], 0.15)  # corner kept
        # soft ramp exists somewhere
        self.assertTrue(np.any((m > 0.1) & (m < 0.9)))

    def test_prepare_outpaint_canvas_and_mask(self):
        img = np.full((40, 60, 3), 180, dtype=np.uint8)
        canvas, mask = prepare_outpaint(img, expand=10, feather=4)
        self.assertEqual(canvas.shape, (60, 80, 3))
        self.assertEqual(mask.shape, (60, 80))
        # original region fully kept
        self.assertTrue((mask[10:50, 10:70] == 0).all())
        # far corner is full fill; first pixel outside has soft ramp
        self.assertGreaterEqual(mask[0, 0], 0.99)
        self.assertGreater(mask[9, 40], 0.0)
        self.assertLess(mask[9, 40], 1.0)
        # photo pixels preserved; borders edge-extended (not black)
        self.assertTrue((canvas[10:50, 10:70] == 180).all())
        self.assertTrue((canvas[0, 10:70] == 180).all())

    def test_composite_blends(self):
        base = np.zeros((8, 8, 3), dtype=np.uint8)
        fill = np.full((8, 8, 3), 200, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[:, 4:] = 1.0
        out = composite_rgb(base, fill, mask)
        self.assertTrue((out[:, :4] == 0).all())
        self.assertTrue((out[:, 4:] == 200).all())

    def test_ensure_even_size_crops_odd(self):
        odd = np.zeros((5, 7, 3), dtype=np.uint8)
        even = ensure_even_size(odd)
        self.assertEqual(even.shape, (4, 6, 3))
        self.assertEqual(ensure_even_size(np.zeros((4, 6), dtype=np.float32)).shape, (4, 6))

    def test_soft_mask_from_gray(self):
        gray = np.zeros((16, 16), dtype=np.float32)
        gray[4:12, 4:12] = 1.0
        soft = soft_mask_from_gray(gray, feather=3)
        self.assertGreater(soft[8, 8], 0.9)
        self.assertLess(soft[0, 0], 0.2)

    def test_paint_frames_composites_one_frame(self):
        base = np.full((32, 48, 3), 40, dtype=np.uint8)
        mask = center_hole_mask(32, 48, cover=0.5, feather=4)
        frames = list(
            paint_frames(
                base_rgb=base,
                mask=mask,
                duration_s=1 / 12,
                fps=12,
                max_iter=32,
                seed=1,
            )
        )
        self.assertEqual(1, len(frames))
        idx, frame = frames[0]
        self.assertEqual(0, idx)
        self.assertEqual((32, 48, 3), frame.shape)
        # corner should stay close to base (outside hole)
        self.assertLess(int(frame[0, 0].mean()), 80)


if __name__ == "__main__":
    unittest.main()
