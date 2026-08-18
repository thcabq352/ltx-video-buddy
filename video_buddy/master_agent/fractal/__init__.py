"""Procedural fractal video: deep-zoom, inpaint, and outpaint."""

from master_agent.fractal.pipeline import run_fractal
from master_agent.fractal.render import MODES, PALETTES, TARGETS

__all__ = ["run_fractal", "MODES", "PALETTES", "TARGETS"]
