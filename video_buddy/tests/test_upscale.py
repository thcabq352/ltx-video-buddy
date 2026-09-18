"""Upscale defaults to SeedVR2; RTX is gated on an on-disk template.

Run: python -m pytest tests/test_upscale.py -q
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from master_agent.config import WORKFLOWS_DIR
from master_agent.upscale import RTX_WORKFLOW, SEEDVR2_WORKFLOW, upscale_video


def test_default_method_is_seedvr2():
    sig = inspect.signature(upscale_video)
    assert sig.parameters["method"].default == "seedvr2"


def test_seedvr2_template_is_in_repo():
    assert SEEDVR2_WORKFLOW.is_file()
    assert SEEDVR2_WORKFLOW == WORKFLOWS_DIR / "upscale_seedvr2_api.json"


def test_rtx_without_file_raises_named_path(tmp_path):
    if RTX_WORKFLOW.is_file():
        pytest.skip("RTX template is present on this checkout")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="rtx") as ctx:
        upscale_video(video, method="rtx")
    assert str(RTX_WORKFLOW) in str(ctx.value)
