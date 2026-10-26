"""Tiny preview VAEs stay off tiled decode.

ComfyUI 0.38 reports taeltx* / tae* as 16x/4x. They actually scale 32x/8x,
so a tiled decode builds the wrong blend mask. No GPU, no Comfy process.

Run: python -m pytest tests/test_tiny_vae_tiled_decode.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.comfy.client import ComfyClient
from master_agent.comfy.vae_guard import TinyVAETiledDecodeError
from master_agent.comfy.workflow_patcher import _heuristic_patch, load_and_patch_workflow
from master_agent.config import LTX23_FULL_VIDEO_VAE, MODEL_FILES, TINY_LTX_PREVIEW_VAE

TINY = TINY_LTX_PREVIEW_VAE
FULL = LTX23_FULL_VIDEO_VAE


def _graph(decode_class: str, vae_name: str) -> dict:
    return {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Load VAE"},
        },
        "2": {
            "class_type": decode_class,
            "inputs": {"samples": ["9", 0], "vae": ["1", 0]},
        },
    }


def test_four_variant_defaults_are_the_tiny_preview_vae():
    for variant in ("base", "eros", "directors", "lipsync"):
        assert MODEL_FILES[variant]["vae"] == TINY


def test_default_tiny_vae_is_replaced_on_tiled_decode():
    workflow = _graph("VAEDecodeTiled", TINY)
    _heuristic_patch(workflow, {"vae_name": TINY})
    assert workflow["1"]["inputs"]["vae_name"] == FULL
    assert TINY not in str(workflow)

    custom = _graph("ExampleTiledVAEDecode", "tae_preview.safetensors")
    _heuristic_patch(custom, {"vae_name": "tae_preview.safetensors"})
    assert custom["1"]["inputs"]["vae_name"] == FULL
    assert "tae_preview.safetensors" not in str(custom)


def test_explicit_tiny_vae_on_tiled_decode_raises():
    workflow = _graph("LTXVTiledVAEDecode", FULL)
    with pytest.raises(TinyVAETiledDecodeError, match="not valid with tiled decode"):
        _heuristic_patch(
            workflow,
            {"vae_name": TINY, "vae_requested": TINY, "vae_explicit": True},
        )
    assert workflow["1"]["inputs"]["vae_name"] == FULL


def test_plain_vaedecode_keeps_tiny_preview():
    workflow = _graph("VAEDecode", TINY)
    _heuristic_patch(workflow, {"vae_name": TINY})
    assert workflow["1"]["inputs"]["vae_name"] == TINY


def test_explicit_tiny_vae_on_plain_base_template_stays():
    """base shares ltx23_av.json. Node 70 is plain VAEDecode, so taeltx2_3 stays."""
    with patch(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        return_value=Path("fake.safetensors"),
    ):
        workflow, _meta = load_and_patch_workflow(
            "base",
            prompt="neon rain",
            seed=1,
            duration_s=1.0,
            vae=TINY,
        )
    assert workflow["70"]["class_type"] == "VAEDecode"
    assert workflow["5"]["inputs"]["vae_name"] == TINY
    assert "LTXVTiledVAEDecode" not in str(workflow)
    assert "VAEDecodeTiled" not in str(workflow)


def test_explicit_tiny_vae_on_lipsync_tiled_template_raises():
    with patch(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        return_value=Path("fake.safetensors"),
    ):
        with pytest.raises(TinyVAETiledDecodeError):
            load_and_patch_workflow(
                "lipsync",
                prompt="neon rain",
                seed=1,
                duration_s=1.0,
                vae=TINY,
            )


def test_queue_prompt_refuses_tiny_vae_before_submit():
    client = ComfyClient("http://127.0.0.1:9")
    with pytest.raises(TinyVAETiledDecodeError):
        client.queue_prompt(_graph("VAEDecodeTiled", TINY))
