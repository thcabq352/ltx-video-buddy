"""Guide length, CLI media, and final-output pick for the LTX 2.3 render graphs.

The tower smoke of ltx23_lipsync_v08 shortened the latent to 73 frames and
left PrimitiveInt 6214 at 145, refused image+audio (and a guide video), and
copied a guide preview instead of the SaveVideo that carries audio.

Run: python -m pytest tests/test_ltx23_render_io.py -q
"""

from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.__main__ import cmd_run
from master_agent.comfy.client import ComfyClient
from master_agent.comfy.workflow_patcher import (
    load_and_patch_workflow,
    load_workflow_template,
    media_wiring_error,
    variant_final_node_id,
)
from master_agent.orchestrator.state import RunState
from master_agent.orchestrator.talking import media_route_error

DEV_FP8 = "ltx-2.3-22b-dev-fp8.safetensors"
RENDER_VARIANTS = (
    "ltx23_lipsync_v08",
    "air_render_030",
    "air_render_050",
    "air_render_businesswoman",
)


def _resolve_on_tower(name: str | None) -> Path | None:
    if name == DEV_FP8:
        return Path("checkpoints") / DEV_FP8
    return None


def _patched(variant: str, **kwargs):
    template = load_workflow_template("ltx23_lipsync_v08")
    try:
        own = load_workflow_template(variant)
    except FileNotFoundError:
        own = None

    def _load(requested: str):
        if requested == variant and own is not None:
            return copy.deepcopy(own)
        if requested == variant:
            return copy.deepcopy(template)
        return load_workflow_template(requested)

    with (
        patch(
            "master_agent.comfy.workflow_patcher.resolve_model_path",
            side_effect=_resolve_on_tower,
        ),
        patch(
            "master_agent.comfy.workflow_patcher.load_workflow_template",
            side_effect=_load,
        ),
    ):
        return load_and_patch_workflow(variant, **kwargs)


@pytest.mark.parametrize("variant", RENDER_VARIANTS)
def test_guide_frame_cap_follows_latent_length(variant: str):
    workflow, meta = _patched(variant, prompt="clay", seed=1, duration_s=3.0)
    assert meta["frames"] == 73
    assert workflow["6214"]["class_type"] == "PrimitiveInt"
    assert workflow["6214"]["inputs"]["value"] == 73
    assert workflow["6202:6062"]["inputs"]["length"] == 73
    assert workflow["6202:6156"]["inputs"]["value"] == 24
    assert workflow["6094"]["inputs"]["frame_load_cap"] == ["6214", 0]
    assert workflow["6059"]["inputs"]["frame_load_cap"] == ["6214", 0]
    assert workflow["5893"]["inputs"]["frame_load_cap"] == ["6214", 0]


@pytest.mark.parametrize("variant", RENDER_VARIANTS)
def test_image_audio_and_optional_guide_video_are_routed(variant: str):
    bare = media_route_error(variant)
    assert bare is not None
    assert "--image" in bare
    assert "--audio" in bare
    assert "source video" not in bare
    assert media_route_error(variant, has_image=True, has_audio=True) is None
    assert media_route_error(variant, has_image=True, has_audio=True, has_video=True) is None

    workflow, _meta = _patched(
        variant,
        prompt="clay",
        seed=1,
        duration_s=3.0,
        image_name="plate.webp",
        audio_name="line.mp3",
    )
    for nid in ("6108", "6102", "6139"):
        assert workflow[nid]["inputs"]["image"] == "plate.webp"
    assert workflow["5883"]["inputs"]["audio"] == "line.mp3"
    assert workflow["6094"]["inputs"]["video"] == "BusinessWoman_CLAY.mp4"
    assert workflow["6059"]["inputs"]["video"] == "BusinessWoman_MASK.mp4"
    assert workflow["5893"]["inputs"]["video"] == "BusinessWoman_DEPTH.mp4"
    assert media_wiring_error(workflow, image_name="plate.webp", audio_name="line.mp3") is None

    guided, _meta = _patched(
        variant,
        prompt="clay",
        seed=1,
        duration_s=3.0,
        image_name="plate.webp",
        audio_name="line.mp3",
        video_name="guide.mp4",
    )
    assert guided["6094"]["inputs"]["video"] == "guide.mp4"
    assert guided["6059"]["inputs"]["video"] == "BusinessWoman_MASK.mp4"
    assert guided["5893"]["inputs"]["video"] == "BusinessWoman_DEPTH.mp4"
    assert (
        media_wiring_error(
            guided,
            image_name="plate.webp",
            audio_name="line.mp3",
            video_name="guide.mp4",
        )
        is None
    )


def test_lipsync_still_requires_a_source_video():
    err = media_route_error("lipsync", has_image=True, has_audio=True)
    assert err is not None
    assert "source video" in err


def test_cli_run_accepts_image_and_audio_for_lipsync_v08(monkeypatch, capsys):
    captured: dict = {}

    def _dry(*_args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("master_agent.orchestrator.pipeline.dry_run_pipeline", _dry)
    rc = cmd_run(
        Namespace(
            request="businesswoman speaks",
            variant="ltx23_lipsync_v08",
            duration=3.0,
            duration_set=True,
            quality="draft",
            seed=1,
            width=1408,
            height=768,
            video=None,
            image="plate.webp",
            audio="line.mp3",
            no_judge=True,
            max_judge_rounds=1,
            storyboard="off",
            llm_panel=None,
            panel_judge=None,
            max_full_judge_rounds=None,
            dry_run=True,
            self_improve_dry=False,
            power_mode=False,
            no_power_mode=False,
            upscale=None,
            no_interview=True,
            attach=None,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert captured["image_name"] == "plate.webp"
    assert captured["audio_name"] == "line.mp3"
    assert captured["video_name"] is None
    assert "source video" not in out
    assert "no node consumes" not in out


def test_cli_run_with_guide_video_is_not_refused(monkeypatch, capsys):
    captured: dict = {}

    def _dry(*_args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("master_agent.orchestrator.pipeline.dry_run_pipeline", _dry)
    rc = cmd_run(
        Namespace(
            request="businesswoman speaks",
            variant="ltx23_lipsync_v08",
            duration=3.0,
            duration_set=True,
            quality="draft",
            seed=1,
            width=768,
            height=512,
            video="guide.mp4",
            image="plate.webp",
            audio="line.mp3",
            no_judge=True,
            max_judge_rounds=1,
            storyboard="off",
            llm_panel=None,
            panel_judge=None,
            max_full_judge_rounds=None,
            dry_run=True,
            self_improve_dry=False,
            power_mode=False,
            no_power_mode=False,
            upscale=None,
            no_interview=True,
            attach=None,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert captured["video_name"] == "guide.mp4"
    assert "no node consumes" not in out
    assert "source video" not in out


def test_final_output_is_the_manifest_save_node_not_the_guide_preview():
    assert variant_final_node_id("ltx23_lipsync_v08") == "6109"
    assert variant_final_node_id("air_render_030") == "6109"
    history = {
        "outputs": {
            "6057": {"gifs": [{"filename": "ltx_00001.mp4", "type": "output"}]},
            "6219": {"gifs": [{"filename": "ltx_00002.mp4", "type": "output"}]},
            "6109": {"videos": [{"filename": "ltx_00002_.mp4", "type": "output"}]},
        }
    }
    files = ComfyClient.extract_video_files(history)
    assert [item["filename"] for item in files] == [
        "ltx_00001.mp4",
        "ltx_00002.mp4",
        "ltx_00002_.mp4",
    ]
    chosen = ComfyClient.choose_final_output(files, final_node_id="6109")
    assert chosen[0]["filename"] == "ltx_00002_.mp4"
    assert chosen[0]["node_id"] == "6109"


def test_final_output_falls_back_to_audio_at_the_requested_length():
    files = [
        {"filename": "ltx_00001.mp4", "node_id": "6057", "subfolder": "", "type": "output"},
        {"filename": "ltx_00002.mp4", "node_id": "6219", "subfolder": "", "type": "output"},
        {"filename": "ltx_00002_.mp4", "node_id": "6109", "subfolder": "", "type": "output"},
    ]

    def _probe(info: dict) -> dict:
        if info["filename"] == "ltx_00002_.mp4":
            return {"has_audio": True, "duration_s": 3.04, "frames": 73}
        return {"has_audio": False, "duration_s": 6.0, "frames": 145}

    chosen = ComfyClient.choose_final_output(
        files,
        final_node_id="missing",
        duration_s=3.0,
        frames=73,
        probe=_probe,
    )
    assert chosen[0]["filename"] == "ltx_00002_.mp4"


def test_orchestrator_orders_the_manifest_final_before_guide_previews():
    from master_agent.orchestrator.machine import Orchestrator

    orch = Orchestrator(client=object())
    st = RunState(request="speak", variant="ltx23_lipsync_v08", duration_s=3.0, frames=73)
    files = [
        {"filename": "ltx_00002.mp4", "node_id": "6219", "subfolder": "", "type": "output"},
        {"filename": "ltx_00002_.mp4", "node_id": "6109", "subfolder": "", "type": "output"},
    ]
    ordered = orch._order_final_output(st, files)
    assert ordered[0]["filename"] == "ltx_00002_.mp4"
