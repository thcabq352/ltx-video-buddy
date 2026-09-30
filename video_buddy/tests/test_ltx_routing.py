"""Local LTX 2.3 Pro / 2.5 routing lock. No Comfy, no GPU.

Run: python -m pytest tests/test_ltx_routing.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from master_agent.orchestrator.director import choose_variant, rule_based_variant
from master_agent.orchestrator.ltx_routing import (
    LTX23_PRO_CHECKPOINT,
    LTX23_PRO_DIFFUSION,
    LTX2_DETAILER_LORA,
    LTX2_UNION_CONTROL_LORA,
)

ROOT = Path(__file__).resolve().parents[1]


def test_new_scene_and_synced_dialogue_prefer_ltx25():
    assert rule_based_variant("a new scene of rain on a window") == "ltx25_t2v_i2v"
    assert rule_based_variant("opening scene on the pier") == "ltx25_t2v_i2v"
    assert rule_based_variant("synced dialogue in the kitchen") == "ltx25_a2v"


def test_retake_and_temporal_extend_stay_on_ltx23_pro():
    assert rule_based_variant("retake the plate, use ltx 2.5") == "directors"
    assert rule_based_variant("extend the shot by two seconds") == "directors"
    assert rule_based_variant("extend this video a little") == "directors"


def test_canvas_extend_and_existing_routes_stay():
    assert rule_based_variant("extend the canvas to vertical") == "ltx23_inoutpaint"
    assert rule_based_variant("extend the frame") == "ltx23_inoutpaint"
    assert rule_based_variant("cinematic brand film with three scenes") == "directors"
    assert rule_based_variant("rain on a window") == "base"
    assert rule_based_variant("ltx 2.5 alley push-in") == "ltx25_t2v_i2v"
    assert rule_based_variant("multi-cut the chase") == "directors"


def test_lock_beats_an_llm_that_picks_ltx25_for_retake():
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", True), patch(
        "master_agent.orchestrator.director._llm_variant",
        return_value="ltx25_t2v_i2v",
    ):
        assert choose_variant("retake this plate") == ("directors", "ltx-lock")
        assert choose_variant("a new scene of the alley") == ("ltx25_t2v_i2v", "ltx-lock")
        assert choose_variant("synced dialogue over the table") == ("ltx25_a2v", "ltx-lock")


def test_source_video_retake_uses_lipsync_not_ltx25():
    assert choose_variant("ltx 2.5 retake the last second", has_video=True) == (
        "lipsync",
        "ltx-lock",
    )
    assert choose_variant("extend the canvas to vertical", has_video=True) == (
        "ltx23_inoutpaint",
        "input",
    )


def test_still_plus_voice_retake_does_not_become_a2v():
    assert choose_variant(
        "retake the plate", has_image=True, has_audio=True
    ) == ("directors", "ltx-lock")
    assert choose_variant(
        "lip sync this photo", has_image=True, has_audio=True
    ) == ("ltx25_a2v", "input")


def test_pro_checkpoint_names_and_v1_loras_still_authored():
    assert LTX23_PRO_DIFFUSION == "LTX-2.3-dev-Q4_K_S.gguf"
    assert LTX23_PRO_CHECKPOINT == "ltx-2.3-22b-dev-fp8.safetensors"
    api = json.loads(
        (ROOT / "workflows" / "260603_LTX2-3_3D-RENDERING_LIP-SYNC_v08_api.json").read_text()
    )
    loras = [
        node["inputs"]["lora_name"]
        for node in api.values()
        if isinstance(node, dict) and "lora_name" in (node.get("inputs") or {})
    ]
    assert LTX2_UNION_CONTROL_LORA in loras
    assert LTX2_DETAILER_LORA in loras
    assert "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" not in loras
