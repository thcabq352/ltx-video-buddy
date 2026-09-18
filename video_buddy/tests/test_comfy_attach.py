"""Live Comfy attach: buddy.comfy.attach/v1 WorkflowPatchPlan.

No GPU. Run: python -m pytest tests/test_comfy_attach.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from master_agent.comfy.attach import (
    ATTACH_SCHEMA,
    AttachError,
    apply_attach_recipe,
    evaluate_attach_judge_rules,
    load_attach_recipe,
    run_attach,
)
from master_agent.orchestrator.director import choose_variant

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "comfy_attach"
RECIPE_PATH = FIXTURES / "recipe_v1.json"
WORKFLOW_PATH = FIXTURES / "workflow_ltx_wan.json"


def _recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def _workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _object_info() -> dict:
    return {
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}},
        "LoadImage": {"input": {"required": {"image": ["COMBO", []]}}},
        "EmptyLTXVLatentVideo": {
            "input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}]}}
        },
        "Wan22FunControlToVideo": {
            "input": {"required": {"prompt": ["STRING", {}], "control_image": ["IMAGE", {}]}}
        },
    }


def test_load_v1_recipe_from_fixture():
    recipe = load_attach_recipe(RECIPE_PATH)
    assert recipe.schema == ATTACH_SCHEMA
    assert recipe.previs_source == "previs/shots/alley_01.export-patch.json"
    assert recipe.control_pack_present is True
    assert recipe.channels["openpose"].image == "previs/openpose_alley.png"
    assert "slow dolly-in" in (recipe.channels["camera"].prompt_additive or "")


def test_reject_unknown_schema():
    with pytest.raises(AttachError, match="schema"):
        load_attach_recipe({"schema": "buddy.comfy.attach/v0", "control_pack": {}})


def test_apply_openpose_depth_edges_and_camera_additives():
    result = apply_attach_recipe(_workflow(), _recipe(), object_info=_object_info())
    assert result.ok
    assert result.workflow["20"]["inputs"]["image"] == "previs/openpose_alley.png"
    assert result.workflow["21"]["inputs"]["image"] == "previs/depth_alley.png"
    assert result.workflow["22"]["inputs"]["image"] == "previs/edges_alley.png"
    text = result.workflow["10"]["inputs"]["text"]
    assert text.startswith("a rainy neon alley at night")
    assert "OpenPose skeleton" in text
    assert "depth map" in text
    assert "canny edges" in text
    assert "slow dolly-in" in text
    assert "blurry" in result.workflow["11"]["inputs"]["text"]
    assert result.control_pack_used["openpose"] is True
    assert result.control_pack_used["depth"] is True
    assert result.control_pack_used["edges"] is True
    assert result.control_pack_used["camera"] is True


def test_fail_honestly_when_control_nodes_missing():
    bare = {
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "alley"},
            "_meta": {"title": "Positive Prompt"},
        }
    }
    with pytest.raises(AttachError, match="openpose"):
        apply_attach_recipe(bare, _recipe(), object_info=_object_info())


def test_validate_against_object_info_missing_class():
    info = _object_info()
    del info["Wan22FunControlToVideo"]
    with pytest.raises(AttachError, match="Wan22FunControlToVideo"):
        apply_attach_recipe(_workflow(), _recipe(), object_info=info)


def test_director_uses_recipe_preferences_when_pack_present():
    recipe = load_attach_recipe(_recipe())
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False):
        variant, source = choose_variant(
            "rain on a window",
            attach_recipe=recipe,
        )
    assert variant == "wan22"
    assert source == "attach"


def test_run_attach_dry_run_does_not_queue(tmp_path: Path):
    queued: list[dict] = []

    class FakeClient:
        def load_object_info(self, prefer_live=True):
            return _object_info(), "cache"

        def queue_prompt(self, workflow):
            queued.append(workflow)
            raise AssertionError("dry-run must not POST /prompt")

    rec = run_attach(
        recipe=_recipe(),
        workflow=_workflow(),
        client=FakeClient(),
        submit=False,
        runs_dir=tmp_path,
    )
    assert rec["ok"] is True
    assert rec["dry_run"] is True
    assert rec["submitted"] is False
    assert rec["prompt_id"] is None
    assert rec["previs_source"] == "previs/shots/alley_01.export-patch.json"
    assert rec["control_pack_present"] is True
    assert rec["control_pack_used"]["openpose"] is True
    assert queued == []
    written = list(tmp_path.glob("*.json"))
    assert written
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["previs_source"] == rec["previs_source"]
    assert payload["control_pack_present"] is True
    rules = evaluate_attach_judge_rules(payload)
    assert rules["c"] is True
    assert rules["d"] is True


def test_run_attach_submit_flag_posts_prompt(tmp_path: Path):
    class FakeClient:
        def load_object_info(self, prefer_live=True):
            return _object_info(), "cache"

        def queue_prompt(self, workflow):
            assert workflow["20"]["inputs"]["image"] == "previs/openpose_alley.png"
            return "prompt-live-1"

    rec = run_attach(
        recipe=_recipe(),
        workflow=_workflow(),
        client=FakeClient(),
        submit=True,
        runs_dir=tmp_path,
    )
    assert rec["ok"] is True
    assert rec["dry_run"] is False
    assert rec["submitted"] is True
    assert rec["prompt_id"] == "prompt-live-1"


def test_cmd_comfy_attach_dry_run(tmp_path: Path, capsys, monkeypatch):
    from master_agent.__main__ import cmd_comfy

    class FakeClient:
        def load_object_info(self, prefer_live=True):
            return _object_info(), "cache"

        def queue_prompt(self, workflow):
            raise AssertionError("CLI dry-run must not queue")

    monkeypatch.setattr("master_agent.comfy.attach.ComfyClient", lambda *a, **k: FakeClient())
    monkeypatch.setattr("master_agent.__main__.ComfyClient", lambda *a, **k: FakeClient())
    args = Namespace(
        comfy_command="attach",
        recipe=str(RECIPE_PATH),
        workflow_json=str(WORKFLOW_PATH),
        variant=None,
        template=None,
        mode="raw",
        prompt="",
        set=[],
        out=None,
        prepare=False,
        submit=False,
        dry_run=True,
        runs_dir=str(tmp_path),
    )
    rc = cmd_comfy(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "previs_source" in out
    assert "prompt-live" not in out.lower() or '"submitted": false' in out.lower()


def test_judge_rule_d_fails_when_pack_present_but_unused():
    rules = evaluate_attach_judge_rules(
        {
            "previs_source": "x",
            "control_pack_present": True,
            "control_pack_used": {},
        }
    )
    assert rules["c"] is True
    assert rules["d"] is False
