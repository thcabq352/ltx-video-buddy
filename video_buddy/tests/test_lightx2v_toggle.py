"""LightX2V distill LoRA toggle: graph_ops.set_lightx2v / apply_lightx2v and its surfaces."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from master_agent.comfy.cli_run import prepare_run
from master_agent.comfy.graph_ops import (
    LIGHTX2V_DEFAULT_STRENGTH,
    apply_lightx2v,
    looks_like_wan_graph,
    resolve_lightx2v_lora,
    set_lightx2v,
)
from master_agent.comfy.workflow_patcher import load_and_patch_workflow, load_workflow_template
from master_agent.orchestrator.state import RunState
from master_agent.provenance import apply_sidecar_to_state, build_clip_provenance

LX = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
LX_WIN = "wan\\" + LX


def _lora_info(*names: str) -> dict:
    return {
        "LoraLoaderModelOnly": {
            "input": {"required": {"model": ["MODEL", {}], "lora_name": [list(names), {}]}},
            "output": ["MODEL"],
        }
    }


def _bare_wan_graph() -> dict:
    """Two Wan UNETs straight into two samplers; no LoRA nodes."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_high_noise.safetensors"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_low_noise.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5.safetensors", "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "x"}},
        "5": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["1", 0], "positive": ["4", 0]}},
        "6": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["2", 0], "positive": ["4", 0]}},
    }


def _assert_links_resolve(wf: dict) -> None:
    for nid, node in wf.items():
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                assert str(value[0]) in wf, f"{nid}.{key} -> missing node {value[0]}"


def _lightx2v_slots(wf: dict) -> dict[str, dict]:
    return {
        nid: node["inputs"]["lora_2"]
        for nid, node in wf.items()
        if node.get("class_type") == "Power Lora Loader (rgthree)"
    }


def test_wan22_template_is_wan_graph_with_baked_power_lora_slots():
    wf = load_workflow_template("wan22")
    assert looks_like_wan_graph(wf)
    slots = _lightx2v_slots(wf)
    assert set(slots) == {"135", "136"}
    assert all("lightx2v" in s["lora"].lower() for s in slots.values())


def test_enable_switches_power_lora_slots_when_present():
    wf = load_workflow_template("wan22")
    wf["136"]["inputs"]["lora_2"]["on"] = False
    report = set_lightx2v(wf, enabled=True, lora_name=LX_WIN, strength=0.8)
    assert report["action"] == "enabled" and report["enabled"] is True
    assert report["nodes"] == ["135", "136"]
    for slot in _lightx2v_slots(wf).values():
        assert slot == {"on": True, "lora": LX_WIN, "strength": 0.8}
    assert wf["136"]["inputs"]["lora_3"]["on"] is True
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]


def test_enable_keeps_backslash_name_and_fixes_wrong_folder():
    wf = load_workflow_template("wan22")
    set_lightx2v(wf, enabled=True, lora_name="wan/" + LX)
    assert wf["136"]["inputs"]["lora_2"]["lora"] == LX_WIN
    set_lightx2v(wf, enabled=True, lora_name=LX)
    assert wf["136"]["inputs"]["lora_2"]["lora"] == LX


def test_disable_switches_slots_off_and_keeps_links():
    wf = load_workflow_template("wan22")
    before = copy.deepcopy(wf)
    report = set_lightx2v(wf, enabled=False, lora_name=None)
    assert report["action"] == "disabled" and report["enabled"] is False
    assert all(s["on"] is False for s in _lightx2v_slots(wf).values())
    assert wf["136"]["inputs"]["lora_3"] == before["136"]["inputs"]["lora_3"]
    for nid in ("168", "169", "115"):
        assert wf[nid]["inputs"]["model"] == before[nid]["inputs"]["model"]
    _assert_links_resolve(wf)


def test_absent_file_is_noop_with_reason():
    wf = load_workflow_template("wan22")
    before = copy.deepcopy(wf)
    report = set_lightx2v(wf, enabled=True, lora_name=None)
    assert wf == before
    assert report["action"] == "noop" and report["enabled"] is False
    assert "not registered" in report["reason"]


def test_absent_file_on_bare_graph_does_not_inject():
    wf = _bare_wan_graph()
    before = copy.deepcopy(wf)
    report = apply_lightx2v(wf, enabled=True, object_info=_lora_info("other.safetensors"))
    assert wf == before
    assert report["action"] == "noop"
    assert report["message"].startswith("lightx2v no-op: LightX2V LoRA not registered")


def test_inject_after_each_wan_unet_then_disable_restores_links():
    wf = _bare_wan_graph()
    original = copy.deepcopy(wf)
    report = set_lightx2v(wf, enabled=True, lora_name=LX)
    assert report["action"] == "enabled"
    new_ids = report["nodes"]
    assert len(new_ids) == 2
    for nid, unet in zip(new_ids, ("1", "2")):
        node = wf[nid]
        assert node["class_type"] == "LoraLoaderModelOnly"
        assert node["inputs"] == {
            "model": [unet, 0],
            "lora_name": LX,
            "strength_model": LIGHTX2V_DEFAULT_STRENGTH,
        }
    assert wf["5"]["inputs"]["model"] == [new_ids[0], 0]
    assert wf["6"]["inputs"]["model"] == [new_ids[1], 0]
    _assert_links_resolve(wf)

    report = set_lightx2v(wf, enabled=False, lora_name=None)
    assert report["action"] == "disabled"
    assert wf == original
    _assert_links_resolve(wf)


def test_disable_removes_core_lora_and_rewires_model_and_clip():
    wf = _bare_wan_graph()
    wf["7"] = {
        "class_type": "LoraLoader",
        "inputs": {"model": ["1", 0], "clip": ["3", 0], "lora_name": LX, "strength_model": 1.0},
    }
    wf["5"]["inputs"]["model"] = ["7", 0]
    wf["4"]["inputs"]["clip"] = ["7", 1]
    set_lightx2v(wf, enabled=False, lora_name=None)
    assert "7" not in wf
    assert wf["5"]["inputs"]["model"] == ["1", 0]
    assert wf["4"]["inputs"]["clip"] == ["3", 0]
    _assert_links_resolve(wf)


@pytest.mark.parametrize("enabled", [True, False])
def test_toggle_is_idempotent(enabled):
    for make in (_bare_wan_graph, lambda: load_workflow_template("wan22")):
        once = make()
        set_lightx2v(once, enabled=enabled, lora_name=LX)
        twice = copy.deepcopy(once)
        set_lightx2v(twice, enabled=enabled, lora_name=LX)
        assert twice == once


def test_enable_disable_enable_round_trip():
    for make in (_bare_wan_graph, lambda: load_workflow_template("wan22")):
        on = make()
        set_lightx2v(on, enabled=True, lora_name=LX)
        cycled = copy.deepcopy(on)
        set_lightx2v(cycled, enabled=False, lora_name=None)
        set_lightx2v(cycled, enabled=True, lora_name=LX)
        assert cycled == on
        _assert_links_resolve(cycled)


def test_non_wan_graph_is_left_alone():
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.3.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    before = copy.deepcopy(wf)
    report = set_lightx2v(wf, enabled=True, lora_name=LX)
    assert wf == before and report["action"] == "noop"


@pytest.mark.parametrize("bad", [0, -0.5, 2.5])
def test_strength_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        set_lightx2v(_bare_wan_graph(), enabled=True, lora_name=LX, strength=bad)


def test_resolve_prefers_object_info_choices(monkeypatch):
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda spec: None)
    assert resolve_lightx2v_lora(_lora_info("a.safetensors", LX_WIN)) == LX_WIN
    assert resolve_lightx2v_lora(_lora_info("a.safetensors")) is None


def test_resolve_falls_back_to_disk(monkeypatch, tmp_path):
    local = tmp_path / "models" / "loras" / LX
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda spec: local)
    assert resolve_lightx2v_lora({}) == LX
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda spec: None)
    assert resolve_lightx2v_lora({}) is None


def _patch_wan22(**kw):
    return load_and_patch_workflow("wan22", prompt="rain", duration_s=3.0, seed=1, **kw)


def test_patcher_wan22_default_on_without_stacking_a_second_lora(monkeypatch):
    monkeypatch.setattr(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        lambda name: Path("fake.safetensors"),
    )
    wf, meta = _patch_wan22(object_info=_lora_info(LX_WIN))
    assert meta["lightx2v"]["action"] == "enabled"
    assert meta["lightx2v"]["strength"] == LIGHTX2V_DEFAULT_STRENGTH
    assert all(s["on"] for s in _lightx2v_slots(wf).values())
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert wf["168"]["inputs"]["model"] == ["136", 0]
    _assert_links_resolve(wf)


def test_patcher_wan22_off_and_strength():
    wf, meta = _patch_wan22(object_info=_lora_info(LX_WIN), lightx2v=False)
    assert meta["lightx2v"]["action"] == "disabled"
    assert all(s["on"] is False for s in _lightx2v_slots(wf).values())
    wf, meta = _patch_wan22(object_info=_lora_info(LX_WIN), lightx2v_strength=1.0)
    assert all(s["strength"] == 1.0 for s in _lightx2v_slots(wf).values())


def test_patcher_wan22_absent_file_reports_noop():
    wf, meta = _patch_wan22(object_info=_lora_info("other.safetensors"))
    assert meta["lightx2v"]["action"] == "noop"
    assert all(s["on"] for s in _lightx2v_slots(wf).values())


def test_patcher_leaves_ltx_variants_without_lightx2v_meta():
    _wf, meta = load_and_patch_workflow("base", prompt="rain", duration_s=1.0, seed=1)
    assert "lightx2v" not in meta


def test_prepare_run_raw_toggles_when_asked(capsys):
    wf = _bare_wan_graph()
    out = prepare_run("raw", workflow=wf, object_info=_lora_info(LX), lightx2v=True)
    assert [n for n in out.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert "INFO  lightx2v on" in capsys.readouterr().out
    off = prepare_run("raw", workflow=out, object_info=_lora_info(LX), lightx2v=False)
    assert off == _bare_wan_graph()
    untouched = prepare_run("raw", workflow=wf, object_info=_lora_info(LX))
    assert untouched == _bare_wan_graph()


def test_provenance_records_and_restores_lightx2v():
    st = RunState(request="rain", variant="wan22", seed=1, lightx2v=False, lightx2v_strength=0.8)
    st.workflow_meta = {
        "lightx2v": {
            "enabled": False,
            "strength": 0.8,
            "lora": None,
            "action": "disabled",
            "reason": "native Wan 2.2 wants about 20 steps at CFG 3.5; sampler not retuned",
        }
    }
    payload = build_clip_provenance(st, omit_hash=True)
    assert payload["params"]["lightx2v"] == {
        "enabled": False,
        "requested": False,
        "strength": 0.8,
        "lora": None,
        "action": "disabled",
        "reason": "native Wan 2.2 wants about 20 steps at CFG 3.5; sampler not retuned",
    }
    fresh = RunState(request="rain")
    apply_sidecar_to_state(fresh, payload)
    assert fresh.lightx2v is False and fresh.lightx2v_strength == 0.8


def test_provenance_omits_lightx2v_for_non_wan():
    st = RunState(request="rain", variant="base", seed=1)
    assert "lightx2v" not in build_clip_provenance(st, omit_hash=True)["params"]


def test_create_video_validates_and_forwards_lightx2v(monkeypatch, tmp_path):
    from master_agent import agent_api

    bad = agent_api.create_video("rain", lightx2v_strength=3.0)
    assert bad["status"] == "error" and "lightx2v_strength" in bad["error"]
    assert agent_api.create_video("rain", lightx2v="off")["status"] == "error"

    captured: dict = {}
    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.dry_run_pipeline",
        lambda *_a, **kw: captured.update(kw) or 0,
    )
    out = agent_api.create_video("rain", dry_run=True, lightx2v=False, lightx2v_strength=0.7)
    assert out["ok"] is True
    assert captured["lightx2v"] is False and captured["lightx2v_strength"] == 0.7


def test_cli_flags_parse():
    import argparse

    from master_agent.cli.common import _add_lightx2v_flags, _lightx2v_flag

    p = argparse.ArgumentParser()
    _add_lightx2v_flags(p)
    assert _lightx2v_flag(p.parse_args([])) is None
    args = p.parse_args(["--lightx2v", "off", "--lightx2v-strength", "0.8"])
    assert _lightx2v_flag(args) is False and args.lightx2v_strength == 0.8
    assert _lightx2v_flag(p.parse_args(["--lightx2v", "on"])) is True


def test_capability_row_is_wired():
    from master_agent.comfy.capabilities import probe_capabilities

    row = next(r for r in probe_capabilities({}) if r.capability.id == "lightx2v")
    assert row.verdict == "yes"
