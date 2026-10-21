"""STG preflight and the ComfyUI / ComfyUI-LTXVideo version floors.

ComfyUI >=0.40 breaks STG on ComfyUI-LTXVideo <1.1.0
(Lightricks/ComfyUI-LTXVideo#579). No ComfyUI, no network, tmp paths only.

Run: python -m pytest tests/test_stg_preflight.py -q
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

from master_agent import automatic_install as ai
from master_agent import setup as setup_mod
from master_agent.comfy import stg_preflight as sp
from master_agent.comfy import updates
from master_agent.comfy import workflow_patcher as wp
from master_agent.comfy.updates import plan_actions, scan_packs

VB = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fresh_install_check_floors", VB / "scripts" / "ci" / "fresh_install_check.py")
fic = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fic
_spec.loader.exec_module(fic)


def _comfy(root: Path, *, comfy: str | None = None, ltx: str | None = None, ltx_folder: str = "ComfyUI-LTXVideo") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# comfy\n", encoding="utf-8")
    if comfy is not None:
        (root / "pyproject.toml").write_text(f'[project]\nname = "ComfyUI"\nversion = "{comfy}"\n', encoding="utf-8")
    if ltx is not None:
        node = root / "custom_nodes" / ltx_folder
        node.mkdir(parents=True, exist_ok=True)
        (node / "pyproject.toml").write_text(f'[project]\nname = "comfyui-ltxvideo"\nversion = "{ltx}"\n', encoding="utf-8")
    return root


def _guided() -> dict:
    return {
        "30": {"class_type": "GuiderParameters", "inputs": {"modality": "AUDIO", "stg": 1.0, "cfg": 7.0}},
        "31": {"class_type": "GuiderParameters", "inputs": {"modality": "VIDEO", "stg": 1.0, "cfg": 3.0}},
        "40": {"class_type": "MultimodalGuider", "inputs": {"parameters": ["31", 0], "skip_blocks": "29"}},
    }


def _stg(workflow: dict) -> list:
    return [n["inputs"]["stg"] for n in workflow.values() if n.get("class_type") == "GuiderParameters"]


# --- decision --------------------------------------------------------------------


@pytest.mark.parametrize("comfy,ltx", [("0.40.0", "1.0.9"), ("0.41.2", "1.0.0"), ("v0.40", "0.9")])
def test_bad_combo_disables(comfy, ltx):
    result = sp.check_stg(comfy, ltx)
    assert result.action == sp.DISABLE
    assert "#579" in result.message
    assert ">=1.1.0" in result.message


@pytest.mark.parametrize(
    "comfy,ltx",
    [("0.40.0", "1.1.0"), ("0.41.0", "1.2.3"), ("0.39.9", "1.0.0"), ("0.3.60", None), (None, "1.1.0")],
)
def test_good_combo_keeps(comfy, ltx):
    assert sp.check_stg(comfy, ltx).action == sp.KEEP


@pytest.mark.parametrize("comfy,ltx", [(None, None), ("0.40.0", None), (None, "1.0.0"), ("nightly", "dev")])
def test_unknown_versions(comfy, ltx):
    result = sp.check_stg(comfy, ltx)
    assert result.action == sp.UNKNOWN
    assert "STG left as authored" in result.message


# --- graph ---------------------------------------------------------------------


def test_apply_bad_combo_sets_stg_zero_and_warns(caplog):
    wf = _guided()
    with caplog.at_level(logging.WARNING, logger=sp.__name__):
        report = sp.apply_stg_preflight(wf, result=sp.check_stg("0.40.0", "1.0.0"))
    assert _stg(wf) == [0.0, 0.0]
    assert report["action"] == sp.DISABLE
    assert report["disabled_nodes"] == ["30", "31"]
    assert wf["40"]["inputs"]["parameters"] == ["31", 0]
    assert any("breaks STG" in r.message for r in caplog.records)


def test_apply_good_combo_leaves_stg_quietly(caplog):
    wf = _guided()
    with caplog.at_level(logging.WARNING, logger=sp.__name__):
        report = sp.apply_stg_preflight(wf, result=sp.check_stg("0.40.0", "1.1.0"))
    assert _stg(wf) == [1.0, 1.0]
    assert report["action"] == sp.KEEP
    assert "disabled_nodes" not in report
    assert caplog.records == []


def test_apply_unknown_warns_and_leaves_stg(caplog):
    wf = _guided()
    with caplog.at_level(logging.WARNING, logger=sp.__name__):
        report = sp.apply_stg_preflight(wf, result=sp.check_stg(None, None))
    assert _stg(wf) == [1.0, 1.0]
    assert report["action"] == sp.UNKNOWN
    assert any("Could not read" in r.message for r in caplog.records)


def test_graph_without_stg_skips_detection(monkeypatch):
    monkeypatch.setattr(sp, "detect_versions", lambda *_a: pytest.fail("no STG, no detection"))
    wf = _guided()
    for nid in ("30", "31"):
        wf[nid]["inputs"]["stg"] = 0.0
    assert sp.apply_stg_preflight(wf) is None
    assert sp.apply_stg_preflight({"1": {"class_type": "KSampler", "inputs": {}}}) is None


# --- detection -------------------------------------------------------------------


def test_detect_versions_reads_pyproject_and_lowercase_registry_folder(tmp_path):
    root = _comfy(tmp_path / "ComfyUI", comfy="0.40.1", ltx="1.0.4", ltx_folder="comfyui-ltxvideo")
    assert sp.detect_versions(root) == ("0.40.1", "1.0.4")
    assert sp.stg_preflight(root).action == sp.DISABLE


def test_detect_comfyui_version_py_fallback(tmp_path):
    root = _comfy(tmp_path / "ComfyUI", ltx="1.1.0")
    (root / "comfyui_version.py").write_text('__version__ = "0.41.0"\n', encoding="utf-8")
    assert sp.detect_versions(root) == ("0.41.0", "1.1.0")
    assert sp.stg_preflight(root).action == sp.KEEP


def test_detect_without_comfy_is_unknown(tmp_path):
    assert sp.detect_versions(tmp_path / "nothing") == (None, None)
    assert sp.stg_preflight(tmp_path / "nothing").action == sp.UNKNOWN


# --- the patcher path ------------------------------------------------------------


@pytest.fixture
def hermetic_weights(monkeypatch):
    monkeypatch.setattr(wp, "resolve_model_path", lambda _n: Path("fake.safetensors"))
    monkeypatch.setattr("master_agent.models.weights.resolve_ltx23_gguf", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)


def _video_stg(wf: dict) -> float:
    return next(
        n["inputs"]["stg"]
        for n in wf.values()
        if n.get("class_type") == "GuiderParameters" and n["inputs"].get("modality") == "VIDEO"
    )


def test_patcher_bad_combo_turns_stg_off_for_the_run(hermetic_weights, monkeypatch):
    monkeypatch.setattr(sp, "detect_versions", lambda *_a: ("0.40.0", "1.0.0"))
    wf, meta = wp.load_and_patch_workflow("base", prompt="p", seed=1, frames=9, stg_scale=1.0)
    assert _video_stg(wf) == 0.0
    assert meta["stg_preflight"]["action"] == sp.DISABLE
    assert meta["stg_scale"] == 0.0


def test_patcher_good_combo_keeps_stg(hermetic_weights, monkeypatch):
    monkeypatch.setattr(sp, "detect_versions", lambda *_a: ("0.40.0", "1.1.0"))
    wf, meta = wp.load_and_patch_workflow("base", prompt="p", seed=1, frames=9, stg_scale=1.0)
    assert _video_stg(wf) == 1.0
    assert meta["stg_preflight"]["action"] == sp.KEEP
    assert meta["stg_scale"] == 1.0


def test_patcher_unknown_versions_keep_stg(hermetic_weights, monkeypatch, caplog):
    monkeypatch.setattr(sp, "detect_versions", lambda *_a: (None, None))
    with caplog.at_level(logging.WARNING, logger=sp.__name__):
        wf, meta = wp.load_and_patch_workflow("base", prompt="p", seed=1, frames=9, stg_scale=1.0)
    assert _video_stg(wf) == 1.0
    assert meta["stg_preflight"]["action"] == sp.UNKNOWN
    assert any("Could not read" in r.message for r in caplog.records)


def test_stg_defaults_stay_on():
    assert wp.GUIDER_AUDIO_DEFAULTS["stg"] == 1.0
    assert wp.GUIDER_VIDEO_DEFAULTS["stg"] == 1.0
    assert wp.GUIDER_VIDEO_DEFAULTS["perturb_attn"] is True


# --- doctor ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "versions,ok,has_fix",
    [(("0.40.0", "1.0.0"), False, True), (("0.40.0", "1.1.0"), True, False), ((None, None), False, False)],
)
def test_doctor_stg_row(monkeypatch, versions, ok, has_fix):
    monkeypatch.setattr(sp, "detect_versions", lambda *_a: versions)
    row = setup_mod.check_stg()
    assert row["name"] == "stg"
    assert row["ok"] is ok
    assert bool(row["fix"]) is has_fix
    assert setup_mod.classify_row(row) in {"OK", "INFO"}
    assert "stg" in {r["name"] for r in setup_mod.snapshot()}


# --- floors: pack_pins.json / automatic install / CI ----------------------------


def test_floors_agree_everywhere():
    pins = json.loads((VB / "master_agent" / "comfy" / "pack_pins.json").read_text(encoding="utf-8"))
    assert pins["comfyui"] == f">={sp.COMFYUI_MIN_VERSION}" == ">=0.40.0"
    ltx = next(n for n in pins["nodes"] if n["name"] == sp.LTXVIDEO_NODE)
    assert ltx["version"] == f">={sp.LTXVIDEO_MIN_VERSION}" == ">=1.1.0"
    assert "commit" not in ltx
    assert ai.NODE_FLOORS == {sp.LTXVIDEO_NODE: sp.LTXVIDEO_MIN_VERSION}
    assert sp.LTXVIDEO_NODE in ai.REQUIRED_NODES
    manifests = (VB / "workflows" / "manifests.yaml").read_text(encoding="utf-8")
    requires = [line for line in manifests.splitlines() if "requires:" in line and "ComfyUI-LTXVideo" in line]
    assert requires and all("ComfyUI-LTXVideo>=1.1.0" in line for line in requires)


@pytest.fixture
def quiet_cli(monkeypatch):
    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: "1.20.0")


def test_floor_pin_style_and_compare():
    assert updates.pin_style(">=1.1.0") == "floor"
    assert updates.pin_style(">= v0.40") == "floor"
    assert updates.floor_version(">=1.1.0") == "1.1.0"
    assert updates.meets_floor("1.1.0", "1.1.0") is True
    assert updates.meets_floor("1.10.0", "1.9.0") is True
    assert updates.meets_floor("1.0.9", "1.1.0") is False
    assert updates.meets_floor(None, "1.1.0") is None


@pytest.mark.parametrize(
    "comfy,ltx,stale",
    [
        ("0.40.0", "1.1.0", set()),
        ("0.42.0", "2.0.0", set()),
        ("0.39.0", "1.1.0", {"ComfyUI"}),
        ("0.40.0", "1.0.0", {"ComfyUI-LTXVideo"}),
        (None, None, set()),
    ],
)
def test_scan_packs_floors(tmp_path, quiet_cli, comfy, ltx, stale):
    ws = _comfy(tmp_path / "ComfyUI", comfy=comfy, ltx=ltx)
    report = scan_packs(ws)
    assert {item["name"] for item in report["stale"]} == stale
    plan = plan_actions(report)
    labels = [a["label"] for a in plan["actions"]]
    if "ComfyUI-LTXVideo" in stale:
        assert "comfy node update ComfyUI-LTXVideo" in labels
    if "ComfyUI" in stale:
        assert "comfy update comfy" in labels
    if not stale:
        assert labels == []


def _installed(monkeypatch):
    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")


def test_step_nodes_updates_ltxvideo_below_floor(tmp_path, monkeypatch):
    ws = _comfy(tmp_path / "ComfyUI", comfy="0.40.0", ltx="1.0.0")
    (ws / "custom_nodes" / "ComfyUI-GGUF").mkdir()
    plan = ai.step_nodes(False, workspace=ws)
    assert plan.status == ai.PLANNED
    assert plan.detail.endswith("node update ComfyUI-LTXVideo")
    assert "node install" not in plan.detail

    _installed(monkeypatch)
    calls = []

    def fake_run(cmd, *, timeout=600.0, env=None):
        calls.append(cmd)
        _comfy(ws, ltx="1.1.2")
        return 0, ""

    monkeypatch.setattr(ai, "_run", fake_run)
    done = ai.step_nodes(True, workspace=ws)
    assert done.status == ai.OK
    assert "ComfyUI-LTXVideo 1.0.0 -> 1.1.2" in done.detail
    assert calls[0][-3:] == ["node", "update", "ComfyUI-LTXVideo"]
    assert ai.step_nodes(True, workspace=ws).status == ai.SKIPPED


def test_step_nodes_fails_when_update_stays_below_floor(tmp_path, monkeypatch):
    ws = _comfy(tmp_path / "ComfyUI", ltx="1.0.0")
    (ws / "custom_nodes" / "ComfyUI-GGUF").mkdir()
    _installed(monkeypatch)
    monkeypatch.setattr(ai, "_run", lambda cmd, **_k: (0, ""))
    result = ai.step_nodes(True, workspace=ws)
    assert result.status == ai.FAILED
    assert "ComfyUI-LTXVideo 1.0.0 (needs >=1.1.0)" in result.detail


def test_step_nodes_newer_ltxvideo_is_left_alone(tmp_path):
    ws = _comfy(tmp_path / "ComfyUI", ltx="1.4.0")
    (ws / "custom_nodes" / "ComfyUI-GGUF").mkdir()
    assert ai.step_nodes(False, workspace=ws).status == ai.SKIPPED


@pytest.mark.parametrize(
    "ltx,folder,want",
    [
        ("1.1.0", "ComfyUI-LTXVideo", None),
        ("1.3.0", "comfyui-ltxvideo", None),
        ("1.0.0", "ComfyUI-LTXVideo", "below the floor >=1.1.0"),
        (None, None, "is not installed"),
    ],
)
def test_ci_node_floors(tmp_path, ltx, folder, want):
    ws = _comfy(tmp_path / "ComfyUI", ltx=ltx, ltx_folder=folder or "ComfyUI-LTXVideo")
    problems = fic.node_floor_problems(ws)
    if want is None:
        assert problems == []
    else:
        assert len(problems) == 1 and want in problems[0]


def test_ci_node_floors_unreadable_version_fails(tmp_path):
    ws = _comfy(tmp_path / "ComfyUI")
    (ws / "custom_nodes" / "ComfyUI-LTXVideo").mkdir(parents=True)
    assert "no readable version" in fic.node_floor_problems(ws)[0]
    assert fic.main(["node-floors", "--workspace", str(ws)]) == 1
    _comfy(ws, comfy="0.40.0", ltx="1.1.0")
    assert fic.main(["node-floors", "--workspace", str(ws)]) == 0
