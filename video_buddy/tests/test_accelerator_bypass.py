"""Optional accelerator (TeaCache) bypass. No GPU, no pack install.

Run: python -m pytest tests/test_accelerator_bypass.py -q
"""

from __future__ import annotations

from master_agent.comfy.graph_ops import (
    bypass_optional_accelerators,
    is_optional_accelerator,
    is_optional_node,
)
from master_agent.comfy.validator import validate_workflow


OBJECT_INFO = {
    "UNETLoader": {
        "input": {"required": {"unet_name": ["STRING", {}]}},
        "output": ["MODEL"],
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL", {}],
                "seed": ["INT", {}],
                "steps": ["INT", {}],
            }
        },
        "output": ["LATENT"],
    },
}


def _graph(accel_class: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x.safetensors"}},
        "2": {
            "class_type": accel_class,
            "inputs": {"model": ["1", 0], "rel_l1_thresh": 0.3},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {"model": ["2", 0], "seed": 1, "steps": 8},
        },
    }


def test_teacache_and_wan_aliases_are_optional():
    assert is_optional_accelerator("TeaCache")
    assert is_optional_accelerator("WanVideoTeaCache")
    assert is_optional_accelerator("WanVideoTeaCacheKJ")
    assert not is_optional_accelerator("KSampler")
    assert not is_optional_accelerator("NotANode")


def test_bypass_rewires_model_and_drops_node():
    wf = _graph("TeaCache")
    dropped = bypass_optional_accelerators(wf, object_info={})
    assert dropped == [("2", "TeaCache")]
    assert "2" not in wf
    assert wf["3"]["inputs"]["model"] == ["1", 0]


def test_bypass_keeps_node_when_registered():
    info = dict(OBJECT_INFO)
    info["TeaCache"] = {
        "input": {"required": {"model": ["MODEL", {}]}},
        "output": ["MODEL"],
    }
    wf = _graph("TeaCache")
    assert bypass_optional_accelerators(wf, object_info=info) == []
    assert "2" in wf
    assert wf["3"]["inputs"]["model"] == ["2", 0]


def test_missing_teacache_warns_and_validate_passes():
    wf = _graph("TeaCache")
    report = validate_workflow(wf, OBJECT_INFO, file_label="tea")
    assert report.ok
    assert not report.errors
    assert report.warnings
    assert any("TeaCache" in str(w) for w in report.warnings)
    assert "2" not in wf
    assert wf["3"]["inputs"]["model"] == ["1", 0]


def test_missing_wan_teacache_kj_still_bypasses():
    wf = _graph("WanVideoTeaCacheKJ")
    report = validate_workflow(wf, OBJECT_INFO, file_label="wan-tea")
    assert report.ok
    assert "2" not in wf
    assert wf["3"]["inputs"]["model"] == ["1", 0]


def test_unknown_required_class_is_still_error():
    wf = {"9": {"class_type": "NotANode", "inputs": {}}}
    report = validate_workflow(wf, OBJECT_INFO, file_label="hard")
    assert not report.ok
    assert any("NotANode" in str(e) for e in report.errors)


def test_live_tower_optional_names():
    assert is_optional_node("LanPaint_KSampler")
    assert is_optional_node("GetWarpedNoiseFromVideo")
    assert is_optional_node("MMAudioSampler")
    assert is_optional_accelerator("LanPaint_KSampler")
    assert not is_optional_node("WanFunInpaintToVideo")
    assert not is_optional_node("Wan22FunControlToVideo")
    assert not is_optional_node("IPAdapterFaceID")
    assert not is_optional_node("ControlNetLoader")
    assert not is_optional_node("CreateVoronoiMask")
    assert not is_optional_node("WanVideoEmptyMMAudioLatents")


def test_missing_lanpaint_and_warp_bypass():
    for class_type in ("LanPaint_KSampler", "GetWarpedNoiseFromVideo"):
        wf = _graph(class_type)
        report = validate_workflow(wf, OBJECT_INFO, file_label=class_type)
        assert report.ok, report.errors
        assert "2" not in wf
        assert wf["3"]["inputs"]["model"] == ["1", 0]
        assert any(class_type in str(w) for w in report.warnings)
