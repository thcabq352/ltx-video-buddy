"""Optional accelerator (TeaCache) inject + bypass. No GPU, no pack install.

Run: python -m pytest tests/test_accelerator_bypass.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from master_agent.comfy.graph_ops import (
    TEACACHE_CLASS,
    bypass_optional_accelerators,
    ensure_teacache,
    is_optional_accelerator,
    teacache_defaults,
    is_optional_node,
)
from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.cli_run import prepare_run
from master_agent.comfy.workflow_patcher import load_and_patch_workflow, load_workflow_template


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


def _teacache_object_info(*, max_skip_steps: bool = False) -> dict:
    required = {
        "model": ["MODEL", {}],
        "model_type": [["ltxv", "flux", "hidream"], {}],
        "rel_l1_thresh": ["FLOAT", {"default": 0.3}],
        "start_percent": ["FLOAT", {"default": 0.0}],
        "end_percent": ["FLOAT", {"default": 1.0}],
        "cache_device": [["cuda", "cpu"], {}],
    }
    if max_skip_steps:
        required["max_skip_steps"] = ["INT", {"default": 1}]
    info = dict(OBJECT_INFO)
    info.update(
        {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": ["STRING", {}]}},
                "output": ["MODEL"],
            },
            "LoraLoaderModelOnly": {
                "input": {
                    "required": {
                        "model": ["MODEL", {}],
                        "lora_name": ["STRING", {}],
                        "strength_model": ["FLOAT", {}],
                    }
                },
                "output": ["MODEL"],
            },
            "EmptyLTXVLatentVideo": {
                "input": {"required": {"length": ["INT", {}]}},
                "output": ["LATENT"],
            },
            "MultimodalGuider": {
                "input": {"required": {"model": ["MODEL", {}]}},
                "output": ["GUIDER"],
            },
            TEACACHE_CLASS: {
                "input": {"required": required},
                "output": ["MODEL"],
            },
        }
    )
    return info


def _ltx_mini_graph() -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "x.safetensors"},
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "y.safetensors",
                "strength_model": 0.6,
            },
        },
        "20": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": 9},
        },
        "32": {
            "class_type": "MultimodalGuider",
            "inputs": {"model": ["4", 0]},
        },
    }


def test_teacache_defaults_ltx_and_optional_max_skip():
    base = teacache_defaults(_teacache_object_info())
    assert base["model_type"] == "ltxv"
    assert base["rel_l1_thresh"] == 0.06
    assert base["start_percent"] == 0.0
    assert base["end_percent"] == 1.0
    assert "max_skip_steps" not in base
    with_skip = teacache_defaults(_teacache_object_info(max_skip_steps=True))
    assert with_skip["max_skip_steps"] == 3


def test_ensure_teacache_inserts_when_registered():
    wf = _ltx_mini_graph()
    info = _teacache_object_info()
    nid = ensure_teacache(wf, info)
    assert nid is not None
    node = wf[nid]
    assert node["class_type"] == "TeaCache"
    assert node["inputs"]["model"] == ["4", 0]
    assert node["inputs"]["model_type"] == "ltxv"
    assert node["inputs"]["rel_l1_thresh"] == 0.06
    assert node["inputs"]["start_percent"] == 0.0
    assert node["inputs"]["end_percent"] == 1.0
    assert "max_skip_steps" not in node["inputs"]
    assert wf["32"]["inputs"]["model"] == [nid, 0]
    report = validate_workflow(wf, info, file_label="tea-in")
    assert report.ok
    assert nid in wf


def test_ensure_teacache_keeps_and_refreshes_existing():
    wf = _ltx_mini_graph()
    wf["2"] = {
        "class_type": "TeaCache",
        "inputs": {
            "model": ["4", 0],
            "model_type": "wan",
            "rel_l1_thresh": 0.3,
            "start_percent": 0.2,
            "end_percent": 0.8,
        },
    }
    wf["32"]["inputs"]["model"] = ["2", 0]
    info = _teacache_object_info()
    assert ensure_teacache(wf, info) == "2"
    assert wf["2"]["inputs"]["model_type"] == "ltxv"
    assert wf["2"]["inputs"]["rel_l1_thresh"] == 0.06
    assert wf["2"]["inputs"]["start_percent"] == 0.0
    assert wf["2"]["inputs"]["end_percent"] == 1.0
    assert wf["32"]["inputs"]["model"] == ["2", 0]


def test_ensure_teacache_absent_does_not_insert():
    wf = _ltx_mini_graph()
    assert ensure_teacache(wf, {}) is None
    assert ensure_teacache(wf, OBJECT_INFO) is None
    assert all(n.get("class_type") != "TeaCache" for n in wf.values() if isinstance(n, dict))


def test_leftover_teacache_still_bypasses_when_absent():
    wf = _ltx_mini_graph()
    wf["2"] = {
        "class_type": "TeaCache",
        "inputs": {"model": ["4", 0], "rel_l1_thresh": 0.06},
    }
    wf["32"]["inputs"]["model"] = ["2", 0]
    info = _teacache_object_info()
    info.pop(TEACACHE_CLASS, None)
    assert ensure_teacache(wf, info) is None
    report = validate_workflow(wf, info, file_label="tea-left")
    assert report.ok
    assert "2" not in wf
    assert wf["32"]["inputs"]["model"] == ["4", 0]
    assert any("TeaCache" in str(w) for w in report.warnings)


def test_ensure_teacache_skips_non_ltx_graph():
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1, "steps": 8}},
    }
    assert ensure_teacache(wf, _teacache_object_info()) is None
    assert all(n.get("class_type") != "TeaCache" for n in wf.values() if isinstance(n, dict))


def test_patcher_inserts_teacache_on_base_when_registered():
    info = _teacache_object_info()
    with patch(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        return_value=Path("fake.safetensors"),
    ):
        wf, _meta = load_and_patch_workflow(
            "base",
            prompt="garden proof",
            frames=9,
            seed=1,
            object_info=info,
        )
    teas = [
        (nid, node)
        for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "TeaCache"
    ]
    assert len(teas) == 1
    nid, node = teas[0]
    assert node["inputs"]["model"] == ["4", 0]
    assert node["inputs"]["model_type"] == "ltxv"
    assert node["inputs"]["rel_l1_thresh"] == 0.06
    assert wf["32"]["inputs"]["model"] == [nid, 0]


def test_patcher_skips_teacache_when_absent():
    with patch(
        "master_agent.comfy.workflow_patcher.resolve_model_path",
        return_value=Path("fake.safetensors"),
    ):
        wf, _meta = load_and_patch_workflow(
            "base",
            prompt="garden proof",
            frames=9,
            seed=1,
            object_info={},
        )
    assert all(
        not (isinstance(n, dict) and n.get("class_type") == "TeaCache") for n in wf.values()
    )
    assert wf["32"]["inputs"]["model"] == ["4", 0]


def test_base_template_has_no_baked_teacache():
    raw = load_workflow_template("base")
    assert all(
        not (isinstance(n, dict) and n.get("class_type") == "TeaCache") for n in raw.values()
    )


def test_prepare_run_raw_injects_when_registered():
    wf = prepare_run("raw", workflow=_ltx_mini_graph(), object_info=_teacache_object_info())
    teas = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "TeaCache"]
    assert len(teas) == 1
    assert teas[0]["inputs"]["rel_l1_thresh"] == 0.06
    assert teas[0]["inputs"]["model_type"] == "ltxv"


def test_ensure_teacache_rewires_lipsync_guiders():
    wf = load_workflow_template("lipsync")
    info = _teacache_object_info()
    nid = ensure_teacache(wf, info)
    assert nid is not None
    assert wf[nid]["inputs"]["model"] == ["5012", 0]
    assert wf["4828"]["inputs"]["model"] == [nid, 0]
    assert wf["4964"]["inputs"]["model"] == [nid, 0]
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
