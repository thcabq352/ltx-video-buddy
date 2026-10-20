"""Graph patcher rewiring: graph_ops + workflow_patcher keep API graphs valid.

Covers node insert / remove / bypass, multi-consumer outputs, pass-through
(Reroute) nodes, widget patching by name and field map, missing-node errors,
TeaCache inject-when-registered, seed / size / frames, API-format round
trip, and golden snapshots of real shipped workflows.

Golden snapshots live in tests/fixtures/golden_patch/. Regenerate with
``UPDATE_GOLDEN=1 pytest tests/test_graph_patcher_rewiring.py``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from master_agent.comfy import workflow_patcher as wp
from master_agent.comfy.graph_ops import (
    TEACACHE_CLASS,
    apply_ops,
    bypass_optional_accelerators,
    ensure_teacache,
)
from master_agent.comfy.validator import validate_workflow
from master_agent.config import OBJECT_INFO_CACHE, WORKFLOWS_DIR

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden_patch"


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    return json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))


@pytest.fixture
def teacache_info(object_info) -> dict[str, Any]:
    info = dict(object_info)
    info[TEACACHE_CLASS] = {
        "input": {
            "required": {
                "model": ["MODEL", {}],
                "model_type": [["ltxv", "flux", "hidream"], {}],
                "rel_l1_thresh": ["FLOAT", {"default": 0.3}],
                "start_percent": ["FLOAT", {"default": 0.0}],
                "end_percent": ["FLOAT", {"default": 1.0}],
                "cache_device": [["cuda", "cpu"], {}],
            }
        },
        "output": ["MODEL"],
    }
    return info


@pytest.fixture
def hermetic_weights(monkeypatch):
    """Every named weight 'exists'; no local GGUF / stub LoRA is found."""
    monkeypatch.setattr(wp, "resolve_model_path", lambda _n: Path("fake.safetensors"))
    monkeypatch.setattr("master_agent.models.weights.resolve_ltx23_gguf", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)


def is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def links(wf: dict[str, Any]):
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        for key, value in (node.get("inputs") or {}).items():
            if is_link(value):
                yield str(nid), key, str(value[0]), int(value[1])


def bad_links(wf: dict[str, Any], object_info: dict[str, Any] | None = None) -> list[str]:
    """Links whose source is gone or whose slot exceeds the source's outputs."""
    out = []
    for nid, key, src, slot in links(wf):
        if src not in wf:
            out.append(f"{nid}.{key} -> missing {src}")
            continue
        ct = wf[src].get("class_type")
        outputs = ((object_info or {}).get(ct) or {}).get("output")
        if isinstance(outputs, list) and slot >= len(outputs):
            out.append(f"{nid}.{key} -> {src}[{slot}] but {ct} has {len(outputs)} outputs")
    return out


def type_mismatches(wf: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    out = []
    for nid, key, src, slot in links(wf):
        if src not in wf:
            continue
        dst_spec = (object_info.get(wf[nid].get("class_type")) or {}).get("input") or {}
        spec = (dst_spec.get("required") or {}).get(key) or (dst_spec.get("optional") or {}).get(key)
        outputs = (object_info.get(wf[src].get("class_type")) or {}).get("output") or []
        if not spec or slot >= len(outputs) or not isinstance(spec[0], str):
            continue
        want, got = spec[0], outputs[slot]
        if want != got and "*" not in (want, got):
            out.append(f"{nid}.{key} wants {want}, gets {src}[{slot}] {got}")
    return out


def by_class(wf: dict[str, Any], class_type: str) -> list[str]:
    return [nid for nid, n in wf.items() if isinstance(n, dict) and n.get("class_type") == class_type]


def consumers(wf: dict[str, Any], src: str, slot: int | None = None) -> set[tuple[str, str]]:
    return {(n, k) for n, k, s, sl in links(wf) if s == str(src) and (slot is None or sl == slot)}


# --------------------------------------------------------------------------- fixtures: small graphs


def multi_consumer_graph() -> dict[str, Any]:
    """Checkpoint MODEL feeds a LoRA; the LoRA feeds two guiders; CLIP feeds two encoders."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx.safetensors"}},
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": "d.safetensors", "strength_model": 0.6},
        },
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["1", 1]}},
        "20": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": 9, "batch_size": 1},
        },
        "32": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["4", 0], "positive": ["10", 0], "negative": ["11", 0], "cfg": 1.0},
        },
        "33": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["4", 0], "positive": ["10", 0], "negative": ["11", 0], "cfg": 1.0},
        },
    }


# --------------------------------------------------------------------------- apply_ops


def test_insert_node_between_producer_and_all_consumers_keeps_links_valid(object_info):
    wf = multi_consumer_graph()
    ops = [
        {
            "op": "add_node",
            "node_id": "5",
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["4", 0], "lora_name": "style.safetensors", "strength_model": 1.0},
        },
        {"op": "rewire", "node_id": "32", "input": "model", "from_node": "5", "from_slot": 0},
        {"op": "rewire", "node_id": "33", "input": "model", "from_node": "5"},
    ]
    out, res = apply_ops(wf, ops)
    assert res.ok and not res.errors
    assert consumers(out, "4") == {("5", "model")}
    assert consumers(out, "5") == {("32", "model"), ("33", "model")}
    assert bad_links(out, object_info) == []
    assert type_mismatches(out, object_info) == []
    assert wf == multi_consumer_graph(), "copy_graph=True must not mutate the input"


def test_remove_node_scrubs_every_consumer_link_and_nothing_else():
    wf = multi_consumer_graph()
    out, res = apply_ops(wf, [{"op": "remove_node", "node_id": "10"}])
    assert res.ok
    assert "10" not in out
    for nid in ("32", "33"):
        assert "positive" not in out[nid]["inputs"]
        assert out[nid]["inputs"]["negative"] == ["11", 0]
        assert out[nid]["inputs"]["model"] == ["4", 0]
    assert bad_links(out) == []


def test_removed_required_input_fails_validation_closed(object_info):
    out, _ = apply_ops(multi_consumer_graph(), [{"op": "remove_node", "node_id": "10"}])
    report = validate_workflow(out, object_info, file_label="removed")
    assert not report.ok
    assert any("positive" in str(e) for e in report.errors)


@pytest.mark.parametrize(
    "op, needle",
    [
        ({"op": "set_widget", "node_id": "99", "input": "x", "value": 1}, "node 99 not found"),
        ({"op": "remove_node", "node_id": "99"}, "node 99 not found"),
        ({"op": "delete_input", "node_id": "99", "input": "x"}, "node 99 not found"),
        ({"op": "rewire", "node_id": "99", "input": "m", "from_node": "1"}, "node 99 not found"),
        ({"op": "rewire", "node_id": "32", "input": "model", "from_node": "77"}, "source node 77"),
        ({"op": "rewire", "node_id": "32", "input": "model", "from_node": "1", "from_slot": "x"}, "from_slot"),
        ({"op": "add_node", "node_id": "4", "class_type": "X"}, "already exists"),
        ({"op": "set_widget_by_class", "class_type": "Nope", "input": "a", "value": 1}, "no nodes"),
        (
            {"op": "set_widget_by_class", "class_type": "CFGGuider", "input": "cfg", "value": 2, "index": 5},
            "out of range",
        ),
        (
            {"op": "set_widget_by_class", "class_type": "CFGGuider", "input": "zz", "value": 2, "only_existing": True},
            "not present",
        ),
        ({"op": "set_widget", "node_id": "32", "input": "cfg"}, "needs value"),
    ],
)
def test_missing_node_and_bad_op_errors_leave_graph_untouched(op, needle):
    wf = multi_consumer_graph()
    out, res = apply_ops(wf, [op])
    assert not res.ok
    assert any(needle in e for e in res.errors), res.errors
    assert out == multi_consumer_graph()


def test_set_widget_by_class_index_targets_one_of_many():
    out, res = apply_ops(
        multi_consumer_graph(),
        [{"op": "set_widget_by_class", "class_type": "CFGGuider", "input": "cfg", "value": 4.5, "index": 1}],
    )
    assert res.ok
    assert out["32"]["inputs"]["cfg"] == 1.0
    assert out["33"]["inputs"]["cfg"] == 4.5


def test_max_ops_truncates_and_reports():
    ops = [{"op": "set_widget", "node_id": "32", "input": "cfg", "value": i} for i in range(5)]
    out, res = apply_ops(multi_consumer_graph(), ops, max_ops=2)
    assert out["32"]["inputs"]["cfg"] == 1
    assert any("truncated 3" in s for s in res.skipped)


# --------------------------------------------------------------------------- optional-node bypass


def test_bypass_rewires_every_consumer_of_the_dropped_node():
    wf = multi_consumer_graph()
    wf["5"] = {"class_type": "TeaCache", "inputs": {"model": ["4", 0], "rel_l1_thresh": 0.06}}
    wf["32"]["inputs"]["model"] = ["5", 0]
    wf["33"]["inputs"]["model"] = ["5", 0]
    assert bypass_optional_accelerators(wf, {}) == [("5", "TeaCache")]
    assert consumers(wf, "4") == {("32", "model"), ("33", "model")}
    assert bad_links(wf) == []


def test_bypass_chained_accelerators_collapses_to_the_real_source():
    wf = multi_consumer_graph()
    wf["5"] = {"class_type": "TeaCache", "inputs": {"model": ["4", 0]}}
    wf["6"] = {"class_type": "TorchCompileModel", "inputs": {"model": ["5", 0]}}
    wf["32"]["inputs"]["model"] = ["6", 0]
    dropped = bypass_optional_accelerators(wf, {})
    assert {d[0] for d in dropped} == {"5", "6"}
    assert wf["32"]["inputs"]["model"] == ["4", 0]
    assert bad_links(wf) == []


def test_bypass_through_reroute_on_both_sides():
    wf = multi_consumer_graph()
    wf["7"] = {"class_type": "Reroute", "inputs": {"": ["4", 0]}}
    wf["5"] = {"class_type": "TeaCache", "inputs": {"model": ["7", 0]}}
    wf["8"] = {"class_type": "Reroute", "inputs": {"": ["5", 0]}}
    wf["32"]["inputs"]["model"] = ["8", 0]
    wf["33"]["inputs"]["model"] = ["5", 0]
    bypass_optional_accelerators(wf, {})
    assert wf["8"]["inputs"][""] == ["7", 0]
    assert wf["33"]["inputs"]["model"] == ["7", 0]
    assert wf["32"]["inputs"]["model"] == ["8", 0]
    assert bad_links(wf) == []


def test_bypass_keeps_registered_nodes_and_ignores_non_optional():
    wf = multi_consumer_graph()
    wf["5"] = {"class_type": "TeaCache", "inputs": {"model": ["4", 0]}}
    wf["32"]["inputs"]["model"] = ["5", 0]
    assert bypass_optional_accelerators(wf, {"TeaCache": {}}) == []
    assert bypass_optional_accelerators(multi_consumer_graph(), {}) == []
    assert wf["32"]["inputs"]["model"] == ["5", 0]


def test_bypass_node_without_upstream_link_drops_consumer_inputs():
    wf = {
        "1": {"class_type": "MMAudioModelLoader", "inputs": {"mmaudio_model": "m.safetensors"}},
        "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0], "filename_prefix": "a"}},
    }
    assert bypass_optional_accelerators(wf, {}) == [("1", "MMAudioModelLoader")]
    assert "audio" not in wf["2"]["inputs"]
    assert bad_links(wf) == []


@pytest.mark.xfail(
    strict=True,
    reason="bypass_optional_accelerators always forwards the MODEL-preferred upstream link. "
    "A missing LanPaint_KSampler in sampler position hands its LATENT consumers a MODEL "
    "link, so validate fails with a type mismatch instead of the documented WARN+PASS.",
)
def test_bypass_sampler_class_optional_node_keeps_link_types(object_info):
    info = {k: v for k, v in object_info.items() if k != "LanPaint_KSampler"}
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "p", "clip": ["1", 1]}},
        "20": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "30": {
            "class_type": "LanPaint_KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["10", 0],
                "latent_image": ["20", 0],
                "seed": 1,
            },
        },
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
    }
    bypass_optional_accelerators(wf, info)
    assert type_mismatches(wf, info) == []


# --------------------------------------------------------------------------- TeaCache inject


def test_teacache_inserts_after_lora_and_rewires_all_model_consumers(teacache_info):
    wf = multi_consumer_graph()
    nid = ensure_teacache(wf, teacache_info)
    assert nid is not None
    assert wf[nid]["inputs"]["model"] == ["4", 0]
    assert consumers(wf, "4") == {(nid, "model")}
    assert consumers(wf, nid) == {("32", "model"), ("33", "model")}
    assert consumers(wf, "1", 1) == {("10", "clip"), ("11", "clip")}
    assert bad_links(wf, teacache_info) == []
    assert type_mismatches(wf, teacache_info) == []


def test_teacache_inject_is_idempotent(teacache_info):
    wf = multi_consumer_graph()
    first = ensure_teacache(wf, teacache_info)
    snapshot = copy.deepcopy(wf)
    assert ensure_teacache(wf, teacache_info) == first
    assert wf == snapshot
    assert len(by_class(wf, TEACACHE_CLASS)) == 1


def test_teacache_not_inserted_when_unregistered_or_non_ltx(object_info, teacache_info):
    wf = multi_consumer_graph()
    assert ensure_teacache(wf, object_info) is None
    assert wf == multi_consumer_graph()
    flux = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    assert ensure_teacache(flux, teacache_info) is None
    assert by_class(flux, TEACACHE_CLASS) == []


def test_teacache_with_existing_unlinked_node_wires_it_in(teacache_info):
    wf = multi_consumer_graph()
    wf["50"] = {"class_type": TEACACHE_CLASS, "inputs": {"rel_l1_thresh": 0.4}}
    assert ensure_teacache(wf, teacache_info) == "50"
    assert wf["50"]["inputs"]["model"] == ["4", 0]
    assert wf["50"]["inputs"]["rel_l1_thresh"] == 0.06
    assert consumers(wf, "50") == {("32", "model"), ("33", "model")}
    assert bad_links(wf) == []


@pytest.mark.xfail(
    strict=True,
    reason="_model_source_id picks the last LoRA in dict order, not the tail of the MODEL "
    "chain. When the downstream LoRA has the lower id TeaCache lands mid-chain.",
)
def test_teacache_lands_after_the_tail_of_a_lora_chain(teacache_info):
    wf = multi_consumer_graph()
    wf["4"]["inputs"]["model"] = ["9", 0]
    wf["9"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["1", 0], "lora_name": "first.safetensors", "strength_model": 1.0},
    }
    nid = ensure_teacache(wf, teacache_info)
    assert wf[nid]["inputs"]["model"] == ["4", 0]


@pytest.mark.parametrize("variant", ["base", "eros", "directors", "lipsync"])
def test_patcher_injects_one_teacache_on_ltx_templates(variant, teacache_info, hermetic_weights):
    wf, _ = wp.load_and_patch_workflow(variant, prompt="p", seed=1, frames=9, object_info=teacache_info)
    teas = by_class(wf, TEACACHE_CLASS)
    assert len(teas) == 1
    tea = teas[0]
    src = wf[tea]["inputs"]["model"][0]
    assert wf[src]["class_type"] in ("LoraLoaderModelOnly", "LTXICLoRALoaderModelOnly")
    assert consumers(wf, src, 0) == {(tea, "model")}
    guiders = [n for n in wf if wf[n]["class_type"] in ("MultimodalGuider", "CFGGuider")]
    assert guiders and all(wf[g]["inputs"]["model"] == [tea, 0] for g in guiders)
    assert bad_links(wf, teacache_info) == []


@pytest.mark.parametrize("variant", ["flux", "wan22", "ltx25_t2v_i2v"])
def test_patcher_does_not_inject_teacache_outside_ltx23_variants(variant, teacache_info, hermetic_weights):
    wf, _ = wp.load_and_patch_workflow(variant, prompt="p", seed=1, object_info=teacache_info)
    assert by_class(wf, TEACACHE_CLASS) == []


# --------------------------------------------------------------------------- LoRA insert / stub bypass


def test_ensure_lora_node_rewires_every_model_consumer_only():
    wf = multi_consumer_graph()
    del wf["4"]
    wf["32"]["inputs"]["model"] = ["1", 0]
    wf["33"]["inputs"]["model"] = ["1", 0]
    nid = wp._ensure_lora_node(wf, "mine.safetensors")
    assert consumers(wf, "1", 0) == {(nid, "model")}
    assert consumers(wf, nid) == {("32", "model"), ("33", "model")}
    assert consumers(wf, "1", 1) == {("10", "clip"), ("11", "clip")}


def test_missing_stub_lora_model_only_is_bypassed_for_every_consumer(monkeypatch):
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: None)
    wf = multi_consumer_graph()
    wf["4"]["inputs"]["lora_name"] = "style.safetensors"
    assert wp._bypass_missing_optional_loras(wf) == ["4"]
    assert consumers(wf, "1", 0) == {("32", "model"), ("33", "model")}
    assert bad_links(wf) == []


def test_present_or_non_stub_lora_is_kept(monkeypatch):
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: Path("x"))
    wf = multi_consumer_graph()
    wf["4"]["inputs"]["lora_name"] = "style.safetensors"
    assert wp._bypass_missing_optional_loras(wf) == []
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: None)
    wf2 = multi_consumer_graph()
    assert wp._bypass_missing_optional_loras(wf2) == []
    assert wf2 == multi_consumer_graph()


def test_missing_stub_lora_loader_routes_clip_consumers_to_clip(monkeypatch, object_info):
    monkeypatch.setattr("master_agent.models.weights.find_weight_file", lambda *_a, **_k: None)
    wf = multi_consumer_graph()
    wf["4"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "model": ["1", 0],
            "clip": ["1", 1],
            "lora_name": "camera-orbit.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
        },
    }
    wf["10"]["inputs"]["clip"] = ["4", 1]
    wf["11"]["inputs"]["clip"] = ["4", 1]
    assert wp._bypass_missing_optional_loras(wf) == ["4"]
    assert wf["32"]["inputs"]["model"] == ["1", 0]
    assert wf["10"]["inputs"]["clip"] == ["1", 1]
    assert wf["11"]["inputs"]["clip"] == ["1", 1]
    assert type_mismatches(wf, object_info) == []


def test_h3_template_drops_missing_turbo_lora_and_keeps_model_path(hermetic_weights, object_info):
    raw = wp.load_workflow_template("h3_t2v")
    lora = next(n for n in by_class(raw, "LoraLoaderModelOnly"))
    wf, _ = wp.load_and_patch_workflow("h3_t2v", prompt="p", seed=1)
    assert lora not in wf
    loader_src = raw[lora]["inputs"]["model"][0]
    assert {k for k, _ in consumers(raw, lora)} <= {k for k, _ in consumers(wf, loader_src, 0)}
    assert bad_links(wf, object_info) == []


# --------------------------------------------------------------------------- widgets by name / field map


def test_named_fields_write_by_node_id_list_and_class_index():
    wf = multi_consumer_graph()
    field_map = {
        "cfg": {"node_id": ["32", "33"], "input": "cfg"},
        "prompt": {"class_type": "CLIPTextEncode", "index": 1, "input": "text"},
        "width": {"class_type": "EmptyLTXVLatentVideo", "index": 9, "input": "width"},
        "locked": {"node_id": "20", "input": "batch_size", "overwrite": False},
        "absent": {"node_id": "20", "input": "length"},
    }
    written = wp._apply_named_fields(
        wf, field_map, {"cfg": 2.5, "prompt": "hello", "width": 640, "locked": 9, "absent": None}
    )
    assert written == {("32", "cfg"), ("33", "cfg"), ("11", "text"), ("20", "width")}
    assert wf["32"]["inputs"]["cfg"] == wf["33"]["inputs"]["cfg"] == 2.5
    assert wf["10"]["inputs"]["text"] == "pos"
    assert wf["11"]["inputs"]["text"] == "hello"
    assert wf["20"]["inputs"]["width"] == 640
    assert wf["20"]["inputs"]["batch_size"] == 1
    assert wf["20"]["inputs"]["length"] == 9


def test_named_field_pointing_at_missing_node_is_skipped_not_invented():
    wf = multi_consumer_graph()
    written = wp._apply_named_fields(wf, {"seed": {"node_id": "404", "input": "seed"}}, {"seed": 3})
    assert written == set()
    assert "404" not in wf


def test_write_widget_preserve_links_never_clobbers_a_link():
    node = {"inputs": {"text": ["7", 0], "cfg": 1.0}}
    assert wp._write_widget(node, "text", "x", preserve_links=True) is False
    assert wp._write_widget(node, "missing", 1, preserve_links=True) is False
    assert wp._write_widget(node, "cfg", 2.0, preserve_links=True) is True
    assert node["inputs"] == {"text": ["7", 0], "cfg": 2.0}


def _shipped_manifest_variants() -> list[str]:
    manifests = yaml.safe_load((WORKFLOWS_DIR / "manifests.yaml").read_text(encoding="utf-8"))
    out = []
    for variant in manifests:
        try:
            wp.load_workflow_template(variant)
        except FileNotFoundError:
            continue
        out.append(variant)
    return out


@pytest.mark.parametrize("variant", _shipped_manifest_variants())
def test_manifest_field_map_targets_exist_in_template(variant):
    """A manifest node_id / input that drifted from its template is skipped silently."""
    wf = wp.load_workflow_template(variant)
    meta = wp.variant_manifest(variant)
    problems = []
    for section in ("fields", "inputs", "extras"):
        for logical, spec in (meta.get(section) or {}).items():
            if not isinstance(spec, dict):
                continue
            key = spec.get("input") or spec.get("key")
            if "node_id" in spec:
                nids = spec["node_id"] if isinstance(spec["node_id"], list) else [spec["node_id"]]
                for nid in map(str, nids):
                    if nid not in wf:
                        problems.append(f"{section}.{logical}: node {nid} missing")
                    elif key and key not in (wf[nid].get("inputs") or {}):
                        problems.append(f"{section}.{logical}: {nid}.{key} missing")
            elif "class_type" in spec and not by_class(wf, spec["class_type"]):
                problems.append(f"{section}.{logical}: no {spec['class_type']}")
    final = wp.variant_final_node_id(variant)
    if final and final not in wf:
        problems.append(f"outputs.final node {final} missing")
    assert problems == []


# --------------------------------------------------------------------------- seed / size / frames


@pytest.mark.parametrize(
    "kw, frames",
    [
        ({"frames": 8}, 9),
        ({"frames": 9}, 9),
        ({"frames": 50}, 49),
        ({"duration_s": 2.0}, 49),
        ({"duration_s": 1.0}, 25),
    ],
)
def test_base_frames_obey_ltx_law_and_pair_audio(kw, frames, hermetic_weights):
    wf, meta = wp.load_and_patch_workflow("base", prompt="p", seed=1, **kw)
    assert meta["frames"] == frames
    assert (frames - 1) % 8 == 0 and frames >= 9
    assert wf["20"]["inputs"]["length"] == frames
    assert wf["21"]["inputs"]["frames_number"] == frames


def test_base_seed_size_steps_land_on_their_nodes(hermetic_weights):
    wf, meta = wp.load_and_patch_workflow(
        "base", prompt="fox", seed=123456, width=650, height=390, steps=6, frames=9
    )
    assert wf["42"]["inputs"] == {"noise_seed": 123456}
    assert wf["40"]["inputs"]["steps"] == 6
    w, h = wf["20"]["inputs"]["width"], wf["20"]["inputs"]["height"]
    assert (w, h) == (meta["width"], meta["height"])
    assert w % 32 == 0 and h % 32 == 0
    assert wf["10"]["inputs"]["text"] == "fox"


def test_random_seed_is_recorded_in_meta_and_graph(hermetic_weights):
    wf, meta = wp.load_and_patch_workflow("base", prompt="p", frames=9)
    assert isinstance(meta["seed"], int)
    assert wf["42"]["inputs"]["noise_seed"] == meta["seed"]


def test_lipsync_prompt_hits_positive_encoder_not_negative(hermetic_weights):
    raw = wp.load_workflow_template("lipsync")
    field = wp.variant_manifest("lipsync")["fields"]["prompt"]
    neg_field = wp.variant_manifest("lipsync")["fields"].get("negative_prompt")
    wf, _ = wp.load_and_patch_workflow(
        "lipsync", prompt="she says hello", negative_prompt="blurry mess", seed=1, frames=9
    )
    pos_ids = field["node_id"] if isinstance(field["node_id"], list) else [field["node_id"]]
    for nid in map(str, pos_ids):
        assert wf[nid]["inputs"][field["input"]] == "she says hello"
    if neg_field:
        assert wf[str(neg_field["node_id"])]["inputs"][neg_field["input"]] == "blurry mess"
    assert set(raw) <= set(wf) | set(wp._bypass_missing_optional_loras(copy.deepcopy(raw)))


# --------------------------------------------------------------------------- round trip / integrity


GOLDEN_VARIANTS = {
    "base": dict(prompt="a red fox in snow", seed=11, width=640, height=384, frames=17, steps=8),
    "lipsync": dict(prompt="she says hello", seed=22, frames=25, video_name="talk.mp4"),
    "wan22": dict(prompt="rainy street at dusk", seed=33, duration_s=2.0, steps=6),
}


def _patched(variant: str, **extra) -> tuple[dict, dict]:
    return wp.load_and_patch_workflow(variant, **{**GOLDEN_VARIANTS[variant], **extra})


@pytest.mark.parametrize("variant", sorted(GOLDEN_VARIANTS))
def test_patch_round_trips_as_api_json_and_keeps_links_valid(variant, hermetic_weights, object_info):
    wf, _ = _patched(variant)
    blob = json.dumps({"prompt": wf})
    assert json.loads(blob)["prompt"] == wf
    for nid, node in wf.items():
        assert isinstance(nid, str)
        assert set(node) <= {"class_type", "inputs", "_meta"}
        assert isinstance(node["class_type"], str) and isinstance(node["inputs"], dict)
    assert bad_links(wf, object_info) == []


@pytest.mark.parametrize("variant", sorted(GOLDEN_VARIANTS))
def test_patch_is_deterministic_and_does_not_mutate_template(variant, hermetic_weights):
    template_before = copy.deepcopy(wp.load_workflow_template(variant))
    a, meta_a = _patched(variant)
    b, meta_b = _patched(variant)
    assert a == b
    assert meta_a == meta_b
    assert wp.load_workflow_template(variant) == template_before


# Template links the patcher replaces today. Any other rewired link fails.
KNOWN_LINK_OVERWRITES = {
    "base": set(),
    "lipsync": {
        ("1241", "frame_rate"),
        ("4849", "fps"),
        ("3980", "frames_number"),
        ("3980", "frame_rate"),
    },
    "wan22": {("168", "noise_seed"), ("169", "noise_seed")},
}


@pytest.mark.parametrize("variant", sorted(GOLDEN_VARIANTS))
def test_patch_preserves_template_topology(variant, hermetic_weights):
    """Every template node and link survives, except stub bypasses and known overwrites."""
    raw = wp.load_workflow_template(variant)
    wf, _ = _patched(variant)
    missing = set(raw) - set(wf)
    assert all(raw[n]["class_type"] in ("LoraLoaderModelOnly", "LoraLoader") for n in missing)
    rewired = {
        (nid, key)
        for nid, key, src, slot in links(raw)
        if nid not in missing and src not in missing and wf[nid]["inputs"].get(key) != [src, slot]
    }
    assert rewired == KNOWN_LINK_OVERWRITES[variant]


def _mask(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    if isinstance(value, str) and value.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt")):
        return "<weight>"
    return value


@pytest.mark.parametrize("variant", sorted(GOLDEN_VARIANTS))
def test_golden_patched_graph(variant, hermetic_weights):
    wf, meta = _patched(variant)
    payload = {
        "variant": variant,
        "args": GOLDEN_VARIANTS[variant],
        "meta": {k: meta[k] for k in ("width", "height", "frames", "seed", "steps", "cfg")},
        "graph": _mask(wf),
    }
    golden = GOLDEN_DIR / f"{variant}.json"
    if os.environ.get("UPDATE_GOLDEN") == "1" or not golden.is_file():
        if os.environ.get("UPDATE_GOLDEN") != "1":
            pytest.fail(f"missing golden {golden}; run with UPDATE_GOLDEN=1")
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert payload == expected
