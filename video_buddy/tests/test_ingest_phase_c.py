"""Phase C: fingerprint specialized families onto existing helpers.

No live Comfy, no GPU, no downloads.

Run: python -m pytest tests/test_ingest_phase_c.py tests/test_ingest_learn.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.comfy.ingest.run import dry_run_slug, run_ingested
from master_agent.comfy.ingest.store import ingest_graph

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ingest" / "ltx_min_api.json"


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "state"
    dest.mkdir()
    monkeypatch.setattr("master_agent.config.STATE_DIR", dest)
    return dest


def _inoutpaint_ltx25() -> dict:
    """Small 2.5 inpaint graph. PR #44 sets trim_to_shortest false and mask repeat."""
    return {
        "1": {"class_type": "LTXVInpaintPreprocess", "inputs": {}},
        "39": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": 25},
        },
        "58": {
            "class_type": "LTXVLaplacianPyramidBlend",
            "inputs": {"trim_to_shortest": True, "mask_low_res_dilation": 1},
            "_meta": {"title": "Blend Stage 1"},
        },
        "80": {
            "class_type": "VHS_DuplicateMasks",
            "inputs": {"multiply_by": 1, "mask": ["58", 0]},
        },
        "40": {"class_type": "LTXVImgToVideoInplace", "inputs": {"strength": 0.7}},
        "90": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "ltx25_inoutpaint"}},
    }


def test_inoutpaint_finalize_applies_pr44_mask_fixes(state_dir, monkeypatch):
    calls: list[dict] = []
    import master_agent.comfy.inoutpaint as inout

    real = inout.finalize_inoutpaint_graph

    def wrap(workflow, **kwargs):
        calls.append(kwargs)
        return real(workflow, **kwargs)

    monkeypatch.setattr(inout, "finalize_inoutpaint_graph", wrap)
    learned = ingest_graph(_inoutpaint_ltx25(), slug="inpaint", source="memory")["learned"]
    assert learned["family"] == "inoutpaint"
    assert learned["family_route"] == "specialized"
    report = dry_run_slug("inpaint")
    assert calls and calls[0]["ltx25"] is True
    assert report["family"] == "inoutpaint"
    assert report["family_route"] == "specialized"
    assert "finalize_inoutpaint_graph" in report["text"]
    blend = report["workflow"]["58"]["inputs"]
    assert blend["trim_to_shortest"] is False
    assert report["workflow"]["80"]["inputs"]["multiply_by"] == 25


def test_no_family_route_skips_finalize(state_dir, monkeypatch):
    import master_agent.comfy.inoutpaint as inout

    def boom(*_args, **_kwargs):
        raise AssertionError("finalize ran under --no-family-route")

    monkeypatch.setattr(inout, "finalize_inoutpaint_graph", boom)
    ingest_graph(
        _inoutpaint_ltx25(),
        slug="forced",
        source="memory",
        no_family_route=True,
    )
    report = dry_run_slug("forced")
    assert report["family"] == "inoutpaint"
    assert report["family_route"] == "generic"
    assert report["workflow"]["58"]["inputs"]["trim_to_shortest"] is True
    assert report["workflow"]["80"]["inputs"]["multiply_by"] == 1
    assert "--no-family-route" in report["text"]

    ingest_graph(_inoutpaint_ltx25(), slug="later", source="memory")
    flagged = dry_run_slug("later", no_family_route=True)
    assert flagged["family_route"] == "generic"
    assert flagged["workflow"]["58"]["inputs"]["trim_to_shortest"] is True


def test_sulphur_uses_patch_sulphur_graph(state_dir, monkeypatch):
    import master_agent.comfy.sulphur as sulphur

    calls: list[str] = []
    real = sulphur.patch_sulphur_graph

    def wrap(workflow, **kwargs):
        calls.append(kwargs.get("prompt") or "")
        return real(workflow, **kwargs)

    monkeypatch.setattr(sulphur, "patch_sulphur_graph", wrap)
    graph = {
        "1": {"class_type": "PathchSageAttentionKJ", "inputs": {}},
        "2": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "baked"},
            "_meta": {"title": "Prompt"},
        },
        "3": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sulphur"}},
    }
    learned = ingest_graph(graph, slug="sulphur", source="memory")["learned"]
    assert learned["family"] == "sulphur"
    report = dry_run_slug("sulphur", prompt="tower sulphur")
    assert calls == ["tower sulphur"]
    assert report["workflow"]["2"]["inputs"]["value"] == "tower sulphur"
    assert report["family_route"] == "specialized"
    assert "patch_sulphur_graph" in report["text"]


def test_lipsync_routes_prepare_queue_inputs_and_not_inoutpaint(state_dir, monkeypatch):
    import master_agent.comfy.inoutpaint as inout

    def boom(*_args, **_kwargs):
        raise AssertionError("inoutpaint finalize ran for lipsync")

    monkeypatch.setattr(inout, "finalize_inoutpaint_graph", boom)
    queued: list[str] = []
    real_prepare = inout.prepare_queue_inputs

    def wrap(workflow, upload):
        queued.append("prepare_queue_inputs")
        return real_prepare(workflow, upload)

    monkeypatch.setattr(inout, "prepare_queue_inputs", wrap)
    graph = {
        "1": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {"positive": ["2", 0]},
        },
        "2": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "baked line"},
            "_meta": {"title": "Positive prompt"},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0]}},
        "4": {"class_type": "LoadVideo", "inputs": {"file": "talk.mp4"}},
        "5": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "lips"}},
    }
    learned = ingest_graph(graph, slug="lips", source="memory")["learned"]
    assert learned["family"] == "lipsync"
    preview = dry_run_slug("lips", prompt="hello mouth")
    assert preview["family_route"] == "specialized"
    assert queued == []
    assert preview["workflow"]["3"]["inputs"]["text"] == ["2", 0]
    assert preview["workflow"]["2"]["inputs"]["value"] == "hello mouth"

    class _Queue:
        def queue_prompt(self, workflow):
            self.workflow = workflow
            return "p1"

        def free_memory(self):
            return None

    run_ingested("lips", client=_Queue(), prompt="hello mouth")
    assert queued == ["prepare_queue_inputs"]


def test_unmatched_graph_stays_generic_with_a_warning(state_dir, monkeypatch):
    import master_agent.comfy.inoutpaint as inout
    import master_agent.comfy.sulphur as sulphur

    def boom(*_args, **_kwargs):
        raise AssertionError("specialized helper ran for an unmatched graph")

    monkeypatch.setattr(inout, "finalize_inoutpaint_graph", boom)
    monkeypatch.setattr(sulphur, "patch_sulphur_graph", boom)
    graph = json.loads(FIXTURE.read_text(encoding="utf-8"))
    learned = ingest_graph(graph, slug="plain", source="memory")["learned"]
    assert learned["family"] is None
    assert learned["family_route"] == "generic"
    assert "unmatched graph stays on the generic path" in learned["family_warning"]
    report = dry_run_slug("plain", prompt="stay generic")
    assert report["family_route"] == "generic"
    assert "unmatched graph stays on the generic path" in report["text"]
    assert report["workflow"]["12"]["inputs"]["value"] == "stay generic"
