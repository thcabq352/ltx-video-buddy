"""Phase D: local draft promote. No git, no live Comfy, no downloads.

Run: python -m pytest tests/test_ingest_phase_d.py -q
"""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.promote import format_promote, promote_slug
from master_agent.comfy.ingest.store import ingest_file, ingest_graph
from master_agent.comfy.workflow_patcher import _apply_named_fields, variant_manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ingest" / "ltx_min_api.json"

_SEED = """\
base:
  file: base_t2v_i2v.json
  description: Simplified LTX-2.3 single-pass T2V (AV joint)
  vram_class: safe
  fields:
    prompt:
      node_id: "10"
      input: text
"""


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "state"
    dest.mkdir()
    monkeypatch.setattr("master_agent.config.STATE_DIR", dest)
    return dest


@pytest.fixture
def workflows(tmp_path: Path) -> Path:
    root = tmp_path / "workflows"
    root.mkdir()
    (root / "manifests.yaml").write_text(_SEED, encoding="utf-8")
    return root


def _ambiguous_graph() -> dict:
    return {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "one"},
            "_meta": {"title": "Encode A"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "two"},
            "_meta": {"title": "Encode B"},
        },
        "3": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": 97},
        },
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
        "5": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "ltx-2.3.safetensors"},
        },
        "6": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "clip"}},
    }


def _bind_workflows(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr("master_agent.config.WORKFLOWS_DIR", root)
    monkeypatch.setattr("master_agent.comfy.workflow_patcher.WORKFLOWS_DIR", root)


def _forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("promote must not spawn a process")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "call", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)


def test_promote_appends_parseable_draft_and_keeps_base(state_dir, workflows, monkeypatch):
    _forbid_subprocess(monkeypatch)
    ingest_file(FIXTURE, slug="phase-d-demo")
    result = promote_slug("phase-d-demo", variant_name="phase-d-draft", workflows_dir=workflows)
    text = (workflows / "manifests.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert loaded["base"]["file"] == "base_t2v_i2v.json"
    assert loaded["base"]["fields"]["prompt"]["node_id"] == "10"
    entry = loaded["phase-d-draft"]
    assert entry["file"] == "phase-d-draft.json"
    assert entry["vram_class"] == "unknown"
    assert entry["draft"] is True
    assert entry["fields"]["prompt"]["node_id"] == "12"
    assert entry["fields"]["prompt"]["input"] == "value"
    assert entry["outputs"]["final"]["node_id"] == "90"
    assert "phase-d-draft" != "base"
    graph = json.loads((workflows / "phase-d-draft.json").read_text(encoding="utf-8"))
    assert graph["12"]["inputs"]["value"] == "a cat on a roof"
    _bind_workflows(monkeypatch, workflows)
    meta = variant_manifest("phase-d-draft")
    assert meta["fields"]["seed"]["input"] == "noise_seed"
    written = _apply_named_fields(graph, meta["fields"], {"prompt": "neon rain"})
    assert ("12", "value") in written
    assert graph["12"]["inputs"]["value"] == "neon rain"
    printed = format_promote(result)
    assert "catalog default unchanged (base)" in printed
    assert "not committed, not pushed, no pull request" in printed
    assert "phase-d-draft" in result["diff"]
    assert result["catalog_default"] == "base"


def test_low_confidence_field_is_a_yaml_comment(state_dir, workflows):
    ingest_graph(_ambiguous_graph(), slug="ambiguous", source="memory")
    promote_slug("ambiguous", workflows_dir=workflows)
    text = (workflows / "manifests.yaml").read_text(encoding="utf-8")
    assert "# low confidence:" in text
    loaded = yaml.safe_load(text)
    prompt = loaded["ambiguous"]["fields"]["prompt"]
    assert set(prompt) <= {"node_id", "class_type", "index", "input"}
    assert prompt["node_id"]
    assert prompt["input"] == "text"
    assert "confidence" not in prompt


def test_missing_readiness_refuses_unless_force(state_dir, workflows):
    graph = _ambiguous_graph()
    graph["99"] = {"class_type": "TotallyUnknownNodeXYZ", "inputs": {}}
    ingest_graph(graph, slug="broken", source="memory")
    with pytest.raises(IngestError, match="refuse to promote"):
        promote_slug("broken", workflows_dir=workflows)
    assert not (workflows / "broken.json").exists()
    loaded = yaml.safe_load((workflows / "manifests.yaml").read_text(encoding="utf-8"))
    assert "broken" not in loaded
    result = promote_slug("broken", workflows_dir=workflows, force=True)
    assert result["forced"] is True
    assert (workflows / "broken.json").is_file()
    assert "TotallyUnknownNodeXYZ" in (workflows / "broken.json").read_text(encoding="utf-8")


def test_catalog_default_and_existing_key_are_refused(state_dir, workflows):
    ingest_file(FIXTURE, slug="phase-d-demo")
    with pytest.raises(IngestError, match="catalog default"):
        promote_slug("phase-d-demo", variant_name="base", workflows_dir=workflows)
    promote_slug("phase-d-demo", variant_name="phase-d-draft", workflows_dir=workflows)
    with pytest.raises(IngestError, match="already has"):
        promote_slug("phase-d-demo", variant_name="phase-d-draft", workflows_dir=workflows)
    loaded = yaml.safe_load((workflows / "manifests.yaml").read_text(encoding="utf-8"))
    assert list(loaded).count("phase-d-draft") == 1
    assert loaded["base"]["description"].startswith("Simplified")


def test_cli_promote_uses_checkout_and_prints_diff(state_dir, workflows, monkeypatch, capsys):
    _forbid_subprocess(monkeypatch)
    _bind_workflows(monkeypatch, workflows)
    ingest_file(FIXTURE, slug="phase-d-demo")
    from master_agent.cli.comfy import cmd_comfy

    rc = cmd_comfy(
        Namespace(
            comfy_command="promote",
            target="phase-d-demo",
            slug=None,
            variant_name="phase-d-cli",
            force=False,
        )
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "phase-d-cli" in printed
    assert "not committed, not pushed, no pull request" in printed
    loaded = yaml.safe_load((workflows / "manifests.yaml").read_text(encoding="utf-8"))
    assert loaded["phase-d-cli"]["draft"] is True
    assert loaded["base"]["vram_class"] == "safe"
