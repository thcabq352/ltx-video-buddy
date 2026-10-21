"""Phase A ingest / learn / dry-run / run. No live Comfy, no downloads.

Run: python -m pytest tests/test_ingest_learn.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy.ingest.classify import OFFLINE_NODE_CATALOG
from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.run import dry_run_slug, run_ingested
from master_agent.comfy.ingest.store import ingest_file, ingest_graph
from master_agent.comfy.ingest.validate import MissingCustomNodeError
from master_agent.comfy.vae_guard import TinyVAETiledDecodeError
from master_agent.config import LTX23_FULL_VIDEO_VAE

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ingest" / "ltx_min_api.json"
ROLES = ("prompt", "seed", "width", "height", "frames", "checkpoint", "filename_prefix")


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "state"
    dest.mkdir()
    monkeypatch.setattr("master_agent.config.STATE_DIR", dest)
    return dest


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


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


def _with_tiny_vae(graph: dict) -> dict:
    patched = json.loads(json.dumps(graph))
    patched["2"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "taeltx2_3.safetensors"},
        "_meta": {"title": "VAE"},
    }
    patched["50"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {"samples": ["40", 0], "vae": ["2", 0]},
        "_meta": {"title": "Tiled decode"},
    }
    return patched


class _Queue:
    def __init__(self):
        self.workflow = None
        self.calls: list[str] = []

    def queue_prompt(self, workflow):
        self.calls.append("queue_prompt")
        self.workflow = json.loads(json.dumps(workflow))
        return "prompt-1"

    def free_memory(self):
        self.calls.append("free_memory")


def test_ingest_writes_api_learned_and_provenance(state_dir: Path):
    result = ingest_file(FIXTURE, slug="demo")
    dest = state_dir / "ingested" / "demo"
    assert result["queued"] is False
    assert "dry-run demo" in result["next"]
    assert (dest / "workflow_api.json").is_file()
    assert (dest / "learned.yaml").is_file()
    assert (dest / "provenance.json").is_file()
    assert not (dest / "workflow_ui.json").exists()
    stored = json.loads((dest / "workflow_api.json").read_text(encoding="utf-8"))
    assert stored["12"]["inputs"]["value"] == "a cat on a roof"
    provenance = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["format"] == "api"
    assert provenance["comfy_version"] is None
    assert provenance["phase"] == "A"
    assert provenance["workflow_sha256"]


def test_role_learning_finds_prompt_seed_size_frames_checkpoint(state_dir: Path):
    learned = ingest_file(FIXTURE, slug="demo")["learned"]
    fields = learned["fields"]
    for role in ROLES:
        assert fields[role]["confidence"] == "high", role
    assert fields["prompt"]["node_id"] == "12"
    assert fields["prompt"]["input"] == "value"
    assert fields["prompt"]["node_id"] != "10"
    assert fields["seed"]["node_id"] == "42"
    assert fields["seed"]["input"] == "noise_seed"
    assert fields["width"]["node_id"] == "20"
    assert fields["width"]["input"] == "width"
    assert fields["height"]["input"] == "height"
    assert fields["frames"]["input"] == "length"
    assert fields["checkpoint"]["node_id"] == "1"
    assert fields["checkpoint"]["input"] == "ckpt_name"
    assert fields["filename_prefix"]["node_id"] == "90"
    assert fields["filename_prefix"]["input"] == "filename_prefix"
    assert learned["outputs"]["final"]["node_id"] == "90"
    assert learned["warnings"] == []


def test_patch_writes_primitive_and_leaves_linked_clip(state_dir: Path):
    ingest_file(FIXTURE, slug="demo")
    report = dry_run_slug("demo", prompt="neon rain", seed=42, frames=25)
    workflow = report["workflow"]
    assert workflow["12"]["inputs"]["value"] == "neon rain"
    assert workflow["10"]["inputs"]["text"] == ["12", 0]
    assert workflow["42"]["inputs"]["noise_seed"] == 42
    assert workflow["40"]["inputs"]["seed"] == 1
    assert workflow["20"]["inputs"]["length"] == 25
    assert report["queued"] is False


def test_dry_run_includes_learned_params_and_low_confidence_warnings(state_dir: Path, capsys):
    ingest_graph(_ambiguous_graph(), slug="ambiguous", source="memory")
    report = dry_run_slug("ambiguous", prompt="hello", seed=3)
    text = report["text"]
    assert "ingested dry-run (not queued)" in text
    assert "prompt:" in text
    assert "seed:" in text
    assert "width:" in text
    assert "height:" in text
    assert "frames:" in text
    assert "checkpoint:" in text
    assert "WARN  low confidence prompt:" in text
    assert "WARN  low confidence negative_prompt:" in text
    assert report["params"]["prompt"]["confidence"] == "low"
    assert report["params"]["seed"]["confidence"] == "high"
    assert report["queued"] is False

    from master_agent.cli.comfy import cmd_comfy

    rc = cmd_comfy(
        Namespace(
            comfy_command="dry-run",
            target="ambiguous",
            prompt="hello",
            negative_prompt=None,
            seed=3,
            width=None,
            height=None,
            frames=None,
            vae=None,
            set=[],
            slug=None,
            ingested=None,
            queue=False,
        )
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "WARN  low confidence prompt:" in printed
    assert "seed:" in printed
    assert "checkpoint:" in printed


def test_tiny_vae_on_tiled_decode_swaps_on_run_and_blocks_explicit(state_dir: Path):
    graph = _with_tiny_vae(_load_fixture())
    ingest_graph(graph, slug="tiny", source="memory")
    stored = json.loads(
        (state_dir / "ingested" / "tiny" / "workflow_api.json").read_text(encoding="utf-8")
    )
    assert stored["2"]["inputs"]["vae_name"] == "taeltx2_3.safetensors"

    preview = dry_run_slug("tiny")
    assert preview["queued"] is False
    assert preview["workflow"]["2"]["inputs"]["vae_name"] == LTX23_FULL_VIDEO_VAE
    assert preview["dangers"][0]["code"] == "tiny_vae_tiled"
    assert "taeltx2_3" in preview["dangers"][0]["detail"]
    again = json.loads(
        (state_dir / "ingested" / "tiny" / "workflow_api.json").read_text(encoding="utf-8")
    )
    assert again["2"]["inputs"]["vae_name"] == "taeltx2_3.safetensors"

    client = _Queue()
    result = run_ingested("tiny", client=client)
    assert result["queued"] is True
    assert client.calls[0] == "free_memory"
    assert "queue_prompt" in client.calls
    assert client.workflow["2"]["inputs"]["vae_name"] == LTX23_FULL_VIDEO_VAE

    blocked = _Queue()
    with pytest.raises(TinyVAETiledDecodeError, match="taeltx2_3"):
        run_ingested("tiny", client=blocked, vae="taeltx2_3.safetensors")
    assert blocked.workflow is None
    assert "queue_prompt" not in blocked.calls


def test_missing_custom_nodes_fail_with_the_names(state_dir: Path):
    graph = _load_fixture()
    graph["77"] = {"class_type": "WeirdCustomNode", "inputs": {"foo": 1}}
    graph["78"] = {"class_type": "AlsoMissingNode", "inputs": {"bar": 1}}
    ingest_graph(graph, slug="custom", source="memory")
    present = {name: {} for name in OFFLINE_NODE_CATALOG}
    client = _Queue()
    with pytest.raises(MissingCustomNodeError, match="does not auto-install") as exc:
        run_ingested("custom", client=client, object_info=present)
    message = str(exc.value)
    assert "AlsoMissingNode" in message
    assert "WeirdCustomNode" in message
    assert client.workflow is None

    with pytest.raises(MissingCustomNodeError, match="WeirdCustomNode"):
        dry_run_slug("custom", object_info=None)


def test_ui_json_is_refused_when_comfy_is_down(state_dir: Path, tmp_path: Path):
    ui = tmp_path / "ui.json"
    ui.write_text(
        json.dumps({"nodes": [{"id": 1, "type": "KSampler", "widgets_values": [1]}]}),
        encoding="utf-8",
    )
    with pytest.raises(IngestError, match="API JSON"):
        ingest_file(ui, slug="ui")
    assert not (state_dir / "ingested" / "ui").exists()


def test_ingest_cli_defaults_to_dry_run_next_and_does_not_queue(state_dir: Path, capsys):
    from master_agent.cli.parser import main

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("master_agent.cli.parser.ensure_dirs", lambda: None)
    try:
        rc = main(["comfy", "ingest", str(FIXTURE), "--slug", "from-cli"])
    finally:
        monkeypatch.undo()
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested slug=from-cli" in out
    assert "dry-run from-cli" in out
    assert "did not queue" in out
    assert (state_dir / "ingested" / "from-cli" / "learned.yaml").is_file()


def test_run_does_not_delete_neighboring_outputs(state_dir: Path, tmp_path: Path):
    ingest_file(FIXTURE, slug="demo")
    sentinel = tmp_path / "keep.mp4"
    sentinel.write_bytes(b"not-an-output")
    client = _Queue()
    run_ingested("demo", client=client, prompt="keep", seed=4)
    assert sentinel.is_file()
    assert sentinel.read_bytes() == b"not-an-output"
    assert client.workflow["12"]["inputs"]["value"] == "keep"


def test_illegal_frame_override_snaps(state_dir: Path):
    ingest_file(FIXTURE, slug="demo")
    report = dry_run_slug("demo", frames=8)
    assert report["workflow"]["20"]["inputs"]["length"] == 9


def test_shipped_base_graph_learns_roles_and_pairs_audio_frames(state_dir: Path):
    path = Path(__file__).resolve().parents[1] / "workflows" / "ltx23_av.json"
    learned = ingest_file(path, slug="base")["learned"]
    assert learned["readiness"]["missing_nodes"] == []
    fields = learned["fields"]
    assert fields["prompt"]["node_id"] == "10"
    assert fields["prompt"]["input"] == "text"
    assert fields["prompt"]["confidence"] == "high"
    assert fields["negative_prompt"]["node_id"] == "11"
    assert fields["seed"]["node_id"] == "42"
    assert fields["seed"]["input"] == "noise_seed"
    assert fields["frames"]["node_id"] == "20"
    assert fields["frames"]["input"] == "length"
    assert fields["checkpoint"]["node_id"] == "1"
    assert fields["checkpoint"]["input"] == "unet_name"
    assert fields["checkpoint"]["class_type"] == "UnetLoaderGGUF"
    assert fields["filename_prefix"]["node_id"] == "90"
    report = dry_run_slug("base", prompt="neon rain", seed=5, frames=25)
    assert report["queued"] is False
    assert report["workflow"]["10"]["inputs"]["text"] == "neon rain"
    assert report["workflow"]["1"]["inputs"]["unet_name"] == "10Eros_v1.5-Q4_K_M.gguf"
    assert report["workflow"]["70"]["class_type"] == "VAEDecode"
    assert report["workflow"]["5"]["inputs"]["vae_name"] == "taeltx2_3.safetensors"
    assert report["dangers"] == []
    assert report["workflow"]["20"]["inputs"]["length"] == 25
    assert report["workflow"]["21"]["inputs"]["frames_number"] == 25
    assert "ComfyUI-LTXVideo" in learned["requires"]
    assert "ComfyUI-GGUF" in learned["requires"]
