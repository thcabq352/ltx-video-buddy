"""Phase B: UI convert, history ingest, readiness packs and models.

No live Comfy, no GPU, no downloads. HTTP is mocked.

Run: python -m pytest tests/test_ingest_phase_b.py tests/test_ingest_learn.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.ingest.classify import OFFLINE_NODE_CATALOG
from master_agent.comfy.ingest.normalize import START_COMFY_OR_API, IngestError
from master_agent.comfy.ingest.run import dry_run_slug, run_ingested
from master_agent.comfy.ingest.store import ingest_file, ingest_graph, ingest_history
from master_agent.comfy.ingest.validate import (
    DOCTOR_POINTER,
    DOWNLOAD_POINTER,
    MissingModelError,
)
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, missing_required, read_clip_provenance

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ingest" / "ltx_min_api.json"


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "state"
    dest.mkdir()
    monkeypatch.setattr("master_agent.config.STATE_DIR", dest)
    return dest


def _api() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _ui_wrapping(api: dict) -> dict:
    return {
        "nodes": [{"id": 1, "type": "KSampler", "widgets_values": [1]}],
        "links": [],
        "extra": {"api_for_test": True},
        "_test_api": api,
    }


class _Http:
    def __init__(self, payload, *, status=200, fail=False):
        self.payload = payload
        self.status = status
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url))
        if self.fail:
            raise OSError("connection refused")
        body = self.payload
        posted = json

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return body

        self.posted = posted
        return _Resp()

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        if self.fail:
            raise OSError("connection refused")
        folder = url.rstrip("/").rsplit("/", 1)[-1]
        body = self.payload.get(folder, [])

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return body

        return _Resp()


class _History:
    def __init__(self, payload=None, *, fail=False):
        self.payload = payload or {}
        self.fail = fail
        self.models = ["ltx-2.3.safetensors"]

    def get_history(self, prompt_id):
        if self.fail:
            raise ComfyClientError("down")
        return self.payload

    def fetch_object_info(self):
        return {name: {"python_module": "nodes"} for name in OFFLINE_NODE_CATALOG}

    def list_model_filenames(self):
        return list(self.models)

    def health(self):
        return {"system": {"comfyui_version": "0.3.0-test"}}


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

    def wait_for_prompt(self, prompt_id):
        return {
            "outputs": {
                "90": {
                    "gifs": [
                        {
                            "filename": "clip.mp4",
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }
        }


def test_ui_without_comfy_refuses_with_the_start_or_api_error(state_dir, tmp_path):
    ui = tmp_path / "ui.json"
    ui.write_text(json.dumps(_ui_wrapping(_api())), encoding="utf-8")
    with pytest.raises(IngestError, match="start Comfy or supply API JSON") as exc:
        ingest_file(ui, slug="ui")
    assert START_COMFY_OR_API in str(exc.value)
    assert not (state_dir / "ingested" / "ui").exists()


def test_ui_convert_keeps_both_files(state_dir, tmp_path):
    api = _api()
    ui_path = tmp_path / "pack.json"
    ui_body = _ui_wrapping(api)
    ui_path.write_text(json.dumps(ui_body), encoding="utf-8")

    def convert(ui):
        assert ui["nodes"][0]["type"] == "KSampler"
        return ui["_test_api"]

    result = ingest_file(ui_path, slug="converted", converter=convert)
    dest = state_dir / "ingested" / "converted"
    assert (dest / "workflow_api.json").is_file()
    assert (dest / "workflow_ui.json").is_file()
    stored_ui = json.loads((dest / "workflow_ui.json").read_text(encoding="utf-8"))
    assert stored_ui["nodes"][0]["type"] == "KSampler"
    stored_api = json.loads((dest / "workflow_api.json").read_text(encoding="utf-8"))
    assert stored_api["12"]["inputs"]["value"] == "a cat on a roof"
    provenance = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["phase"] == "B"
    assert provenance["source_format"] == "ui"
    assert provenance["converted_via"] == "/workflow/convert"
    assert result["queued"] is False


def test_convert_workflow_posts_to_comfy_and_fails_closed(monkeypatch):
    client = ComfyClient("http://127.0.0.1:9")
    http = _Http({"4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}}})
    monkeypatch.setattr(client, "_http", lambda: http)
    api = client.convert_workflow({"nodes": [{"type": "SaveImage"}]})
    assert api["4"]["class_type"] == "SaveImage"
    assert http.calls[0] == ("POST", "http://127.0.0.1:9/workflow/convert")
    assert http.posted == {"nodes": [{"type": "SaveImage"}]}

    down = ComfyClient("http://127.0.0.1:9")
    monkeypatch.setattr(down, "_http", lambda: _Http({}, fail=True))
    with pytest.raises(ComfyClientError, match="start Comfy or supply API JSON"):
        down.convert_workflow({"nodes": []})


def test_history_ingest_reads_prompt_graph(state_dir):
    api = _api()
    prompt_id = "abc123prompt"
    client = _History(
        {
            prompt_id: {
                "prompt": [1, prompt_id, api, {}, ["90"]],
            }
        }
    )
    result = ingest_history(prompt_id, slug="from-hist", client=client)
    dest = state_dir / "ingested" / "from-hist"
    stored = json.loads((dest / "workflow_api.json").read_text(encoding="utf-8"))
    assert stored["12"]["class_type"] == "PrimitiveStringMultiline" or stored["12"]["inputs"]["value"]
    provenance = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["phase"] == "B"
    assert provenance["source_format"] == "history"
    assert provenance["prompt_id"] == prompt_id
    assert provenance["comfy_version"] == "0.3.0-test"
    assert result["learned"]["readiness"]["models_checked"] is True
    assert "ltx-2.3.safetensors" not in result["learned"]["readiness"]["missing_models"]


def test_history_down_refuses(state_dir):
    with pytest.raises(IngestError, match="start Comfy or supply API JSON"):
        ingest_history("missing", slug="nope", client=_History(fail=True))
    assert not (state_dir / "ingested" / "nope").exists()


def test_cli_history_uses_the_client(state_dir, monkeypatch, capsys):
    api = _api()
    prompt_id = "clihist01"
    fake = _History({prompt_id: {"prompt": [0, prompt_id, api, {}, []]}})

    class _Factory:
        def __init__(self, *args, **kwargs):
            pass

        def get_history(self, prompt_id_):
            return fake.get_history(prompt_id_)

        def fetch_object_info(self):
            return fake.fetch_object_info()

        def list_model_filenames(self):
            return fake.list_model_filenames()

        def health(self):
            return fake.health()

    monkeypatch.setattr("master_agent.comfy.client.ComfyClient", _Factory)
    from master_agent.cli.parser import main

    monkeypatch.setattr("master_agent.cli.parser.ensure_dirs", lambda: None)
    rc = main(["comfy", "ingest", "--from", f"history:{prompt_id}", "--slug", "cli-hist"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested slug=cli-hist" in out
    assert "did not queue" in out
    assert (state_dir / "ingested" / "cli-hist" / "workflow_api.json").is_file()


def test_missing_node_pack_from_known_map_and_python_module(state_dir):
    graph = _api()
    graph["77"] = {"class_type": "VHS_NotRealNode", "inputs": {}}
    graph["78"] = {"class_type": "OddPackNode", "inputs": {}}
    info = {name: {"python_module": "nodes"} for name in OFFLINE_NODE_CATALOG}
    info["OddPackNode"] = {"python_module": "custom_nodes.ComfyUI-OddPack.nodes.odd"}
    learned = ingest_graph(graph, slug="packs", source="memory", object_info=info)["learned"]
    assert learned["readiness"]["missing_nodes"] == ["VHS_NotRealNode"]
    packs = {row["class_type"]: row for row in learned["readiness"]["missing_node_packs"]}
    assert packs["VHS_NotRealNode"]["pack"] == "VideoHelperSuite"
    assert packs["VHS_NotRealNode"]["source"] == "known_map"
    assert "ComfyUI-OddPack" in learned["requires"]
    assert DOCTOR_POINTER in learned["readiness"]["pointers"]


def test_missing_model_is_reported_and_not_substituted(state_dir):
    graph = _api()
    graph["8"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "missing_lora.safetensors", "strength_model": 1.0},
    }
    inventory = {"ltx-2.3.safetensors", "LTX23_video_vae_bf16.safetensors"}
    learned = ingest_graph(
        graph, slug="models", source="memory", model_inventory=inventory
    )["learned"]
    assert learned["readiness"]["models_checked"] is True
    assert learned["readiness"]["missing_models"] == ["missing_lora.safetensors"]
    assert DOCTOR_POINTER in learned["readiness"]["pointers"]
    assert DOWNLOAD_POINTER in learned["readiness"]["pointers"]
    stored = json.loads(
        (state_dir / "ingested" / "models" / "workflow_api.json").read_text(encoding="utf-8")
    )
    assert stored["8"]["inputs"]["lora_name"] == "missing_lora.safetensors"
    assert stored["1"]["inputs"]["ckpt_name"] == "ltx-2.3.safetensors"

    report = dry_run_slug("models")
    assert report["workflow"]["8"]["inputs"]["lora_name"] == "missing_lora.safetensors"
    assert "LTX23_video_vae_bf16.safetensors" not in json.dumps(report["workflow"]["8"])
    assert "download-models" in report["text"]
    assert report["queued"] is False

    client = _Queue()
    with pytest.raises(MissingModelError, match="download-models"):
        run_ingested("models", client=client, object_info={name: {} for name in OFFLINE_NODE_CATALOG})
    assert client.workflow is None
    assert "queue_prompt" not in client.calls
    again = json.loads(
        (state_dir / "ingested" / "models" / "workflow_api.json").read_text(encoding="utf-8")
    )
    assert again["8"]["inputs"]["lora_name"] == "missing_lora.safetensors"


def test_model_list_http_is_mocked(monkeypatch):
    client = ComfyClient("http://127.0.0.1:9")
    http = _Http(
        {
            "checkpoints": ["ltx-2.3.safetensors"],
            "loras": [{"name": "folder/some_lora.safetensors"}],
            "vae": [],
        }
    )
    monkeypatch.setattr(client, "_http", lambda: http)
    names = client.list_model_filenames(("checkpoints", "loras", "vae"))
    assert "ltx-2.3.safetensors" in names
    assert "folder/some_lora.safetensors" in names
    assert all(call[0] == "GET" and "/models/" in call[1] for call in http.calls)


def test_clip_provenance_uses_the_existing_schema(state_dir, tmp_path):
    ingest_file(FIXTURE, slug="demo")
    out = tmp_path / "outputs"
    out.mkdir()
    clip = out / "clip.mp4"
    clip.write_bytes(b"not-deleted")
    client = _Queue()
    report = run_ingested("demo", client=client, output_dir=out, prompt="neon rain", seed=4)
    assert clip.read_bytes() == b"not-deleted"
    sidecars = report["provenance_sidecars"]
    assert len(sidecars) == 1
    payload = read_clip_provenance(clip)
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(payload) == []
    assert payload["engine"]["workflow_id"] == "demo"
    assert payload["engine"]["variant"] == "ingested:demo"
    assert payload["prompts"]["positive"] == "neon rain"
    assert payload["params"]["seed"] == 4
    assert payload["params"]["workflow_sha256"]
    assert clip.is_file()


def test_extra_model_paths_inventory(state_dir, tmp_path, monkeypatch):
    root = tmp_path / "weights"
    (root / "loras").mkdir(parents=True)
    (root / "loras" / "present_lora.safetensors").write_bytes(b"x")
    yaml_path = state_dir / "extra_model_paths.yaml"
    yaml_path.write_text(
        "buddy_models:\n"
        f"  base_path: {json.dumps(root.as_posix())}\n"
        "  loras: loras\n",
        encoding="utf-8",
    )
    graph = _api()
    graph["8"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "absent_lora.safetensors"},
    }
    learned = ingest_file(FIXTURE, slug="disk")["learned"]
    assert "ltx-2.3.safetensors" in learned["readiness"]["missing_models"]
    learned_absent = ingest_graph(
        graph,
        slug="disk-lora",
        source="memory",
        model_inventory=None,
    )
    # explicit None skips the file. The file path is covered by buddy_model_inventory
    # through ingest_file above (checkpoint missing, no silent rename).
    stored = json.loads(
        (state_dir / "ingested" / "disk" / "workflow_api.json").read_text(encoding="utf-8")
    )
    assert stored["1"]["inputs"]["ckpt_name"] == "ltx-2.3.safetensors"
    assert learned_absent["learned"]["readiness"]["models_checked"] is False


def test_dry_run_namespace_still_accepts_phase_a_flags(state_dir, capsys):
    ingest_file(FIXTURE, slug="demo")
    from master_agent.cli.comfy import cmd_comfy

    rc = cmd_comfy(
        Namespace(
            comfy_command="dry-run",
            target="demo",
            prompt="hello",
            negative_prompt=None,
            seed=1,
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
    assert "hello" in capsys.readouterr().out
