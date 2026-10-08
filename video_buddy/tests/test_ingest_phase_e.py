"""Phase E: local LLM names for low-confidence widgets, and the web drop zone.

No live Comfy, no GPU, no weight download, no cloud LLM.

Run: python -m pytest tests/test_ingest_phase_e.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from master_agent.comfy.client import ComfyClientError
from master_agent.comfy.ingest.run import dry_run_slug
from master_agent.comfy.ingest.store import ingest_graph
from master_agent.web.app import STATIC_DIR, app

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ingest" / "ltx_min_api.json"


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


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "state"
    dest.mkdir()
    monkeypatch.setattr("master_agent.config.STATE_DIR", dest)
    return dest


def _boom(*_args, **_kwargs):
    raise AssertionError("cloud or autostart LLM was called")


def test_llm_assist_off_leaves_heuristics(state_dir: Path):
    learned = ingest_graph(_ambiguous_graph(), slug="plain", source="memory")["learned"]
    assert "llm_proposals" not in learned
    assert "source" not in learned["fields"]["prompt"]
    assert learned["fields"]["seed"]["confidence"] == "high"


def test_llm_assist_names_only_low_confidence(state_dir: Path):
    def proposer(ctx):
        if ctx["current_role"] == "prompt":
            return {"role": "prompt", "confidence": "medium"}
        if ctx["current_role"] == "negative_prompt":
            return {"role": "seed", "confidence": "high"}
        return {"role": "prompt", "confidence": "high"}

    learned = ingest_graph(
        _ambiguous_graph(),
        slug="named",
        source="memory",
        llm_assist=True,
        proposer=proposer,
    )["learned"]
    prompt = learned["fields"]["prompt"]
    assert prompt["source"] == "llm"
    assert prompt["confidence"] == "medium"
    assert learned["fields"]["seed"]["confidence"] == "high"
    assert "source" not in learned["fields"]["seed"]
    seed_proposals = [row for row in learned["llm_proposals"] if row["from"] == "negative_prompt"]
    assert seed_proposals and seed_proposals[0]["applied"] is False
    report = dry_run_slug("named")
    assert "source llm" in report["text"]
    assert "llm: prompt -> prompt [medium] source llm" in report["text"]
    assert report["params"]["seed"].get("source") in (None, "")


def test_no_local_llm_keeps_heuristics_and_skips_cloud(state_dir, monkeypatch):
    monkeypatch.setattr("master_agent.llm.preferred_local_provider", lambda: None)
    monkeypatch.setattr("master_agent.llm.get_llm", _boom)
    monkeypatch.setattr("master_agent.llm._llm_for", _boom)
    monkeypatch.setattr("master_agent.llm.prepare_local_llm", _boom)
    learned = ingest_graph(
        _ambiguous_graph(),
        slug="offline-llm",
        source="memory",
        llm_assist=True,
    )["learned"]
    assert "source" not in learned["fields"]["prompt"]
    assert "No cloud call" in learned["llm_warning"]
    assert "llamacpp" in learned["llm_warning"]


def test_llamacpp_is_used_without_auto_or_autostart(state_dir, monkeypatch):
    seen: list[tuple] = []

    class _Chat:
        def invoke(self, message):
            seen.append(("invoke", message))
            return type("R", (), {"content": '{"role":"prompt","confidence":"low"}'})()

    def llm_for(spec, temperature):
        seen.append(("for", spec, temperature))
        assert spec == "llamacpp"
        return _Chat()

    monkeypatch.setattr("master_agent.llm.preferred_local_provider", lambda: "llamacpp")
    monkeypatch.setattr("master_agent.llm._llm_for", llm_for)
    monkeypatch.setattr("master_agent.llm.get_llm", _boom)
    monkeypatch.setattr("master_agent.llm.prepare_local_llm", _boom)
    learned = ingest_graph(
        _ambiguous_graph(),
        slug="local",
        source="memory",
        llm_assist=True,
    )["learned"]
    assert seen and seen[0][0] == "for" and seen[0][1] == "llamacpp"
    assert learned["fields"]["prompt"]["source"] == "llm"
    assert learned["llm_proposals"][0]["provider"] == "llamacpp"
    assert "grok" not in json.dumps(learned["llm_proposals"])


def test_cli_flag_is_off_unless_passed(state_dir, monkeypatch, capsys):
    calls = {"n": 0}

    def proposer(ctx):
        calls["n"] += 1
        return {"role": ctx["current_role"], "confidence": "low"}

    monkeypatch.setattr(
        "master_agent.comfy.ingest.llm_assist.build_local_proposer",
        lambda: proposer,
    )
    monkeypatch.setattr("master_agent.llm.get_llm", _boom)
    path = state_dir / "amb.json"
    path.write_text(json.dumps(_ambiguous_graph()), encoding="utf-8")
    from master_agent.cli.comfy import cmd_comfy

    rc = cmd_comfy(
        Namespace(
            comfy_command="ingest",
            target=str(path),
            slug="cli-off",
            prompt="",
            negative_prompt=None,
            seed=None,
            width=None,
            height=None,
            frames=None,
            vae=None,
            set=[],
            ingested=None,
            queue=False,
            ingest_from=None,
            no_family_route=False,
            llm_assist=False,
            workflow_json=None,
        )
    )
    assert rc == 0
    assert calls["n"] == 0
    rc = cmd_comfy(
        Namespace(
            comfy_command="learn",
            target="cli-off",
            slug=None,
            no_family_route=False,
            llm_assist=True,
        )
    )
    assert rc == 0
    assert calls["n"] > 0
    printed = capsys.readouterr().out
    assert "source: llm" in printed or "source llm" in printed


class _FakeComfy:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def convert_workflow(self, workflow):
        raise ComfyClientError("start Comfy or supply API JSON. mocked")

    def queue_prompt(self, workflow):
        _FakeComfy.calls += 1
        return "prompt-e"

    def free_memory(self):
        return None


@pytest.fixture
def web(state_dir, monkeypatch):
    monkeypatch.setattr("master_agent.llm.prepare_local_llm", lambda *a, **k: None)
    monkeypatch.setattr("master_agent.kb.ingest.schedule_knowledge_ingest", lambda: None)
    monkeypatch.setattr("master_agent.comfy.client.ComfyClient", _FakeComfy)
    _FakeComfy.calls = 0
    with TestClient(app) as client:
        yield client


def test_drop_zone_markup_sits_beside_the_template_picker():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    template_at = html.index('id="wf-template"')
    ingest_at = html.index('id="ingest-drop"')
    raw_at = html.index('id="wf-drop"')
    assert template_at < ingest_at < raw_at
    assert 'id="ingest-confirm"' in html
    assert 'id="ingest-dry"' in html
    assert 'id="ingest-queue" type="button" disabled' in html
    assert "comfy run --ingested" in html
    lowered = html.lower()
    assert "react.development" not in lowered
    assert "vue.global" not in lowered
    assert "unpkg.com/react" not in lowered
    assert "cdn.jsdelivr.net/npm/vue" not in lowered


def test_web_ingest_dry_run_and_confirmed_queue(web, monkeypatch):
    import master_agent.comfy.ingest.run as run_mod

    guarded = {"n": 0}
    real = run_mod.apply_vae_guard

    def wrap(workflow, **kwargs):
        guarded["n"] += 1
        return real(workflow, **kwargs)

    monkeypatch.setattr(run_mod, "apply_vae_guard", wrap)
    payload = FIXTURE.read_bytes()
    ingested = web.post(
        "/api/comfy/ingest",
        files={"file": ("ltx_min_api.json", payload, "application/json")},
    )
    assert ingested.status_code == 200, ingested.text
    body = ingested.json()
    assert body["queued"] is False
    assert body["fields"]["prompt"]["input"] == "value"
    assert "readiness" in body
    assert body["slug"] == "ltx_min_api"

    dry = web.post("/api/comfy/ingest/dry-run", json={"slug": body["slug"], "prompt": "neon rain"})
    assert dry.status_code == 200, dry.text
    assert dry.json()["queued"] is False
    assert "not queued" in dry.json()["text"]
    assert _FakeComfy.calls == 0
    assert guarded["n"] == 1

    refused = web.post("/api/comfy/ingest/run", json={"slug": body["slug"], "confirm": False})
    assert refused.status_code == 400
    assert "confirm" in refused.json()["detail"]
    assert _FakeComfy.calls == 0

    queued = web.post(
        "/api/comfy/ingest/run",
        json={"slug": body["slug"], "confirm": True, "prompt": "neon rain"},
    )
    assert queued.status_code == 200, queued.text
    assert queued.json()["queued"] is True
    assert queued.json()["prompt_id"] == "prompt-e"
    assert _FakeComfy.calls == 1
    assert guarded["n"] == 2


def test_web_ui_json_refuses_when_convert_is_down(web):
    ui = json.dumps({"nodes": [{"type": "CLIPTextEncode", "widgets_values": ["hi"]}]}).encode()
    response = web.post(
        "/api/comfy/ingest",
        files={"file": ("ui.json", ui, "application/json")},
    )
    assert response.status_code == 400
    assert "start Comfy or supply API JSON" in response.json()["detail"]
    assert _FakeComfy.calls == 0
