"""Pack C Seedance routing is hard local-only.

Partner template ids stay field-shape records. They are not loaded, enabled,
or queued. No ComfyUI process, no GPU, no Partner HTTP.

Run: python -m pytest tests/test_seedance_partner_pointers.py -q
"""

from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

from master_agent.comfy.catalog import default_variant_ids, is_known_variant, list_catalog_items
from master_agent.comfy.partner_pointers import (
    POINTERS,
    FORBIDDEN_CLASS_TYPES,
    PartnerPointerError,
    cloud_video_host,
    local_catalog_for_pack_c_brief,
    lookup_pointer,
    partner_class_types_in,
    partner_graphs_executable,
    partner_refusal,
    reject_partner_or_cloud_queue,
    request_asks_seedance,
    route_pack_c,
)
from master_agent.config import PACK_C_COMFY_URL, PACK_C_LOCAL_ONLY, WORKFLOW_FILES, XAI_BASE_URL
from master_agent.orchestrator.director import rule_based_variant

VIDEO_BUDDY = Path(__file__).resolve().parents[1]
_URL_RE = re.compile(r"https?://[^\s\"'`<>)\]]+")


def test_stub_ids_match_recorded_partner_shape():
    by_mode = {item.mode: item for item in POINTERS}
    assert by_mode["t2v"].template == "api_seedance2_5_draft_t2v"
    assert by_mode["t2v"].scout_node == "ByteDance2TextToVideoNode"
    assert by_mode["i2v"].template == "api_seedance2_5_draft_i2v"
    assert by_mode["i2v"].scout_node == "ByteDance2FirstLastFrameNode"
    assert by_mode["r2v"].template == "api_seedance2_5_draft_r2v"
    assert by_mode["r2v"].scout_node == "ByteDance2ReferenceNodeV2"
    assert lookup_pointer("templates/api_seedance2_5_draft_t2v.json").id == "seedance25_draft_t2v"
    for item in POINTERS:
        assert item.executable is False
        assert item.scout_node in FORBIDDEN_CLASS_TYPES


def test_local_only_policy_is_closed():
    assert partner_graphs_executable() is False
    assert PACK_C_LOCAL_ONLY["hardRequirement"] is True
    assert PACK_C_LOCAL_ONLY["cloudApis"] is False
    assert PACK_C_LOCAL_ONLY["cloudServices"] is False
    assert PACK_C_LOCAL_ONLY["hostedInference"] is False
    assert PACK_C_LOCAL_ONLY["partnerGraphsExecutable"] is False
    assert PACK_C_LOCAL_ONLY["execution"] == "local-comfy"
    assert PACK_C_LOCAL_ONLY["videoInference"] == "local-comfy"
    assert PACK_C_LOCAL_ONLY["directingLlmSeparateFromVideo"] is True
    assert PACK_C_COMFY_URL == "http://127.0.0.1:8188"
    assert PACK_C_LOCAL_ONLY["comfy"] == PACK_C_COMFY_URL
    assert PACK_C_LOCAL_ONLY["localPacks"]["t2v"] == "ltx25_t2v_i2v"
    assert PACK_C_LOCAL_ONLY["localPacks"]["flf"] == "ltx25_flf2v"
    assert PACK_C_LOCAL_ONLY["localPacks"]["r2v"] == "ltx25_msr"
    assert "wan" in PACK_C_LOCAL_ONLY["keptLocalFamilies"]
    assert "h3" in PACK_C_LOCAL_ONLY["keptLocalFamilies"]
    assert XAI_BASE_URL != PACK_C_COMFY_URL
    assert cloud_video_host(XAI_BASE_URL) is None

    agents = (VIDEO_BUDDY / "AGENTS.md").read_text(encoding="utf-8")
    doc = (VIDEO_BUDDY / "docs" / "SEEDANCE_2_5_DRAFT.md").read_text(encoding="utf-8")
    skill = (VIDEO_BUDDY / "skills" / "video-buddy" / "SKILL.md").read_text(encoding="utf-8")
    local_only = (
        VIDEO_BUDDY / "skills" / "video-buddy" / "references" / "local-only.md"
    ).read_text(encoding="utf-8")
    for text in (agents, doc, skill, local_only):
        assert "Local-only is a hard requirement" in text
        assert "http://127.0.0.1:8188" in text
    assert "Open Comfy template" not in doc
    assert "queue once" not in doc.lower()
    assert "choose to spend" not in skill
    assert "Enable Stage 1" not in skill


def test_pointers_are_not_default_catalog_or_director_allowlist():
    ids = {item.id for item in POINTERS}
    templates = {item.template for item in POINTERS}
    assert ids.isdisjoint(WORKFLOW_FILES)
    assert templates.isdisjoint(WORKFLOW_FILES)
    assert ids.isdisjoint(default_variant_ids())
    for item in POINTERS:
        assert is_known_variant(item.id) is False
        assert is_known_variant(item.template) is False
    paths = " ".join(row.get("path") or "" for row in list_catalog_items())
    assert "seedance" not in paths
    assert "ByteDance" not in paths


def test_seedance_brief_fail_closes_onto_local_catalog():
    assert request_asks_seedance("quality draft garden proof") is False
    assert request_asks_seedance("draft") is False
    assert rule_based_variant("rain on a window") == "base"
    assert rule_based_variant("seedance 2.5 draft one-take 30s") == "ltx25_t2v_i2v"
    assert local_catalog_for_pack_c_brief("seedance draft image to video") == "ltx25_t2v_i2v"
    assert local_catalog_for_pack_c_brief("seedance first-last") == "ltx25_flf2v"
    assert local_catalog_for_pack_c_brief("seedance reference to video") == "ltx25_msr"

    routed, err = route_pack_c("Seedance 2.5 draft one-take", None)
    assert err is None
    assert routed == "ltx25_t2v_i2v"
    assert partner_refusal("Seedance 2.5 draft one-take", None) is None

    i2v, i2v_err = route_pack_c("seedance draft image to video", "auto")
    assert i2v_err is None and i2v == "ltx25_t2v_i2v"

    kept, kept_err = route_pack_c("Seedance-like push", "ltx25_t2v_i2v")
    assert kept_err is None and kept == "ltx25_t2v_i2v"
    assert partner_refusal("Seedance-like push", "ltx25_t2v_i2v") is None
    wan, wan_err = route_pack_c("seedance one-take", "wan22")
    assert wan_err is None and wan == "wan22"
    h3, h3_err = route_pack_c("seedance one-take", "h3_t2v")
    assert h3_err is None and h3 == "h3_t2v"

    plain, plain_err = route_pack_c("rain on a window", None)
    assert plain_err is None and plain is None

    closed, closed_err = route_pack_c("rain on a window", "api_seedance2_5_draft_i2v")
    assert closed_err is None
    assert closed == "ltx25_t2v_i2v"
    refusal = partner_refusal("", "seedance25_draft_t2v")
    assert refusal is not None
    assert "not executable" in refusal
    assert "http://127.0.0.1:8188" in refusal
    assert "Open Comfy template" not in refusal
    assert "ltx25_t2v_i2v" in refusal


def test_remote_comfy_url_refuses_pack_c(monkeypatch):
    monkeypatch.setattr("master_agent.config.COMFYUI_URL", "https://api.comfy.org")
    routed, err = route_pack_c("Seedance 2.5 draft one-take", None)
    assert routed is None
    assert err is not None
    assert "Local-only is a hard requirement" in err
    assert "http://127.0.0.1:8188" in err
    assert "Open Comfy template" not in err
    # A non-Pack-C brief still names its own variant. This guard does not
    # rewrite LTX / Wan / H3 calls.
    kept, kept_err = route_pack_c("rain on a window", "wan22")
    assert kept_err is None and kept == "wan22"


def test_workflows_list_prints_field_shape_records(capsys):
    from master_agent.__main__ import cmd_workflows

    assert cmd_workflows(Namespace(json=False, vram=False)) == 0
    out = capsys.readouterr().out
    assert "field-shape record, not executable" in out
    assert "Open Comfy template" not in out
    assert "seedance25_draft_r2v" in out
    assert "api_seedance2_5_draft_r2v" in out
    variant_block, _, pointer_block = out.partition("Partner pointer")
    assert "seedance25_draft_t2v" not in variant_block
    assert "seedance25_draft_t2v" in pointer_block
    assert "not executable" in pointer_block


def test_partner_template_is_not_loaded_and_generate_uses_local_graph():
    from master_agent.comfy.cli_run import prepare_run, resolve_template

    prepared = prepare_run("generate", variant="seedance25_draft_t2v", prompt="one take")
    classes = partner_class_types_in(prepared)
    assert classes == set()
    blob = str(prepared)
    assert "ByteDance" not in blob
    assert "api_seedance" not in blob

    try:
        resolve_template("api_seedance2_5_draft_r2v")
    except PartnerPointerError as exc:
        text = str(exc)
        assert "api_seedance2_5_draft_r2v" in text
        assert "not executable" in text
        assert "Open Comfy template" not in text
    else:
        raise AssertionError("template resolve loaded a Partner id")

    try:
        prepare_run(
            "raw",
            workflow={"1": {"class_type": "ByteDance2TextToVideoNode", "inputs": {}}},
        )
    except PartnerPointerError as exc:
        assert "not executable" in str(exc)
    else:
        raise AssertionError("raw mode accepted a Partner graph")


def test_queue_prompt_refuses_partner_graphs_and_cloud_hosts():
    from master_agent.comfy.client import ComfyClient, ComfyClientError

    partner = {"1": {"class_type": "ByteDance2DraftToFinalVideoNode", "inputs": {}}}
    assert reject_partner_or_cloud_queue(partner, "http://127.0.0.1:8188")
    local = {"1": {"class_type": "KSampler", "inputs": {}}}
    assert reject_partner_or_cloud_queue(local, "http://127.0.0.1:8188") is None
    assert reject_partner_or_cloud_queue(local, "http://comfy.test") is None
    blocked = reject_partner_or_cloud_queue(local, "https://api.comfy.org/api")
    assert blocked is not None and "comfy.org" in blocked
    for host in (
        "https://ark.ap-southeast.byteplus.com",
        "https://ark.example.modelark.com",
        "https://api.kie.ai/v1",
        "https://ark.cn-beijing.volces.com",
    ):
        assert cloud_video_host(host)

    client = ComfyClient("http://127.0.0.1:8188")
    try:
        client.queue_prompt(partner)
    except ComfyClientError as exc:
        assert "not executable" in str(exc)
    else:
        raise AssertionError("queue_prompt submitted a Partner graph")

    cloud = ComfyClient("https://api.comfy.org")
    try:
        cloud.queue_prompt(local)
    except ComfyClientError as exc:
        assert "comfy.org" in str(exc)
    else:
        raise AssertionError("queue_prompt submitted to comfy.org")


def test_run_fail_closes_seedance_before_any_partner_queue(capsys):
    from master_agent.__main__ import cmd_run

    rc = cmd_run(
        Namespace(
            request="Seedance 2.5 draft one-take, 20-30s",
            variant=None,
            duration=5.0,
            duration_set=False,
            quality="draft",
            seed=1,
            width=768,
            height=512,
            video=None,
            image=None,
            audio=None,
            no_judge=True,
            max_judge_rounds=1,
            storyboard="off",
            llm_panel=None,
            panel_judge=None,
            max_full_judge_rounds=None,
            dry_run=True,
            self_improve_dry=False,
            power_mode=False,
            no_power_mode=False,
            upscale=None,
            no_interview=True,
            attach=None,
        )
    )
    out = capsys.readouterr().out
    assert "variant: ltx25_t2v_i2v" in out
    assert "Open Comfy template" not in out
    assert "api_seedance2_5_draft_" not in out
    assert rc == 0


def test_api_jobs_rewrite_partner_ids_to_local_variants(monkeypatch):
    from fastapi.testclient import TestClient

    from master_agent.web.app import app

    captured: dict[str, object] = {}

    class _Job:
        def to_dict(self):
            return {"id": "job", "status": "queued", "variant": captured.get("variant")}

    def fake_submit(kind, request, **params):
        captured["kind"] = kind
        captured["request"] = request
        captured["variant"] = params.get("variant")
        return _Job()

    monkeypatch.setattr("master_agent.web.app.MANAGER.submit", fake_submit)
    client = TestClient(app)

    listed = client.get("/api/variants")
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()["items"]}
    assert "seedance25_draft_t2v" not in ids
    assert "ltx25_t2v_i2v" in ids

    refused = client.post(
        "/api/jobs",
        json={"request": "rain on a window", "variant": "api_seedance2_5_draft_i2v"},
    )
    assert refused.status_code == 200
    assert captured["variant"] == "ltx25_t2v_i2v"
    assert "api_seedance2_5_draft_i2v" not in str(refused.json())

    brief = client.post(
        "/api/jobs",
        json={"request": "promote this seedance draft_task_id", "quality": "draft", "dry_run": True},
    )
    assert brief.status_code == 200
    assert captured["variant"] == "ltx25_t2v_i2v"
    assert captured["kind"] == "dry-run"

    monkeypatch.setattr("master_agent.config.COMFYUI_URL", "https://api.kie.ai")
    blocked = client.post(
        "/api/jobs",
        json={"request": "Seedance 2.5 draft one-take", "dry_run": True},
    )
    assert blocked.status_code == 400
    assert "Local-only is a hard requirement" in blocked.json()["detail"]
    assert "http://127.0.0.1:8188" in blocked.json()["detail"]


def test_capability_probe_does_not_treat_partner_nodes_as_wired():
    from master_agent.comfy.capabilities import probe_capabilities

    rows = {
        row.capability.id: row
        for row in probe_capabilities(
            {
                "ByteDance2TextToVideoNode": {},
                "ByteDance2FirstLastFrameNode": {},
                "ByteDance2ReferenceNodeV2": {},
                "ByteDance2DraftToFinalVideoNode": {},
            }
        )
    }
    row = rows["seedance25_draft"]
    assert row.verdict == "no"
    assert row.capability.surfaces == ()
    assert row.director is False
    assert "api_seedance2_5_draft_" in row.capability.notes
    assert "not executable" in row.capability.notes.lower() or "Not executable" in row.capability.notes
    assert row.gap.startswith("Nodes exist")


def test_no_executable_partner_workflow_or_cloud_video_url():
    for path in (VIDEO_BUDDY / "workflows").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "ByteDance" not in text, path
        assert "api_seedance" not in text, path
    for path in (VIDEO_BUDDY / "master_agent").rglob("*.py"):
        for url in _URL_RE.findall(path.read_text(encoding="utf-8")):
            assert cloud_video_host(url) is None, (path, url)
