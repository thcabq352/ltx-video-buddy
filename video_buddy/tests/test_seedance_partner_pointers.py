"""Seedance 2.5 Draft pointers are visible and not queueable.

No ComfyUI, no GPU, no Partner HTTP.

Run: python -m pytest tests/test_seedance_partner_pointers.py -q
"""

from __future__ import annotations

from argparse import Namespace

from master_agent.comfy.catalog import default_variant_ids, is_known_variant, list_catalog_items
from master_agent.comfy.partner_pointers import (
    POINTERS,
    PartnerPointerError,
    lookup_pointer,
    partner_refusal,
    request_asks_seedance,
)
from master_agent.config import WORKFLOW_FILES
from master_agent.orchestrator.director import rule_based_variant


def test_stub_ids_match_official_templates():
    by_mode = {item.mode: item for item in POINTERS}
    assert by_mode["t2v"].template == "api_seedance2_5_draft_t2v"
    assert by_mode["t2v"].scout_node == "ByteDance2TextToVideoNode"
    assert by_mode["i2v"].template == "api_seedance2_5_draft_i2v"
    assert by_mode["i2v"].scout_node == "ByteDance2FirstLastFrameNode"
    assert by_mode["r2v"].template == "api_seedance2_5_draft_r2v"
    assert by_mode["r2v"].scout_node == "ByteDance2ReferenceNodeV2"
    assert lookup_pointer("templates/api_seedance2_5_draft_t2v.json").id == "seedance25_draft_t2v"


def test_pointers_are_not_default_catalog_or_director_allowlist():
    ids = {item.id for item in POINTERS}
    assert ids.isdisjoint(WORKFLOW_FILES)
    assert ids.isdisjoint(default_variant_ids())
    for item in POINTERS:
        assert is_known_variant(item.id) is False
        assert is_known_variant(item.template) is False
    paths = " ".join(row.get("path") or "" for row in list_catalog_items())
    assert "seedance" not in paths
    assert "ByteDance" not in paths


def test_seedance_brief_does_not_become_a_local_variant():
    assert rule_based_variant("seedance 2.5 draft one-take 30s") == "base"
    assert request_asks_seedance("quality draft garden proof") is False
    assert request_asks_seedance("draft") is False
    refusal = partner_refusal("Seedance 2.5 draft one-take", None)
    assert refusal is not None
    assert "api_seedance2_5_draft_t2v" in refusal
    assert "ByteDance2DraftToFinalVideoNode" in refusal
    assert "480p" in refusal
    assert partner_refusal("Seedance-like push", "ltx25_t2v_i2v") is None
    assert partner_refusal("rain on a window", None) is None
    i2v = partner_refusal("seedance draft image to video", "auto")
    assert i2v is not None and "api_seedance2_5_draft_i2v" in i2v


def test_workflows_list_prints_pointers_outside_variant_rows(capsys):
    from master_agent.__main__ import cmd_workflows

    assert cmd_workflows(Namespace(json=False, vram=False)) == 0
    out = capsys.readouterr().out
    assert "Partner pointer(s) (not queueable" in out
    assert "seedance25_draft_r2v" in out
    assert "api_seedance2_5_draft_r2v" in out
    variant_block, _, pointer_block = out.partition("Partner pointer")
    assert "seedance25_draft_t2v" not in variant_block
    assert "seedance25_draft_t2v" in pointer_block


def test_comfy_prepare_refuses_pointer_variant():
    from master_agent.comfy.cli_run import prepare_run, resolve_template

    try:
        prepare_run("generate", variant="seedance25_draft_t2v", prompt="one take")
    except PartnerPointerError as exc:
        assert "not a Video Buddy graph" in str(exc)
    else:
        raise AssertionError("generate mode queued a Partner pointer")

    try:
        resolve_template("api_seedance2_5_draft_r2v")
    except PartnerPointerError as exc:
        assert "api_seedance2_5_draft_r2v" in str(exc)
    else:
        raise AssertionError("template resolve treated the Partner id as a local file")


def test_run_refuses_seedance_before_queue(capsys):
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
    assert rc == 2
    assert "api_seedance2_5_draft_t2v" in out
    assert "variant:" not in out


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
    assert row.gap.startswith("Nodes exist")
    from fastapi.testclient import TestClient

    from master_agent.web.app import app

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
    assert refused.status_code == 400
    assert "api_seedance2_5_draft_i2v" in refused.json()["detail"]

    brief = client.post(
        "/api/jobs",
        json={"request": "promote this seedance draft_task_id", "quality": "draft"},
    )
    assert brief.status_code == 400
    assert "ByteDance2DraftToFinalVideoNode" in brief.json()["detail"]
