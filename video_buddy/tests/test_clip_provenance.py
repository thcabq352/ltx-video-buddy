"""ClipProvenance sidecar + run-row contract. No GPU, no live Comfy.

Run: python -m pytest tests/test_clip_provenance.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.state import LOOP_PASSED, RunState
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    REQUIRED_FIELDS,
    build_clip_provenance,
    inherit_clip_provenance,
    latest_revise_notes,
    persist_clip_provenance,
    read_clip_provenance,
    sidecar_path,
    write_clip_provenance,
)


def _assert_required(payload: dict) -> None:
    for name in REQUIRED_FIELDS:
        assert name in payload, f"missing ClipProvenance field {name}"
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA


def test_sidecar_path_is_clip_stem_provenance_json(tmp_path):
    clip = tmp_path / "shot_01.mp4"
    assert sidecar_path(clip) == tmp_path / "shot_01.provenance.json"


def test_build_includes_required_fields_and_keys_path_hash():
    st = RunState(
        request="neon rain alley",
        prompt="neon rain alley, handheld",
        run_id="abc123",
        variant="ltx25_t2v_i2v",
        seed=42,
        steps=8,
        cfg=3.5,
        width=768,
        height=512,
        duration_s=3.0,
        attempt=2,
        judge_score=0.81,
        judge_issues=["quality_bar.c: thin still"],
        judge_reason="thin_still_i2v",
        workflow_meta={"checkpoint": "ltx-2.5.safetensors"},
    )
    payload = build_clip_provenance(st, revise_notes="quality_bar.c: plan the still→I2V shot")
    _assert_required(payload)
    assert payload["prompt"] == "neon rain alley, handheld"
    assert payload["model"] == "ltx-2.5.safetensors"
    assert payload["workflow_id"] == "ltx25_t2v_i2v"
    assert payload["seed"] == 42
    assert payload["params"]["steps"] == 8
    assert payload["params"]["cfg"] == 3.5
    assert payload["attempt"] == 2
    assert payload["iteration"] == 2
    assert payload["judge_score"] == 0.81
    assert "thin_still_i2v" in payload["judge_reasons"]
    assert "quality_bar.c" in payload["revise_notes"]
    assert payload["run_id"] == "abc123"
    assert "path" in payload
    assert "hash" in payload


def test_write_read_sidecar_roundtrip(tmp_path):
    clip = tmp_path / "clip_a.mp4"
    clip.write_bytes(b"fake-mp4-bytes")
    st = RunState(
        request="rain",
        prompt="rain on glass",
        variant="base",
        seed=7,
        attempt=1,
        judge_score=0.4,
        judge_issues=["missing_music_bed"],
        video_path=str(clip),
        run_id="run1",
    )
    payload = persist_clip_provenance(st, revise_notes="first generate")
    dest = sidecar_path(clip)
    assert dest.is_file()
    assert dest.name == "clip_a.provenance.json"
    loaded = read_clip_provenance(clip)
    assert loaded is not None
    _assert_required(loaded)
    assert loaded["prompt"] == "rain on glass"
    assert loaded["hash"] == payload["hash"]
    assert len(payload["hash"]) == 64
    assert loaded["path"] == str(clip)
    assert st.provenance is payload
    assert st.provenance_sidecar == str(dest)
    assert st.provenance_history[-1]["attempt"] == 1


def test_inherit_fills_required_fields_and_rewrites_path_hash(tmp_path):
    src = tmp_path / "seg_000.mp4"
    dest = tmp_path / "master.mp4"
    src.write_bytes(b"seg")
    dest.write_bytes(b"stitched")
    write_clip_provenance(
        src,
        {
            "schema": CLIP_PROVENANCE_SCHEMA,
            "prompt": "seg prompt",
            "model": "base",
            "workflow_id": "base",
            "seed": 1,
            "params": {"steps": 4},
            "attempt": 1,
            "iteration": 1,
            "judge_score": 0.9,
            "judge_reasons": [],
            "revise_notes": "",
            "path": str(src),
            "hash": "abc",
        },
    )
    payload = inherit_clip_provenance(
        src, dest, revise_notes="pipeline stitch", extra={"prompt": "full brief"}
    )
    _assert_required(payload)
    assert payload["prompt"] == "full brief"
    assert payload["revise_notes"] == "pipeline stitch"
    assert payload["path"] == str(dest)
    assert payload["hash"] != "abc"
    assert payload["hash"]
    loaded = read_clip_provenance(dest)
    assert loaded["hash"] == payload["hash"]


def test_inherit_from_missing_sidecar_still_emits_required_fields(tmp_path):
    src = tmp_path / "missing.mp4"
    dest = tmp_path / "muxed.mp4"
    dest.write_bytes(b"mux")
    payload = inherit_clip_provenance(
        src, dest, revise_notes="music mux: source track as music bed"
    )
    _assert_required(payload)
    assert payload["revise_notes"].startswith("music mux")


def test_latest_revise_notes_summarizes_what_changed():
    st = RunState(request="mv")
    st.revise_history.append(
        {
            "attempt": 1,
            "reason": "quality_bar.a: attach a music bed",
            "prompt_deltas": ["music bed under the vocal"],
            "param_deltas": {"steps": 12},
            "music_bed_attached": True,
            "control_pack_used": {},
            "shot_patch": {},
        }
    )
    notes = latest_revise_notes(st)
    assert "quality_bar.a" in notes
    assert "music bed under the vocal" in notes
    assert "steps=12" in notes
    assert "music_bed_attached" in notes


def test_dry_run_embeds_provenance_in_run_json(tmp_path):
    runs = tmp_path / "runs"
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", runs
    ):
        st = Orchestrator().run(
            "music video for a synthwave track",
            kind="music_video",
            dry_run=True,
            judge_enabled=False,
            variant="base",
            seed=99,
            max_judge_rounds=3,
        )
    assert st.loop_status == LOOP_PASSED
    assert st.provenance
    _assert_required(st.provenance)
    assert st.provenance["prompt"]
    assert st.provenance["workflow_id"] == "base"
    assert st.provenance["seed"] == 99
    assert st.provenance["attempt"] == st.attempt
    assert st.provenance["iteration"] == st.attempt
    assert st.provenance["revise_notes"]
    assert st.provenance_history
    assert len(st.provenance_history) >= 2  # fail then pass
    assert st.provenance_history[0]["attempt"] == 1
    assert st.provenance_history[-1]["attempt"] == st.attempt
    records = list(runs.glob("*.json"))
    assert records, "run JSON not written"
    rec = json.loads(records[0].read_text(encoding="utf-8"))
    _assert_required(rec["provenance"])
    assert rec["provenance"]["hash"] == st.provenance["hash"]
    assert rec["run_id"] == st.run_id
    assert rec["provenance"]["run_id"] == st.run_id


def test_generate_revise_rerun_writes_sidecar_and_history(tmp_path):
    clip = tmp_path / "outputs" / "run_clip.mp4"
    runs = tmp_path / "runs"
    orch = Orchestrator()
    n = {"i": 0}

    def ok_patch(st):
        return True

    def ok_validate(st):
        return True

    def ok_submit(st):
        return True

    def ok_resolve(st):
        n["i"] += 1
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"clip-{n['i']}".encode())
        st.video_path = str(clip)
        persist_clip_provenance(st, revise_notes=latest_revise_notes(st), path=clip)
        return True

    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.analyze", return_value=(0.9, [])
    ), patch.object(orch, "_patch", ok_patch), patch.object(
        orch, "_validate", ok_validate
    ), patch.object(orch, "_submit_and_poll", ok_submit), patch.object(
        orch, "_resolve", ok_resolve
    ), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", runs
    ):
        st = orch.run(
            "i2v from this still, neon alley",
            image_name="still.png",
            variant="base",
            judge_enabled=False,
            dry_run=False,
            seed=3,
            max_judge_rounds=3,
        )
    assert st.loop_status == LOOP_PASSED
    assert n["i"] >= 2
    dest = sidecar_path(clip)
    assert dest.is_file()
    loaded = read_clip_provenance(clip)
    assert loaded is not None
    _assert_required(loaded)
    assert loaded["path"] == str(clip)
    assert loaded["hash"]
    assert loaded["attempt"] == st.attempt
    assert loaded["workflow_id"] == "base"
    assert loaded["seed"] == 3
    assert any(h.get("revise_notes") for h in st.provenance_history)
    rec = json.loads(next(runs.glob("*.json")).read_text(encoding="utf-8"))
    _assert_required(rec["provenance"])
    assert rec["provenance_sidecar"] == str(dest)
    assert rec["provenance"]["hash"] == loaded["hash"]
