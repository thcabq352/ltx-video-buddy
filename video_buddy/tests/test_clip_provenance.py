"""ClipProvenance sidecar + run-row contract. No GPU, no live Comfy.

Aligned to buddy.clip.provenance/v1 (Rust PR #7 shape).
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
    apply_sidecar_to_state,
    build_clip_provenance,
    inherit_clip_provenance,
    latest_revise_notes,
    missing_required,
    persist_clip_provenance,
    plan_clip_paths,
    read_clip_provenance,
    sidecar_path,
    write_clip_provenance,
)


def _assert_required(payload: dict) -> None:
    assert payload.get("schema") == CLIP_PROVENANCE_SCHEMA
    missing = missing_required(payload)
    assert missing == [], f"missing ClipProvenance fields: {missing}"


def test_sidecar_path_is_shot_n_buddy_json(tmp_path):
    clip = tmp_path / "shot-1.mp4"
    assert sidecar_path(clip) == tmp_path / "shot-1.buddy.json"


def test_build_nested_pr7_fields():
    st = RunState(
        request="neon rain alley",
        prompt="neon rain alley, handheld",
        negative_prompt="blur, text",
        run_id="abc123",
        variant="ltx25_t2v_i2v",
        seed=42,
        steps=8,
        cfg=3.5,
        width=768,
        height=512,
        duration_s=3.0,
        fps=24,
        attempt=2,
        shot_index=3,
        judge_score=0.81,
        judge_issues=["quality_bar.c: thin still"],
        judge_reason="thin_still_i2v",
        quality_bar={
            "fails": [
                {
                    "id": "c",
                    "code": "thin_still_i2v",
                    "detail": "still→I2V with no shot plan",
                }
            ]
        },
        image_name="still.png",
        parent_shot_id="shot-3",
        parent_attempt_id="shot-3.a1",
    )
    payload = build_clip_provenance(
        st, revise_notes="quality_bar.c: plan the still→I2V shot"
    )
    _assert_required(payload)
    assert payload["prompts"]["brief"] == "neon rain alley"
    assert payload["prompts"]["positive"] == "neon rain alley, handheld"
    assert payload["prompts"]["negative"] == "blur, text"
    assert payload["engine"]["backend"] == "comfy"
    assert payload["engine"]["workflow_id"] == "ltx25_t2v_i2v"
    assert payload["engine"]["variant"] == "ltx25_t2v_i2v"
    assert payload["params"]["seed"] == 42
    assert payload["params"]["steps"] == 8
    assert payload["params"]["cfg"] == 3.5
    assert payload["params"]["size"] == [768, 512]
    assert payload["params"]["fps"] == 24
    assert payload["params"]["duration"] == 3.0
    assert "still.png" in payload["params"]["refs"]
    assert payload["lineage"]["shot_id"] == "shot-3"
    assert payload["lineage"]["attempt_id"] == "shot-3.a2"
    assert payload["lineage"]["iteration"] == 2
    assert payload["lineage"]["parent_shot_id"] == "shot-3"
    assert payload["lineage"]["parent_attempt_id"] == "shot-3.a1"
    assert payload["judge"]["score"] == 0.81
    assert payload["judge"]["fail_reasons"][0]["kind"] == "cpu_fail_rules"
    assert payload["judge"]["fail_reasons"][0]["id"] == "c"
    assert payload["revise_notes"].startswith("quality_bar.c")
    assert payload["created_at"]
    assert payload["hash"] is None  # no file


def test_write_read_sidecar_roundtrip(tmp_path):
    clip = tmp_path / "shot-1.mp4"
    clip.write_bytes(b"fake-mp4-bytes")
    st = RunState(
        request="rain",
        prompt="rain on glass",
        variant="base",
        seed=7,
        attempt=1,
        shot_index=1,
        judge_score=0.4,
        quality_bar={
            "fails": [
                {"id": "a", "code": "missing_music_bed", "detail": "no bed"}
            ]
        },
        video_path=str(clip),
        planned_clip=str(clip),
        output_dir=str(tmp_path),
        run_id="run1",
    )
    payload = persist_clip_provenance(st, revise_notes="first generate")
    dest = sidecar_path(clip)
    assert dest.is_file()
    assert dest.name == "shot-1.buddy.json"
    loaded = read_clip_provenance(clip)
    assert loaded is not None
    _assert_required(loaded)
    assert loaded["prompts"]["positive"] == "rain on glass"
    assert loaded["hash"] == payload["hash"]
    assert loaded["hash"] and len(loaded["hash"]) == 64
    assert loaded["output_path"] == str(clip)
    assert loaded["judge"]["fail_reasons"][0]["kind"] == "cpu_fail_rules"
    assert st.provenance_sidecar == str(dest)


def test_hash_is_null_when_file_missing(tmp_path):
    st = RunState(
        request="x",
        prompt="x",
        run_id="r",
        shot_index=2,
        output_dir=str(tmp_path),
        planned_clip=str(tmp_path / "shot-2.mp4"),
    )
    payload = persist_clip_provenance(st)
    assert payload["hash"] is None
    assert (tmp_path / "shot-2.buddy.json").is_file()
    assert not (tmp_path / "shot-2.mp4").exists()


def test_inherit_rewrites_output_path_and_hash(tmp_path):
    src = tmp_path / "shot-1.mp4"
    dest = tmp_path / "master.mp4"
    src.write_bytes(b"seg")
    dest.write_bytes(b"stitched")
    write_clip_provenance(
        src,
        build_clip_provenance(
            RunState(
                request="seg",
                prompt="seg prompt",
                variant="base",
                seed=1,
                attempt=1,
                shot_index=1,
                video_path=str(src),
            ),
            path=src,
        ),
    )
    payload = inherit_clip_provenance(
        src, dest, revise_notes="pipeline stitch", extra={"brief": "full brief"}
    )
    _assert_required(payload)
    assert payload["prompts"]["brief"] == "full brief"
    assert payload["revise_notes"] == "pipeline stitch"
    assert payload["output_path"] == str(dest)
    assert payload["hash"]
    loaded = read_clip_provenance(dest)
    assert loaded["hash"] == payload["hash"]
    assert sidecar_path(dest).name == "master.buddy.json"


def test_apply_sidecar_is_source_of_truth_before_revise(tmp_path):
    clip = tmp_path / "shot-1.mp4"
    clip.write_bytes(b"v1")
    st = RunState(
        request="brief",
        prompt="original positive",
        seed=11,
        steps=8,
        width=768,
        height=512,
        duration_s=4.0,
        variant="base",
        shot_index=1,
        output_dir=str(tmp_path),
        planned_clip=str(clip),
        video_path=str(clip),
        attempt=1,
        run_id="r",
    )
    persist_clip_provenance(st)
    st.prompt = "drifted"
    st.seed = 99
    loaded = read_clip_provenance(clip)
    apply_sidecar_to_state(st, loaded)
    assert st.prompt == "original positive"
    assert st.seed == 11
    assert st.parent_attempt_id == "shot-1.a1"


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


def test_dry_run_embeds_provenance_and_writes_shot_buddy_json(tmp_path):
    runs = tmp_path / "runs"
    outs = tmp_path / "outputs"
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", runs
    ), patch("master_agent.provenance.OUTPUTS_DIR", outs):
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
    assert st.provenance["prompts"]["brief"].startswith("music video")
    assert st.provenance["engine"]["backend"] == "comfy"
    assert st.provenance["engine"]["workflow_id"] == "base"
    assert st.provenance["params"]["seed"] == 99
    assert st.provenance["lineage"]["shot_id"] == "shot-1"
    assert st.provenance["lineage"]["iteration"] == st.attempt
    assert st.provenance["lineage"]["attempt_id"] == f"shot-1.a{st.attempt}"
    assert st.provenance["lineage"]["parent_attempt_id"]
    assert st.provenance["revise_notes"]
    sidecar = Path(st.provenance_sidecar)
    assert sidecar.name == "shot-1.buddy.json"
    assert sidecar.is_file()
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    _assert_required(loaded)
    assert loaded["hash"] is None
    rec = json.loads(next(runs.glob("*.json")).read_text(encoding="utf-8"))
    _assert_required(rec["provenance"])
    assert rec["provenance"]["lineage"]["attempt_id"] == st.provenance["lineage"]["attempt_id"]


def test_generate_revise_rerun_writes_shot_sidecar_and_history(tmp_path):
    outs = tmp_path / "outputs"
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
        plan_clip_paths(st)
        clip = Path(st.planned_clip)
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
    ), patch(
        "master_agent.provenance.OUTPUTS_DIR", outs
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
    clip = Path(st.planned_clip)
    dest = sidecar_path(clip)
    assert dest.is_file()
    assert dest.name == "shot-1.buddy.json"
    loaded = read_clip_provenance(clip)
    assert loaded is not None
    _assert_required(loaded)
    assert loaded["output_path"] == str(clip)
    assert loaded["hash"]
    assert loaded["lineage"]["iteration"] == st.attempt
    assert loaded["params"]["seed"] == 3
    assert loaded["engine"]["backend"] == "comfy"
    assert any((h.get("judge") or {}).get("fail_reasons") for h in st.provenance_history)
    rec = json.loads(next(runs.glob("*.json")).read_text(encoding="utf-8"))
    _assert_required(rec["provenance"])
    assert rec["provenance_sidecar"].endswith("shot-1.buddy.json")
    assert rec["provenance"]["hash"] == loaded["hash"]
