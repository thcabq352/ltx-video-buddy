"""Rainey1 batch top-cut. No GPU, no Comfy queue.

Run: python -m pytest tests/test_rainey1_batch.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.comfy.client import ComfyClient
from master_agent.config import DOWNSCALE_LADDER, is_valid_ltx_frames
from master_agent.judge.probe import TINY_FILE_BYTES
from master_agent.orchestrator.director_presets import recipe_by_id
from master_agent.orchestrator.oom import apply_oom_downscale
from master_agent.orchestrator.state import RunState
from master_agent.provenance import missing_required, read_clip_provenance
from master_agent.rainey1.batch import (
    SeedRow,
    format_batch_report,
    is_junk_output,
    rank_top_k,
    run_rainey1_batch,
)
from master_agent.rainey1.recipes import LOOK_FLOOR, resolve_recipe
from master_agent.__main__ import main


def test_help_documents_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rainey1-batch", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--recipe",
        "--seeds",
        "--top-k",
        "--out",
        "--dry-run",
        "--no-interview",
        "--allow-empty",
        "lock_open",
        "breach",
        "density",
        "myth_16x9",
        "story_9x16",
    ):
        assert flag in out
    assert "zero Comfy" in out
    assert "operator tower" in out


def test_recipes_snap_to_8n_plus_1():
    for slug in ("lock_open", "breach", "density", "myth_16x9", "story_9x16"):
        recipe = resolve_recipe(slug)
        assert is_valid_ltx_frames(recipe.frames)
        assert recipe.frames != 8
        assert recipe.width % 32 == 0
        assert recipe.height % 32 == 0
    assert resolve_recipe("lock_open").frames == 9
    assert resolve_recipe("breach").frames == 17
    assert resolve_recipe("density").frames == 25
    assert resolve_recipe("myth_16x9").frames == 25
    assert resolve_recipe("story_9x16").frames == 17
    story = resolve_recipe("rainey1_story_9x16")
    packed = recipe_by_id("rainey1_story_9x16")
    assert (story.width, story.height, story.frames) == (packed.width, packed.height, packed.frames)
    assert story.source == "preset"
    assert "ffmpeg" in story.notes


def test_resolve_recipe_uses_shipped_preset():
    recipe = resolve_recipe("lock_open")
    packed = recipe_by_id("rainey1_lock_open")
    assert recipe.source == "preset"
    assert recipe.id == packed.id
    assert recipe.frames == packed.frames == 9
    assert (recipe.width, recipe.height) == (packed.width, packed.height)
    assert recipe.negative
    assert recipe.additives


def test_dry_run_queues_zero_comfy_jobs(tmp_path, monkeypatch, capsys):
    calls: list[object] = []

    def boom(self, workflow):
        calls.append(workflow)
        raise AssertionError("queue_prompt must not run on --dry-run")

    def live_boom(*_args, **_kwargs):
        raise AssertionError("live generate must not run on --dry-run")

    monkeypatch.setattr(ComfyClient, "queue_prompt", boom)
    monkeypatch.setattr("master_agent.rainey1.batch._generate_live", live_boom)
    out = tmp_path / "topcut"
    rc = main(
        [
            "rainey1-batch",
            "--recipe",
            "lock_open",
            "--seeds",
            "42,43,44,45",
            "--top-k",
            "2",
            "--out",
            str(out),
            "--dry-run",
            "--no-interview",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    assert calls == []
    assert "comfy_jobs_queued=0" in printed
    assert "dry-run: queued zero Comfy jobs" in printed
    assert not out.exists() or not any(out.rglob("*.mp4"))
    assert "42" in printed and "drop" in printed


def test_rank_top_k_by_look_then_brief():
    rows = [
        SeedRow(seed=1, recipe_id="rainey1_lock_open", look_score=0.90, brief_adherence=0.20),
        SeedRow(seed=2, recipe_id="rainey1_lock_open", look_score=0.80, brief_adherence=0.70),
        SeedRow(seed=3, recipe_id="rainey1_lock_open", look_score=0.80, brief_adherence=0.60),
        SeedRow(seed=4, recipe_id="rainey1_lock_open", look_score=0.70, brief_adherence=0.99),
        SeedRow(seed=5, recipe_id="rainey1_lock_open", look_score=0.40, brief_adherence=0.99),
        SeedRow(
            seed=6,
            recipe_id="rainey1_lock_open",
            look_score=0.99,
            brief_adherence=0.99,
            junk=True,
            junk_reason="tiny",
        ),
    ]
    keepers = rank_top_k(rows, 2, look_floor=LOOK_FLOOR)
    assert [row.seed for row in keepers] == [2, 3]
    assert [row.rank for row in keepers] == [1, 2]
    by_seed = {row.seed: row for row in rows}
    assert by_seed[1].decision == "drop"
    assert by_seed[1].human_veto is True
    assert by_seed[4].decision == "drop"
    assert by_seed[5].decision == "drop"
    assert by_seed[6].decision == "drop"
    assert by_seed[6].keep is False
    assert all(row.decision in {"keep", "drop"} for row in rows)


def test_junk_filter_skips_judge(tmp_path):
    judged: list[int] = []
    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"x" * 50)
    fat = tmp_path / "fat.mp4"
    fat.write_bytes(b"x" * (TINY_FILE_BYTES + 10))

    def generate(recipe, seed, *, dry_run):
        assert dry_run is False
        return (tiny if seed == 1 else fat), 0

    def judge(recipe, seed, path):
        judged.append(seed)
        from master_agent.rainey1.batch import JudgeView

        return JudgeView(look_score=0.9, brief_adherence=0.8, fail_reasons=[])

    result = run_rainey1_batch(
        recipe="breach",
        seeds=[1, 2],
        top_k=1,
        out=tmp_path / "out",
        dry_run=False,
        llm_judge=False,
        generate_fn=generate,
        judge_fn=judge,
        probe_fn=lambda path: {
            "size_bytes": path.stat().st_size if path else 0,
            "frames": 9,
            "exists": bool(path and path.is_file()),
        },
    )
    assert judged == [2]
    assert result.comfy_jobs_queued == 0
    assert result.rows[0].junk is True
    assert result.rows[0].decision == "drop"
    assert result.keepers[0].seed == 2
    assert result.exit_code == 0


def test_keepers_write_schema_valid_provenance(tmp_path):
    clip = tmp_path / "staging.mp4"
    clip.write_bytes(b"frame-bytes" * 20000)

    def generate(recipe, seed, *, dry_run):
        return clip, 0

    def judge(recipe, seed, path):
        from master_agent.rainey1.batch import JudgeView

        return JudgeView(
            look_score=0.91 if seed == 7 else 0.62,
            brief_adherence=0.8,
            fail_reasons=["soft grain"] if seed == 8 else [],
        )

    out = tmp_path / "keepers"
    result = run_rainey1_batch(
        recipe="density",
        seeds="7,8,9",
        top_k=2,
        out=out,
        dry_run=False,
        generate_fn=generate,
        judge_fn=judge,
        probe_fn=lambda path: {"size_bytes": TINY_FILE_BYTES + 5, "frames": 25},
    )
    assert result.exit_code == 0
    assert [row.seed for row in result.keepers] == [7, 8]
    for row in result.keepers:
        sidecar = Path(row.path).with_name(f"seed-{row.seed}.buddy.json")
        assert sidecar.is_file()
        payload = read_clip_provenance(row.path)
        assert payload is not None
        assert missing_required(payload) == []
        assert payload["schema"] == "buddy.clip.provenance/v1"
        assert payload["params"]["seed"] == row.seed
        assert payload["rainey1"]["recipe_id"] == "rainey1_density"
        assert payload["rainey1"]["look_score"] == row.look_score
        assert payload["prompts"]["additives"]
        assert payload["hash"]
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["rainey1"]["kept"] is True
    assert result.rows[2].decision == "drop"


def test_zero_keepers_exit_nonzero_unless_allow_empty(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * (TINY_FILE_BYTES + 1))

    def generate(recipe, seed, *, dry_run):
        return clip, 0

    def judge(recipe, seed, path):
        from master_agent.rainey1.batch import JudgeView

        return JudgeView(look_score=0.2, brief_adherence=0.9)

    common = dict(
        recipe="lock_open",
        seeds=[1],
        top_k=2,
        dry_run=False,
        generate_fn=generate,
        judge_fn=judge,
        probe_fn=lambda path: {"size_bytes": TINY_FILE_BYTES + 1, "frames": 9},
    )
    blocked = run_rainey1_batch(**common, out=tmp_path / "empty")
    assert blocked.exit_code == 1
    assert blocked.keepers == []
    assert "zero keepers" in format_batch_report(blocked)
    allowed = run_rainey1_batch(**common, allow_empty=True, out=tmp_path / "empty2")
    assert allowed.exit_code == 0


def test_is_junk_thresholds(tmp_path):
    missing, why = is_junk_output(None, {})
    assert missing and "missing" in why
    small = tmp_path / "small.mp4"
    small.write_bytes(b"nope")
    junk, reason = is_junk_output(small, {"size_bytes": 10, "frames": 9})
    assert junk and "size" in reason
    short = tmp_path / "short.mp4"
    short.write_bytes(b"x" * (TINY_FILE_BYTES + 1))
    junk, reason = is_junk_output(short, {"size_bytes": TINY_FILE_BYTES + 1, "frames": 1})
    assert junk and "frames" in reason
    ok, _ = is_junk_output(short, {"size_bytes": TINY_FILE_BYTES + 1, "frames": None})
    assert ok is False


def test_batch_selects_shipped_rainey1_rubric():
    from master_agent.judge.judge import judge_system_prompt

    text = judge_system_prompt(
        user_request="locked character at a desk",
        context={"judge_rubric": "rainey1"},
    )
    assert "Rainey1 caliber" in text
    plain = judge_system_prompt(user_request="locked character at a desk")
    assert "Rainey1 caliber" not in plain


def test_oom_downscale_writes_ladder_frames():
    state = RunState(request="rainey1", variant="base", downscale_level=0)
    assert apply_oom_downscale(state, DOWNSCALE_LADDER) is True
    assert (state.width, state.height, state.frames) == DOWNSCALE_LADDER[1]
    assert is_valid_ltx_frames(state.frames)
    state.downscale_level = len(DOWNSCALE_LADDER) - 1
    assert apply_oom_downscale(state, DOWNSCALE_LADDER) is False


def test_refuses_dataset_output(tmp_path):
    dataset = tmp_path / "training" / "datasets" / "styles" / "rainey1_caliber"
    with pytest.raises(ValueError, match="training/datasets"):
        run_rainey1_batch(
            recipe="lock_open",
            seeds=[1],
            out=dataset,
            dry_run=True,
        )


def test_one_seed_at_a_time(tmp_path):
    order: list[int] = []
    inflight = {"n": 0, "max": 0}

    def generate(recipe, seed, *, dry_run):
        inflight["n"] += 1
        inflight["max"] = max(inflight["max"], inflight["n"])
        order.append(seed)
        inflight["n"] -= 1
        return None, 0

    result = run_rainey1_batch(
        recipe="myth_16x9",
        seeds=[4, 5, 6],
        dry_run=False,
        allow_empty=True,
        out=tmp_path / "none",
        generate_fn=generate,
        probe_fn=lambda path: {"size_bytes": 0, "frames": 0},
    )
    assert order == [4, 5, 6]
    assert inflight["max"] == 1
    assert result.comfy_jobs_queued == 0
    assert result.exit_code == 0
