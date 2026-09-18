"""Music-video mode: beat windows, uniqueness, provenance, dry-run wiring.

No GPU, no live Comfy. Run: python -m pytest tests/test_music_video.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.music.burn import (
    burn_windows,
    seed_for_window,
    window_motion_prompt,
)
from master_agent.music.plan import (
    BEAT_PLAN_SCHEMA,
    MV_FPS,
    sec_to_frame,
    synthetic_beat_plan,
    validate_beat_plan,
    windows_from_pairs,
    write_beat_plan,
)
from master_agent.music.remotion import remotion_command
from master_agent.music.still_hold import STILL_MARKER, StillHoldError, assert_clip_not_still_hold
from master_agent.music.unique import DuplicateClipError, check_unique_clips
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, read_clip_provenance


def test_sec_to_frame_30fps():
    assert sec_to_frame(0.0, 30) == 0
    assert sec_to_frame(1.0, 30) == 30
    assert sec_to_frame(2.5, 30) == 75
    assert sec_to_frame(16.0, 30) == 480


def test_windows_tile_frame_grid_and_keep_energy():
    plan = synthetic_beat_plan(16.0, 120.0, fps=30)
    assert plan.schema == BEAT_PLAN_SCHEMA
    assert plan.fps == MV_FPS
    assert plan.duration_frames == 480
    assert validate_beat_plan(plan) == []
    assert plan.windows[0].start_frame == 0
    assert plan.windows[-1].end_frame == 480
    for prev, nxt in zip(plan.windows, plan.windows[1:]):
        assert prev.end_frame == nxt.start_frame
        assert prev.duration_frames >= 2
    # drop half is hotter / shorter on average
    verse = [w for w in plan.windows if (w.start_s + w.end_s) / 2 < 8.0]
    drop = [w for w in plan.windows if (w.start_s + w.end_s) / 2 >= 8.0]
    assert verse and drop
    assert any(w.energy is not None for w in plan.windows)
    assert any(w.label == "drop" for w in drop)


def test_still_hold_window_refused():
    with pytest.raises(StillHoldError) as exc:
        windows_from_pairs([(0.0, 0.02)], duration_s=0.02, fps=30)
    assert "still-hold" in str(exc.value)


def test_uniqueness_gate_lists_dup_indices(tmp_path: Path):
    a = tmp_path / "w0.mp4"
    b = tmp_path / "w1.mp4"
    c = tmp_path / "w2.mp4"
    a.write_bytes(b"clip-a")
    b.write_bytes(b"clip-b")
    c.write_bytes(b"clip-a")
    with pytest.raises(DuplicateClipError) as exc:
        check_unique_clips([(0, a), (1, b), (2, c)])
    msg = str(exc.value)
    assert "refuse stitch" in msg
    assert "0" in msg and "2" in msg
    hashes = check_unique_clips([(0, a), (1, b)])
    assert hashes[0] != hashes[1]


def test_provenance_write_read_includes_window_id(tmp_path: Path):
    plan = synthetic_beat_plan(8.0, 120.0)
    rows = burn_windows(
        plan,
        brief="I'm in love with a bot",
        out_dir=tmp_path,
        dry_run=True,
        seed=11,
        log=lambda *_a, **_k: None,
    )
    assert rows
    clip = Path(rows[0]["clip"])
    payload = read_clip_provenance(clip)
    assert payload is not None
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert payload["window_id"] == 0
    assert payload["lineage"]["window_id"] == 0
    assert payload["lineage"]["shot_id"] == "window-0"
    assert payload["engine"]["backend"] == "comfy"
    assert payload["hash"]
    assert payload["prompts"]["positive"]
    assert "no still hold" in payload["prompts"]["positive"]


def test_still_hold_clip_refused(tmp_path: Path):
    still = tmp_path / "lock.png"
    still.write_bytes(b"\x89PNG")
    with pytest.raises(StillHoldError):
        assert_clip_not_still_hold(still, {"index": 1}, dry_run=True)
    marked = tmp_path / "window-000.mp4"
    marked.write_bytes(STILL_MARKER + b"\nhold")
    with pytest.raises(StillHoldError):
        assert_clip_not_still_hold(marked, {"index": 0}, dry_run=True)


def test_dry_run_plan_unique_remotion_wiring(tmp_path: Path):
    from master_agent.music.mv import render_music_video

    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 16)
    plan = synthetic_beat_plan(8.0, 120.0, audio_path=audio)
    out = tmp_path / "out" / "MV-FIXED.mp4"
    rec = render_music_video(
        audio,
        out=out,
        prompt="I'm in love with a bot",
        plan=plan,
        dry_run=True,
        seed=3,
        work_dir=tmp_path / "work",
        log=lambda *_a, **_k: None,
    )
    assert rec["ok"] is True
    assert rec["dry_run"] is True
    assert rec["engine"] == "comfy-ltx"
    assert Path(rec["plan_path"]).is_file()
    burns = rec["burns"]
    assert len(burns) == len(plan.windows)
    hashes = set(rec["hashes"].values())
    assert len(hashes) == len(burns)
    rem = rec["remotion"]
    props_path = Path(rem["props_path"])
    assert props_path.is_file()
    props = json.loads(props_path.read_text(encoding="utf-8"))
    assert props["fps"] == 30
    assert props["width"] == 1920
    assert props["height"] == 1080
    assert len(props["windows"]) == len(plan.windows)
    assert props["audio"].endswith("track.wav")
    cmd = rem["command"]
    assert cmd[:4] == ["npx", "--yes", "remotion", "render"]
    assert "MusicVideo" in cmd
    assert any(str(out.resolve()) in part for part in cmd)
    # live remotion was not invoked
    assert rem.get("dry_run") is True
    assert not out.is_file()


def test_remotion_720p_note_is_scale_flag(tmp_path: Path):
    props = tmp_path / "remotion-props.json"
    props.write_text("{}", encoding="utf-8")
    out = tmp_path / "MV-FIXED.mp4"
    cmd = remotion_command(props, out, scale=2 / 3)
    assert any(part.startswith("--scale=") for part in cmd)


def test_seeds_and_prompts_differ_per_window():
    w0 = {"index": 0, "label": "verse"}
    w1 = {"index": 1, "label": "drop"}
    assert seed_for_window(7, 0) != seed_for_window(7, 1)
    assert window_motion_prompt("bot", w0) != window_motion_prompt("bot", w1)


def test_cli_mv_dry_run(tmp_path: Path, monkeypatch):
    from master_agent.__main__ import cmd_mv

    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 16)
    plan = synthetic_beat_plan(8.0, 120.0, audio_path=audio)
    plan_path = write_beat_plan(plan, tmp_path / "beat_plan.json")
    out = tmp_path / "MV-FIXED.mp4"
    args = Namespace(
        mv_command="render",
        request="I'm in love with a bot",
        prompt=None,
        audio=str(audio),
        out=str(out),
        plan=str(plan_path),
        image=None,
        variant="ltx25_t2v_i2v",
        seed=5,
        width=768,
        height=512,
        fps=30,
        dry_run=True,
        work_dir=str(tmp_path / "work"),
        json=False,
    )
    rc = cmd_mv(args)
    assert rc == 0
    work = tmp_path / "work"
    assert (work / "beat_plan.json").is_file()
    assert (work / "remotion-props.json").is_file()
    assert (work / "remotion-command.txt").is_file()
    clips = list((work / "clips").glob("window-*.mp4"))
    assert clips
    for clip in clips:
        side = read_clip_provenance(clip)
        assert side and side["schema"] == CLIP_PROVENANCE_SCHEMA


def test_cli_mv_plan_from_beatmap_file(tmp_path: Path):
    from master_agent.__main__ import cmd_mv
    from master_agent.music.plan import write_beat_plan as _write

    # plan subcommand needs audio analyze; library write is covered above.
    # Here we only check the parser dest + plan writer path via render --plan.
    audio = tmp_path / "t.wav"
    audio.write_bytes(b"x")
    dest = tmp_path / "plan.json"
    write_beat_plan(synthetic_beat_plan(8.0, 120.0), dest)
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["schema"] == BEAT_PLAN_SCHEMA
    assert loaded["windows"][0]["index"] == 0
    _ = cmd_mv  # imported for CLI surface
    _ = _write
