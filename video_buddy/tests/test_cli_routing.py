"""Photo + voice CLI routing. No ComfyUI, no GPU, no weight downloads.

Run: python -m pytest tests/test_cli_routing.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from master_agent.__main__ import _music_intent, cmd_run
from master_agent.orchestrator.talking import (
    H3_R2V_AUDIO_LABEL,
    plan_talking_slices,
    skip_music_autoroute,
)


def _ns(**overrides) -> Namespace:
    base = dict(
        request="dub",
        variant="lipsync",
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
        dry_run=False,
        self_improve_dry=False,
        power_mode=False,
        no_power_mode=False,
        upscale=None,
        no_interview=True,
        attach=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_lipsync_without_video_fails(capsys):
    rc = cmd_run(_ns())
    out = capsys.readouterr().out
    assert rc == 1
    assert "source video" in out
    assert "ltx25_a2v" in out


def test_image_skips_music_autoroute(monkeypatch, capsys):
    monkeypatch.setattr("master_agent.music.beats.audio_duration", lambda _path: 12.0)
    assert _music_intent("hello", "x.wav", "draft") is True
    assert skip_music_autoroute("hello", has_image=True) is True
    assert skip_music_autoroute("lip sync the voice", has_image=False) is True
    assert skip_music_autoroute("a landscape", has_image=False) is False

    called = {}

    def _boom(*_a, **_k):
        called["hit"] = True
        return {"status": "done", "video_path": "x.mp4"}

    monkeypatch.setattr("master_agent.music.pipeline.run_music_video", _boom)

    class _Client:
        def is_up(self):
            return True

        def upload_image(self, _path):
            raise OSError("missing image")

        def upload_audio(self, _path):
            raise OSError("missing audio")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", lambda *a, **k: _Client())
    rc = cmd_run(
        _ns(
            request="music video for a soundtrack",
            variant=None,
            image="face.png",
            audio="song.wav",
        )
    )
    out = capsys.readouterr().out
    assert called == {}
    assert rc == 1
    assert "photo + voice" in out
    assert H3_R2V_AUDIO_LABEL not in out


def test_h3_named_route_prints_voice_reference_warning(monkeypatch, capsys):
    monkeypatch.setattr("master_agent.music.beats.audio_duration", lambda _path: 3.9)

    class _Client:
        def is_up(self):
            return True

        def upload_image(self, _path):
            raise OSError("missing image")

        def upload_audio(self, _path):
            raise OSError("missing audio")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", lambda *a, **k: _Client())

    rc = cmd_run(
        _ns(
            request="hailuo, she says the line",
            variant=None,
            image="gator.png",
            audio="line.wav",
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "photo + voice → h3_r2v" in out
    assert f"warn: {H3_R2V_AUDIO_LABEL}" in out

    rc = cmd_run(
        _ns(
            request="she says the line",
            variant="h3_r2v",
            image="gator.png",
            audio="line.wav",
        )
    )
    forced = capsys.readouterr().out
    assert rc == 1
    assert "photo + voice → h3_r2v" in forced
    assert H3_R2V_AUDIO_LABEL in forced

    rc = cmd_run(
        _ns(
            request="she says the line",
            variant=None,
            image="gator.png",
            audio="line.wav",
        )
    )
    default = capsys.readouterr().out
    assert rc == 1
    assert "photo + voice → ltx25_a2v" in default
    assert H3_R2V_AUDIO_LABEL not in default


def test_run_help_and_workflows_list_label_h3_r2v(capsys):
    from master_agent.__main__ import cmd_workflows, main

    try:
        main(["run", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_out = " ".join(capsys.readouterr().out.split())
    assert H3_R2V_AUDIO_LABEL in help_out

    assert cmd_workflows(Namespace(json=False, vram=False)) == 0
    listing = capsys.readouterr().out
    line = next(row for row in listing.splitlines() if row.strip().startswith("h3_r2v"))
    assert H3_R2V_AUDIO_LABEL in line


def test_mcp_create_video_warns_on_h3_photo_voice(monkeypatch, tmp_path: Path):
    image = tmp_path / "gator.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")

    class _Result:
        status = "done"
        video_path = "out.mp4"
        segment_paths = []
        segment_scores = []
        full_judge_score = 0.0
        full_judge_pass = True
        full_judge_notes = ""
        storyboard = []
        panel_meta = {}
        error = None

    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.run_pipeline",
        lambda *_a, **_k: _Result(),
    )
    monkeypatch.setattr(
        "master_agent.comfy.client.ComfyClient",
        lambda *a, **k: type("C", (), {
            "upload_image": lambda self, path: Path(path).name,
            "upload_audio": lambda self, path: Path(path).name,
        })(),
    )
    monkeypatch.setattr(
        "master_agent.orchestrator.talking.duration_following_audio",
        lambda _path, probe=None: (3.9, None),
    )
    # Live H3 voice mode probes the sample before queue. Stub bytes have
    # duration 0, so the gate must see a 2–12 s sample.
    monkeypatch.setattr("master_agent.music.beats.audio_duration", lambda _path: 4.0)
    from master_agent.mcp_server import create_video

    warned = create_video(
        "a cartoon gator says the line",
        variant="h3_r2v",
        image_path=str(image),
        audio_path=str(audio),
    )
    assert warned["status"] == "done"
    assert warned["warning"] == H3_R2V_AUDIO_LABEL
    assert warned["notes"] == H3_R2V_AUDIO_LABEL
    assert H3_R2V_AUDIO_LABEL in (create_video.__doc__ or "")

    quiet = create_video(
        "she says the line",
        image_path=str(image),
        audio_path=str(audio),
    )
    assert quiet["warning"] is None
    assert quiet["notes"] is None


def test_talking_slices_h3_is_one_clip_and_ltx_continues():
    h3 = plan_talking_slices(20, variant="h3_r2v")
    assert h3.durations == [12.0]
    assert h3.audio_starts == [0.0]
    assert h3.note
    ltx = plan_talking_slices(12, variant="ltx25_a2v")
    assert len(ltx.durations) > 1
    assert ltx.audio_starts[0] == 0.0
    assert ltx.audio_starts[1] == ltx.durations[0]


def test_mcp_create_video_forwards_image_and_audio(monkeypatch, tmp_path: Path):
    image = tmp_path / "face.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")
    captured: dict = {}

    class _Result:
        status = "done"
        video_path = "out.mp4"
        segment_paths = []
        segment_scores = []
        full_judge_score = 0.0
        full_judge_pass = True
        full_judge_notes = ""
        storyboard = []
        panel_meta = {}
        error = None

    def _run(*_a, **kwargs):
        captured.update(kwargs)
        return _Result()

    class _Client:
        def upload_image(self, path):
            return Path(path).name

        def upload_audio(self, path):
            return Path(path).name

    monkeypatch.setattr("master_agent.orchestrator.pipeline.run_pipeline", _run)
    monkeypatch.setattr("master_agent.comfy.client.ComfyClient", lambda *a, **k: _Client())
    monkeypatch.setattr(
        "master_agent.orchestrator.talking.duration_following_audio",
        lambda _path, probe=None: (3.2, None),
    )
    from master_agent.mcp_server import create_video

    out = create_video(
        "she says the line",
        image_path=str(image),
        audio_path=str(audio),
    )
    assert out["status"] == "done"
    assert captured["image_name"] == "face.png"
    assert captured["audio_name"] == "line.wav"
    assert captured["duration_s"] == 3.2
    missing = create_video("x", image_path=str(tmp_path / "nope.png"))
    assert missing["status"] == "error"
    assert "not found" in missing["error"]
