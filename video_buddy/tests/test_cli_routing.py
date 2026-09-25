"""Photo + voice CLI routing. No ComfyUI, no GPU, no weight downloads.

Run: python -m pytest tests/test_cli_routing.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from master_agent.__main__ import _music_intent, cmd_run
from master_agent.orchestrator.talking import (
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
