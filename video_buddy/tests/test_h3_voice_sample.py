"""H3 voice-sample gate, spoken line, and --no-judge. No GPU, no Comfy, no network.

Run: python -m pytest tests/test_h3_voice_sample.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from master_agent.orchestrator.h3_voice import (
    H3_MISSING_LINE_WARNING,
    H3_VOICE_MAX_S,
    TRIM_METHOD,
    VoiceSampleError,
    best_speech_window,
    h3_missing_line_warning,
    inject_spoken_line,
    optional_local_transcript,
    prepare_h3_voice_sample,
)
from master_agent.orchestrator.machine import Orchestrator
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA, build_clip_provenance, missing_required
from master_agent.orchestrator.state import RunState


def test_under_two_seconds_rejected_before_queue(tmp_path: Path):
    src = tmp_path / "short.wav"
    src.write_bytes(b"stub")
    with pytest.raises(VoiceSampleError) as exc:
        prepare_h3_voice_sample(str(src), probe=lambda _p: 1.25)
    message = str(exc.value)
    assert "1.25" in message
    assert "2" in message
    assert "Nothing was queued" in message


def test_two_to_twelve_seconds_passthrough(tmp_path: Path):
    src = tmp_path / "ok.wav"
    src.write_bytes(b"stub")
    sample = prepare_h3_voice_sample(str(src), probe=lambda _p: 7.5)
    assert sample.trimmed is False
    assert sample.path == str(src)
    assert sample.method == "passthrough"
    assert sample.start_s == 0.0
    assert sample.end_s == 7.5
    edge = prepare_h3_voice_sample(str(src), probe=lambda _p: 2.0)
    assert edge.trimmed is False
    cap = prepare_h3_voice_sample(str(src), probe=lambda _p: H3_VOICE_MAX_S)
    assert cap.trimmed is False


def test_over_twelve_seconds_trimmed_with_provenance(tmp_path: Path):
    src = tmp_path / "long.wav"
    src.write_bytes(b"stub")
    sr = 8000
    pcm = np.zeros(sr * 20, dtype=np.float32)
    tone = np.arange(sr * 12) / sr
    pcm[sr * 6 : sr * 18] = (0.8 * np.sin(2 * np.pi * 180 * tone)).astype(np.float32)
    captured: dict = {}

    def _decode(_path):
        return pcm, sr

    def _trim(_src, dest, start, duration):
        captured["start"] = start
        captured["duration"] = duration
        Path(dest).write_bytes(b"RIFF")

    sample = prepare_h3_voice_sample(
        str(src),
        probe=lambda _p: 20.0,
        decode=_decode,
        trim=_trim,
        out_dir=tmp_path,
    )
    assert sample.trimmed is True
    assert sample.method == TRIM_METHOD
    assert abs(captured["duration"] - 12.0) < 1e-6
    assert abs(sample.end_s - sample.start_s - 12.0) < 0.05
    assert abs(sample.start_s - 6.0) < 0.25
    assert Path(sample.path).is_file()
    record = sample.provenance()
    assert record["original_duration_s"] == 20.0
    assert record["method"] == TRIM_METHOD
    assert record["start_s"] == sample.start_s
    assert record["end_s"] == sample.end_s

    st = RunState(request="hailuo close-up", prompt="close-up", variant="h3_r2v")
    st.voice_sample = record
    st.spoken_line = "Hey there."
    payload = build_clip_provenance(st)
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert missing_required(payload) == []
    assert payload["params"]["voice_sample"]["original_duration_s"] == 20.0
    assert payload["params"]["voice_sample"]["method"] == TRIM_METHOD
    assert payload["prompts"]["spoken_line"] == "Hey there."


def test_best_window_selects_continuous_speech_not_the_first_twelve():
    sr = 8000
    pcm = np.zeros(sr * 20, dtype=np.float32)
    tone = np.arange(sr * 12) / sr
    pcm[sr * 6 : sr * 18] = (0.8 * np.sin(2 * np.pi * 180 * tone)).astype(np.float32)
    start, end = best_speech_window(pcm, sr, window_s=12.0)
    assert abs(start - 6.0) < 0.25
    assert abs(end - start - 12.0) < 0.05


def test_best_window_prefers_continuous_over_gappy_spikes():
    sr = 8000
    pcm = np.zeros(sr * 30, dtype=np.float32)
    frame = int(sr * 0.1)
    head = pcm[: sr * 12]
    for i in range(0, len(head), frame * 2):
        head[i : i + frame] = 1.0
    pcm[: sr * 12] = head
    tone = np.arange(sr * 12) / sr
    pcm[sr * 15 : sr * 27] = (0.55 * np.sin(2 * np.pi * 200 * tone)).astype(np.float32)
    start, _end = best_speech_window(pcm, sr, window_s=12.0)
    assert 14.0 < start < 16.5


def test_word_for_word_prompt_injection_keeps_exact_line():
    line = "Hey there, Keep Local AI runs on your own machine."
    prompt = inject_spoken_line(
        "The ringmaster clown speaks directly to camera, lips synced to the voice",
        line,
    )
    assert line in prompt
    assert "saying word for word:" in prompt
    assert inject_spoken_line(prompt, line) == prompt

    apostrophe = "it's your machine"
    quoted = inject_spoken_line("Close-up of the clown", apostrophe)
    assert apostrophe in quoted
    assert f'"{apostrophe}"' in quoted

    both = 'say "hello" — it\'s live'
    mixed = inject_spoken_line("Close-up", both)
    assert both in mixed


def test_h3_node_prompt_contains_the_line_exactly():
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow

    line = "Hey there, Keep Local AI runs on your own machine."
    tricky = 'say "hello" — it\'s live'
    wf, _meta = load_and_patch_workflow(
        "h3_r2v",
        prompt="The ringmaster clown speaks directly to camera",
        seed=1,
        duration_s=4.0,
        image_name="face.png",
        audio_name="sample.wav",
        spoken_line=tricky,
    )
    refs = [
        node
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3ReferenceToVideo"
    ]
    assert refs
    text = refs[0]["inputs"]["prompt"]
    assert tricky in text
    assert "saying word for word:" in text.lower()
    plain, _meta = load_and_patch_workflow(
        "h3_r2v",
        prompt="The ringmaster clown speaks directly to camera",
        seed=1,
        duration_s=4.0,
        image_name="face.png",
        audio_name="sample.wav",
        spoken_line=line,
    )
    plain_refs = [
        node
        for node in plain.values()
        if isinstance(node, dict) and node.get("class_type") == "MiniMaxH3ReferenceToVideo"
    ]
    assert line in plain_refs[0]["inputs"]["prompt"]


def test_missing_line_warning_is_explicit_and_not_a_failure():
    assert h3_missing_line_warning("", h3_voice=True) == H3_MISSING_LINE_WARNING
    assert "line" in H3_MISSING_LINE_WARNING.lower()
    assert h3_missing_line_warning("Hello there", h3_voice=True) is None
    assert h3_missing_line_warning("", h3_voice=False) is None


def test_transcriber_skipped_without_a_local_model(monkeypatch):
    monkeypatch.delenv("H3_LOCAL_TRANSCRIBER_MODEL", raising=False)
    assert optional_local_transcript("sample.wav") is None


def test_cli_short_sample_does_not_open_comfy(monkeypatch, capsys, tmp_path: Path):
    from master_agent.__main__ import cmd_run

    audio = tmp_path / "short.wav"
    audio.write_bytes(b"x")

    def _boom(_path, **_kwargs):
        raise VoiceSampleError(
            "H3 voice sample is 1.20s, shorter than 2 s. "
            "Record at least 2 seconds of the reference voice (up to 12 s) and "
            "pass it with --audio. Nothing was queued."
        )

    monkeypatch.setattr(
        "master_agent.orchestrator.h3_voice.prepare_h3_voice_sample",
        _boom,
    )

    def _client(*_a, **_k):
        raise AssertionError("Comfy client constructed before the length gate")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", _client)
    rc = cmd_run(
        Namespace(
            request="hailuo speaks",
            variant="h3_r2v",
            duration=5.0,
            duration_set=False,
            quality="draft",
            seed=1,
            width=768,
            height=512,
            video=None,
            image=str(tmp_path / "face.png"),
            audio=str(audio),
            line="Hello there friend",
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
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "Nothing was queued" in out


def test_cli_missing_line_warns(monkeypatch, capsys, tmp_path: Path):
    from master_agent.__main__ import cmd_run

    audio = tmp_path / "line.wav"
    audio.write_bytes(b"wav")
    monkeypatch.setattr("master_agent.music.beats.audio_duration", lambda _p: 4.0)

    class _Client:
        def is_up(self):
            return True

        def upload_image(self, _path):
            raise OSError("stop after warning")

        def upload_audio(self, _path):
            raise OSError("stop after warning")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", lambda *a, **k: _Client())
    rc = cmd_run(
        Namespace(
            request="hailuo, she says the line",
            variant="h3_r2v",
            duration=5.0,
            duration_set=False,
            quality="draft",
            seed=1,
            width=768,
            height=512,
            video=None,
            image=str(tmp_path / "face.png"),
            audio=str(audio),
            line=None,
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
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert H3_MISSING_LINE_WARNING in out


def test_no_judge_submits_once_and_does_not_revise(tmp_path: Path):
    submits = {"n": 0}
    orch = Orchestrator()

    def ok_patch(_st):
        return True

    def ok_validate(_st):
        return True

    def ok_submit(_st):
        submits["n"] += 1
        return True

    def ok_resolve(st):
        st.video_path = str(tmp_path / "shot.mp4")
        return True

    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs"
    ), patch("master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"), patch.object(
        orch, "_patch", ok_patch
    ), patch.object(orch, "_validate", ok_validate), patch.object(
        orch, "_submit_and_poll", ok_submit
    ), patch.object(orch, "_resolve", ok_resolve):
        st = orch.run(
            "music video for a synthwave track",
            kind="music_video",
            variant="base",
            judge_enabled=False,
            revise_enabled=False,
            dry_run=False,
            max_judge_rounds=3,
            voice_sample={
                "original_duration_s": 18.0,
                "start_s": 3.0,
                "end_s": 15.0,
                "method": TRIM_METHOD,
                "trimmed": True,
            },
        )
    assert submits["n"] == 1
    assert st.attempt == 1
    assert st.revise_history == []
    assert st.quality_bar == {}
    assert st.judge_decision == "skipped"
    assert not any("revise applied" in msg for msg in st.messages)
    assert st.provenance["params"]["voice_sample"]["start_s"] == 3.0
    assert st.provenance["params"]["voice_sample"]["end_s"] == 15.0
    assert st.provenance["params"]["voice_sample"]["method"] == TRIM_METHOD
    assert st.provenance["params"]["voice_sample"]["original_duration_s"] == 18.0
    assert missing_required(st.provenance) == []
    assert st.provenance["schema"] == CLIP_PROVENANCE_SCHEMA
