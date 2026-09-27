"""Long-audio ltx25_a2v segmentation. No GPU. Renders are not queued."""

from __future__ import annotations

import json
from unittest.mock import patch

from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.orchestrator.lipdub import (
    CLOSED_MOUTH_CLAUSE,
    TRIPOD_I2V_STRENGTH,
    TRIPOD_NEGATIVE,
    TRIPOD_POSITIVE,
    WordSpan,
    alignment_errors,
    attach_lipdub_params,
    cuts_inside_words,
    energy_silence_spans,
    format_lipdub_plan,
    lipdub_param_block,
    load_words,
    plan_lipdub,
)
from master_agent.orchestrator.state import RunState
from master_agent.orchestrator.talking import per_clip_cap_s, plan_talking_slices
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    build_clip_provenance,
    missing_required,
    read_clip_provenance,
    write_clip_provenance,
)

# Scott's 12s sample. Whisper smears the 0–0.9s and 5.7–6.25s pauses into words.
RINGMASTER_WORDS = [
    {"w": "I", "s": 0.0, "e": 1.02},
    {"w": "would", "s": 1.02, "e": 1.14},
    {"w": "like", "s": 1.14, "e": 1.38},
    {"w": "to", "s": 1.38, "e": 1.58},
    {"w": "make", "s": 1.58, "e": 1.78},
    {"w": "this", "s": 1.78, "e": 2.08},
    {"w": "a", "s": 2.08, "e": 2.62},
    {"w": "sound", "s": 2.62, "e": 3.52},
    {"w": "sample", "s": 3.52, "e": 4.14},
    {"w": "of", "s": 4.14, "e": 4.44},
    {"w": "what", "s": 4.44, "e": 4.64},
    {"w": "my", "s": 4.64, "e": 4.86},
    {"w": "voice", "s": 4.86, "e": 5.26},
    {"w": "is", "s": 5.26, "e": 5.52},
    {"w": "so", "s": 5.52, "e": 6.34},
    {"w": "we", "s": 6.34, "e": 6.54},
    {"w": "could", "s": 6.54, "e": 6.72},
    {"w": "clone", "s": 6.72, "e": 7.04},
    {"w": "my", "s": 7.04, "e": 7.4},
    {"w": "voice", "s": 7.4, "e": 7.88},
    {"w": "so", "s": 7.88, "e": 8.22},
    {"w": "we", "s": 8.22, "e": 8.54},
    {"w": "could", "s": 8.54, "e": 8.9},
    {"w": "clone", "s": 8.9, "e": 9.46},
    {"w": "whatever", "s": 9.46, "e": 9.72},
    {"w": "we", "s": 9.72, "e": 10.22},
    {"w": "want.", "s": 10.22, "e": 10.48},
    {"w": "Hey,", "s": 11.26, "e": 11.28},
    {"w": "thank", "s": 11.42, "e": 11.5},
    {"w": "you.", "s": 11.5, "e": 11.78},
]
TOWER_PAUSES = [(0.0, 0.9), (5.7, 6.25), (10.6, 11.15)]


def _words(rows=None) -> list[WordSpan]:
    return load_words_from(rows if rows is not None else RINGMASTER_WORDS)


def load_words_from(rows) -> list[WordSpan]:
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "words.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return load_words(str(path))


def test_short_audio_is_one_pass():
    plan = plan_lipdub(5.0, words=_words(), max_segment_s=6.5, base_prompt="she speaks")
    assert plan.segmented is False
    assert plan.pieces == []
    text = format_lipdub_plan(plan)
    assert "single pass" in text
    assert "same render as a short lipdub" in text
    assert CLOSED_MOUTH_CLAUSE not in text
    edge = plan_lipdub(6.5, max_segment_s=6.5, silences=[])
    assert edge.segmented is False


def test_split_never_mid_word_and_respects_max():
    words = []
    t = 0.0
    for label in ("a", "b", "c", "d", "e", "f"):
        words.append(WordSpan(label, t, t + 1.0))
        t += 1.05
    plan = plan_lipdub(
        t,
        words=words,
        silences=[],
        max_segment_s=3.2,
        silence_min_s=0.25,
        fps=24,
    )
    assert plan.segmented is True
    assert cuts_inside_words(plan, words) == []
    assert alignment_errors(plan) == []
    for piece in plan.speech_pieces():
        assert piece.end_s - piece.start_s <= 3.2 + 0.05
        assert (piece.render_frames - 1) % 8 == 0
        assert piece.render_frames >= 9

    long = [WordSpan("supercal", 0.0, 8.0)]
    whole = plan_lipdub(8.0, words=long, silences=[], max_segment_s=3.0, fps=24)
    assert cuts_inside_words(whole, long) == []
    assert len(whole.speech_pieces()) == 1
    assert whole.speech_pieces()[0].end_s - whole.speech_pieces()[0].start_s > 3.0
    assert whole.warning and "mid-word" in whole.warning


def test_hard_split_frame_grid_matches_audio():
    plan = plan_lipdub(12.0, words=[], silences=[], max_segment_s=6.5, overlap_frames=8, fps=24)
    assert plan.segmented is True
    assert plan.timeline_frame_count == 288
    assert sum(p.keep_frames for p in plan.pieces) == 288
    assert alignment_errors(plan) == []
    assert plan.pieces[0].place_start == 0
    assert plan.pieces[-1].place_end == 288
    for piece in plan.speech_pieces():
        assert piece.audio_duration_s * plan.fps == piece.render_frames
        assert abs(piece.audio_start_s - (piece.place_start - piece.drop_leading) / plan.fps) < 1e-6


def test_ringmaster_pauses_close_the_mouth_and_continue():
    words = _words()
    plan = plan_lipdub(
        12.0,
        words=words,
        silences=TOWER_PAUSES,
        max_segment_s=6.5,
        silence_min_s=0.25,
        overlap_frames=8,
        fps=24,
        tripod=True,
        base_prompt="ringmaster",
        seed=7,
    )
    assert plan.segmented is True
    assert plan.timeline_frame_count == 288
    assert alignment_errors(plan) == []
    assert sum(p.keep_frames for p in plan.pieces) == 288
    assert cuts_inside_words(plan, words, allowed_silences=TOWER_PAUSES) == []
    assert plan.pieces[0].kind == "silence_plate"
    assert plan.pieces[0].end_s >= 0.7
    assert plan.pieces[0].hold_frames == plan.pieces[0].keep_frames
    assert plan.pieces[0].render_frames == 0
    bridges = [p for p in plan.pieces if p.kind == "mouth_bridge"]
    assert bridges
    for bridge in bridges:
        assert bridge.continuity == "previous_last_frame"
        assert bridge.render_frames == 9 or bridge.keep_frames <= 9
        assert (bridge.render_frames - 1) % 8 == 0
    speech = plan.speech_pieces()
    assert speech[0].continuity == "still"
    assert all(p.continuity == "previous_last_frame" for p in speech[1:])
    assert any(p.drop_leading == 0 and p.continuity == "previous_last_frame" for p in speech)
    assert plan.tripod is True
    assert "full audio" in format_lipdub_plan(plan)
    for piece in plan.comfy_pieces():
        assert (piece.render_frames - 1) % 8 == 0
        assert piece.render_frames <= 169 or (plan.warning and "safe cap" in plan.warning)


def test_energy_vad_finds_pauses_and_ignores_a_steady_tone():
    import numpy as np

    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    n = sr * 2
    tone = (0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    pcm = np.concatenate([silence, tone, silence])
    spans = energy_silence_spans(pcm, sr, 4.0, min_silence_s=0.25)
    assert spans
    assert spans[0][0] == 0.0
    assert spans[0][1] >= 0.7
    assert spans[-1][1] == 4.0
    assert spans[-1][0] >= 2.5
    steady = (0.25 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    assert energy_silence_spans(steady, sr, 2.0, min_silence_s=0.25) == []
    pure = np.zeros(sr, dtype=np.float32)
    assert energy_silence_spans(pure, sr, 1.0, min_silence_s=0.25) == [(0.0, 1.0)]
    # Pauses that are a small fraction of a 12s line must still count.
    long_n = sr * 12
    long = (0.2 * np.sin(2 * np.pi * 180 * np.arange(long_n) / sr)).astype(np.float32)
    long[: int(0.9 * sr)] = 0
    long[int(5.7 * sr) : int(6.25 * sr)] = 0
    long[int(10.6 * sr) : int(11.15 * sr)] = 0
    found = energy_silence_spans(long, sr, 12.0, min_silence_s=0.25)
    assert any(a <= 0.05 and b >= 0.8 for a, b in found)
    assert any(a <= 5.85 and b >= 6.1 for a, b in found)
    assert any(a <= 10.75 and b >= 11.0 for a, b in found)


def test_provenance_round_trip_stays_on_clip_schema(tmp_path):
    words = _words()
    plan = plan_lipdub(
        12.0,
        words=words,
        silences=TOWER_PAUSES,
        max_segment_s=6.5,
        fps=24,
        tripod=True,
        seed=7,
        base_prompt="ringmaster",
    )
    st = RunState(
        request="ringmaster brief",
        prompt="ringmaster brief",
        negative_prompt="blurry",
        variant="ltx25_a2v",
        run_id="lipdubtest",
        duration_s=12.0,
        seed=7,
        width=352,
        height=480,
        spoken_line="Hey, thank you.",
        shot_id="shot-1",
    )
    payload = build_clip_provenance(st, path=str(tmp_path / "shot-1.mp4"))
    block = lipdub_param_block(plan, audio_sha256="904e9a63")
    attach_lipdub_params(payload, block)
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert "stitched_clip" not in payload["schema"]
    assert missing_required(payload) == []
    assert set(payload["params"]) >= {"seed", "steps", "cfg", "size", "fps", "duration", "refs", "lipdub"}
    lip = payload["params"]["lipdub"]
    assert lip["segmented"] is True
    assert lip["full_audio_mux"] is True
    assert lip["tripod"] is True
    assert lip["audio_codec"] == "aac"
    assert lip["timeline_frames"] == 288
    assert lip["segments"]
    assert {"seed", "attempt", "start_s", "end_s", "source_frame", "continuity"} <= set(lip["segments"][0])
    clip = tmp_path / "shot-1.mp4"
    write_clip_provenance(clip, payload)
    loaded = read_clip_provenance(clip)
    assert loaded["schema"] == CLIP_PROVENANCE_SCHEMA
    assert loaded["params"]["lipdub"]["split_points_s"] == lip["split_points_s"]
    assert missing_required(loaded) == []


def test_tripod_flag_reaches_a2v_workflow_and_default_strength_stays():
    prompt = "ringmaster " + TRIPOD_POSITIVE
    negative = f"blurry, low quality, distorted face, watermark, text overlay, {TRIPOD_NEGATIVE}"
    with patch(
        "master_agent.comfy.workflow_patcher._text_enhancer_filename",
        return_value=None,
    ):
        locked, _meta = load_and_patch_workflow(
            "ltx25_a2v",
            prompt=prompt,
            negative_prompt=negative,
            seed=7,
            duration_s=3.0,
            image_name="face.png",
            audio_name="voice.wav",
            i2v_strength=TRIPOD_I2V_STRENGTH,
        )
        plain, _meta2 = load_and_patch_workflow(
            "ltx25_a2v",
            prompt="she speaks",
            seed=1,
            duration_s=3.0,
            image_name="face.png",
            audio_name="voice.wav",
        )
    assert "locked-off" in locked["5508"]["inputs"]["value"]
    assert "push-in" in locked["5509"]["inputs"]["value"]
    # The a2v manifest writes the negative onto the CLIP encode input. The
    # primitive keeps the same string so either path reaches the sampler.
    clip_neg = locked["5014:2612"]["inputs"]["text"]
    assert clip_neg == ["5509", 0] or (
        isinstance(clip_neg, str) and "push-in" in clip_neg
    )
    strengths = {
        node["inputs"]["strength"]
        for node in locked.values()
        if isinstance(node, dict) and node.get("class_type") == "LTXVImgToVideoInplace"
    }
    assert TRIPOD_I2V_STRENGTH in strengths
    assert 1 in strengths or 1.0 in strengths
    plain_strengths = [
        node["inputs"]["strength"]
        for node in plain.values()
        if isinstance(node, dict) and node.get("class_type") == "LTXVImgToVideoInplace"
    ]
    assert 0.7 in plain_strengths
    assert plain_strengths[-1] == 1 or plain_strengths[-1] == 1.0


def test_h3_cap_unchanged_and_a2v_cap_is_the_lipdub_threshold():
    assert per_clip_cap_s("h3_r2v") == 12.0
    assert abs(per_clip_cap_s("ltx25_a2v") - 6.5) < 1e-6
    h3 = plan_talking_slices(20, variant="h3_r2v")
    assert h3.durations == [12.0]
    ltx = plan_talking_slices(12, variant="ltx25_a2v")
    assert len(ltx.durations) > 1
    assert ltx.audio_starts[0] == 0.0
    assert ltx.audio_starts[1] == ltx.durations[0]
