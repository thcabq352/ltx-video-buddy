"""Likeness lock: pause-reset anchor and short speech pieces. No GPU."""

from __future__ import annotations

import copy
import json

import numpy as np

from master_agent.comfy.validator import validate_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import OBJECT_INFO_CACHE
from master_agent.orchestrator.lipdub import (
    ANCHOR_PAUSE_RESET,
    CONTINUITY_PREV,
    CONTINUITY_SOURCE,
    CONTINUITY_STILL,
    WordSpan,
    alignment_errors,
    cuts_inside_words,
    lipdub_param_block,
    plan_lipdub,
    _word_interior,
    quietest_cut,
    resolve_pause_reset,
    snap_words_to_frames,
    split_span_at_lowest_energy,
)
from master_agent.orchestrator.lipdub_guide import (
    CROP_CLASS,
    GUIDE_CLASS,
    GUIDE_MISSING_MESSAGE,
    apply_last_frame_guide,
    guide_nodes_present,
    probe_pause_reset_guide,
)
from master_agent.orchestrator.pipeline import _is_topology_error
from master_agent.provenance import CLIP_PROVENANCE_SCHEMA
from master_agent.setup import check_ltx_guide

# Same smeared timestamps the tower plan uses.
RINGMASTER = [
    ("I", 0.0, 1.02),
    ("would", 1.02, 1.14),
    ("like", 1.14, 1.38),
    ("to", 1.38, 1.58),
    ("make", 1.58, 1.78),
    ("this", 1.78, 2.08),
    ("a", 2.08, 2.62),
    ("sound", 2.62, 3.52),
    ("sample", 3.52, 4.14),
    ("of", 4.14, 4.44),
    ("what", 4.44, 4.64),
    ("my", 4.64, 4.86),
    ("voice", 4.86, 5.26),
    ("is", 5.26, 5.52),
    ("so", 5.52, 6.34),
    ("we", 6.34, 6.54),
    ("could", 6.54, 6.72),
    ("clone", 6.72, 7.04),
    ("my", 7.04, 7.4),
    ("voice", 7.4, 7.88),
    ("so", 7.88, 8.22),
    ("we", 8.22, 8.54),
    ("could", 8.54, 8.9),
    ("clone", 8.9, 9.46),
    ("whatever", 9.46, 9.72),
    ("we", 9.72, 10.22),
    ("want.", 10.22, 10.48),
    ("Hey,", 11.26, 11.28),
    ("thank", 11.42, 11.5),
    ("you.", 11.5, 11.78),
]
PAUSES = [(0.0, 0.9), (5.7, 6.25), (10.6, 11.15)]


def _words() -> list[WordSpan]:
    return [WordSpan(text, start, end) for text, start, end in RINGMASTER]


def _pause_plan(**kwargs):
    params = dict(
        words=_words(),
        silences=PAUSES,
        max_segment_s=6.5,
        max_piece_s=6.5,
        silence_min_s=0.25,
        overlap_frames=8,
        fps=24,
        silence_mode="idle",
        anchor="pause-reset",
        reframe=False,
        seed=42,
    )
    params.update(kwargs)
    return plan_lipdub(12.0, **params)


def test_max_piece_splits_on_the_quietest_frame_and_stays_on_the_grid():
    sr = 16000
    duration = 8.0
    pcm = np.full(int(sr * duration), 0.2, dtype=np.float32)
    # Quiet dips inside each 3s window. The cut must land on them, on a frame.
    pcm[int(2.70 * sr) : int(2.90 * sr)] = 0.001
    pcm[int(5.40 * sr) : int(5.60 * sr)] = 0.001
    plan = plan_lipdub(
        duration,
        pcm=pcm,
        sample_rate=sr,
        silences=[],
        words=[],
        max_segment_s=6.5,
        max_piece_s=3.0,
        fps=24,
        anchor="previous",
        reframe=False,
    )
    assert plan.segmented is True
    assert plan.max_piece_s == 3.0
    speech = plan.speech_pieces()
    assert len(speech) >= 3
    assert alignment_errors(plan) == []
    assert sum(p.keep_frames for p in plan.pieces) == plan.timeline_frame_count
    for piece in speech:
        assert piece.keep_frames / plan.fps <= 3.0 + (1.0 / plan.fps) + 1e-6
        assert abs(piece.audio_start_s * plan.fps - (piece.place_start - piece.drop_leading)) < 1e-6
        assert abs(piece.audio_duration_s * plan.fps - piece.render_frames) < 1e-6
    cuts = [p.end_s for p in speech[:-1]]
    assert cuts
    assert any(abs(cut - 2.8) < 0.15 for cut in cuts)
    assert any(abs(cut - 5.5) < 0.2 for cut in cuts)
    for cut in cuts:
        assert abs(cut * 24 - round(cut * 24)) < 1e-6


def test_quietest_cut_prefers_a_word_edge_over_a_louder_interior():
    fps = 24
    # Energy is lowest in the middle of a word; a quieter-but-not-quietest edge exists.
    times = [i / fps for i in range(0, 24 * 4)]
    values = [0.5 for _ in times]
    for i, t in enumerate(times):
        if 1.4 <= t <= 1.6:
            values[i] = 0.01  # inside "hello" 1.0-2.0
        if abs(t - 2.0) < 0.02:
            values[i] = 0.05
    words = [WordSpan("hello", 1.0, 2.0), WordSpan("there", 2.0, 3.2)]
    cut = quietest_cut(0.0, 4.0, 3.0, fps=fps, times=times, values=values, words=words)
    assert cut is not None
    assert abs(cut - 2.0) < 0.05
    assert not (1.0 + 1e-3 < cut < 2.0 - 1e-3)
    spans = split_span_at_lowest_energy(
        0.0, 4.0, 3.0, fps=fps, track=(times, values), words=words
    )
    assert spans[0][1] == cut


def test_whisper_20ms_grid_splits_on_snapped_frames_at_24fps():
    """Faster-whisper edges are 0.02s. At 24fps those match a frame only every 0.5s.

    Before the snap, the opening speech run has no legal cut, so the piece cap
    warns and keeps the run whole. After the snap the quietest edge is a real
    frame and the ringmaster line becomes 9 pieces.
    """
    words = _words()
    for word in words:
        assert abs(word.start / 0.02 - round(word.start / 0.02)) < 1e-6
        assert abs(word.end / 0.02 - round(word.end / 0.02)) < 1e-6
    # 2.625s is the frame that holds the 2.62s edge between "a" and "sound".
    assert _word_interior(2.625, words)
    snapped = snap_words_to_frames(words, 24)
    assert not _word_interior(2.625, snapped)
    sr = 16000
    pcm = np.full(int(sr * 12), 0.30, dtype=np.float32)
    for word in snapped:
        for edge in (word.start, word.end):
            i0 = int(round(edge * sr))
            pcm[max(0, i0 - int(0.02 * sr)) : i0 + int(0.02 * sr)] = 0.08
    # The long "a" (2.08–2.62) is the quietest snapped edge in the first speech run.
    dip = int(round(2.625 * sr))
    pcm[max(0, dip - int(0.03 * sr)) : dip + int(0.03 * sr)] = 0.001
    plan = plan_lipdub(
        12.0,
        words=words,
        silences=PAUSES,
        pcm=pcm,
        sample_rate=sr,
        max_segment_s=6.5,
        max_piece_s=3.0,
        fps=24,
        overlap_frames=8,
        silence_mode="idle",
        anchor="pause-reset",
        reframe=False,
        seed=42,
    )
    assert plan.warning is None or "kept whole" not in plan.warning
    assert len(plan.pieces) == 9
    assert len(plan.speech_pieces()) == 6
    assert len(plan.idle_pieces()) == 3
    # Deep dip at the snapped end of "a" (2.625s); shallower dips on the other
    # snapped edges. Neighboring 20ms dips overlap, so the second-run cut is
    # 8.542s rather than the later 8.917s boundary.
    ends = [round(p.end_s, 3) for p in plan.pieces]
    assert ends == [0.875, 2.625, 4.875, 5.708, 6.250, 8.542, 10.500, 11.250, 12.000]
    assert [p.render_frames for p in plan.pieces] == [25, 49, 65, 33, 17, 57, 57, 25, 25]
    assert alignment_errors(plan) == []
    assert cuts_inside_words(plan, words, allowed_silences=PAUSES) == []
    assert sum(p.keep_frames for p in plan.pieces) == 288
    speech_ends = [p.end_s for p in plan.speech_pieces()[:-1]]
    assert any(abs(end - 2.625) < 1e-6 for end in speech_ends)
    for piece in plan.speech_pieces():
        assert piece.end_s - piece.start_s <= 3.0 + (1.0 / 24) + 1e-6
    guided = [p for p in plan.pieces if p.end_keyframe == "source_still"]
    assert guided
    assert all(p.end_s - p.start_s + 1e-9 >= plan.pause_reset_min_s for p in guided)
    assert all(p.guide_strength == plan.pause_reset_strength for p in guided)
    assert plan.pieces[0].end_keyframe == ""


def test_pause_shorter_than_half_a_second_is_plain_idle():
    words = [WordSpan("hello", 0.0, 1.2), WordSpan("there", 1.55, 3.2)]
    plan = plan_lipdub(
        4.0,
        words=words,
        silences=[(1.2, 1.55)],
        max_segment_s=2.0,
        max_piece_s=6.5,
        fps=24,
        silence_mode="idle",
        anchor="pause-reset",
        reframe=False,
    )
    bridge = [p for p in plan.pieces if p.kind == "mouth_bridge"]
    assert bridge
    assert all(p.end_s - p.start_s < 0.5 for p in bridge)
    assert all(p.end_keyframe == "" for p in bridge)
    assert all(p.guide_frame_idx is None for p in bridge)
    assert all(p.continuity == CONTINUITY_PREV for p in bridge)
    pinned = [p for p in plan.pieces if p.end_keyframe == "source_still"]
    assert pinned
    assert all(p.end_s - p.start_s + 1e-9 >= 0.5 for p in pinned)


def test_pause_reset_anchor_plan_per_piece():
    plan = _pause_plan()
    assert plan.anchor == ANCHOR_PAUSE_RESET
    assert plan.continuity_method == "pause_reset"
    assert plan.reframe is False
    assert alignment_errors(plan) == []
    assert sum(p.keep_frames for p in plan.pieces) == 288
    assert cuts_inside_words(plan, _words(), allowed_silences=PAUSES) == []
    assert plan.pieces[0].kind == "silence_idle"
    assert plan.pieces[0].continuity == CONTINUITY_STILL
    assert plan.pieces[0].end_keyframe == ""
    guided = [p for p in plan.pieces if p.end_keyframe == "source_still"]
    assert guided
    assert all(p.kind in ("silence_idle", "mouth_bridge") for p in guided)
    assert all(p.continuity == CONTINUITY_PREV for p in guided)
    assert all(p.end_s - p.start_s + 1e-9 >= 0.5 for p in guided)
    assert all(p.guide_frame_idx == p.keep_frames - 1 - 4 for p in guided)
    assert all(p.guide_strength == 0.65 for p in guided)
    assert all(p.silence_crossfade_frames == 0 for p in guided)
    assert all(p.crossfade_frames == 0 for p in plan.speech_pieces())
    speech = plan.speech_pieces()
    assert speech[0].continuity == CONTINUITY_PREV
    assert all(p.end_keyframe == "" for p in speech)
    # A speech-to-speech seam trims overlap. A speech piece after a silence does not.
    after_silence = []
    for prev, piece in zip(plan.pieces, plan.pieces[1:]):
        if piece.kind == "speech" and prev.kind != "speech":
            after_silence.append(piece)
    assert after_silence
    assert all(p.drop_leading == 0 for p in after_silence)
    block = lipdub_param_block(plan)
    assert block["anchor"] == "pause-reset"
    assert block["max_piece_s"] == 6.5
    assert block["pause_reset"] is None
    assert block["pause_reset_strength"] == 0.65
    assert block["pause_reset_min_s"] == 0.5
    assert any(seg["end_keyframe"] == "source_still" for seg in block["segments"])
    assert "settles back" in guided[0].prompt


def test_pause_reset_fallback_crossfades_inside_silence_only():
    plan = _pause_plan()
    before = [(p.index, p.continuity, p.crossfade_frames, p.drop_leading) for p in plan.speech_pieces()]
    resolve_pause_reset(plan, guide_available=False)
    assert plan.guide_available is False
    assert plan.guide_node is None
    assert plan.pause_reset == "silence_crossfade"
    assert all(p.end_keyframe == "" for p in plan.pieces)
    silenced = [p for p in plan.pieces if p.kind in ("silence_idle", "mouth_bridge") and p.index > 0]
    assert silenced
    for piece in silenced:
        assert piece.continuity == CONTINUITY_SOURCE
        assert 0 < piece.silence_crossfade_frames <= min(8, piece.keep_frames)
        assert piece.crossfade_frames == 0
    after = [(p.index, p.continuity, p.crossfade_frames, p.drop_leading) for p in plan.speech_pieces()]
    assert after == before
    assert all(p.silence_crossfade_frames == 0 for p in plan.speech_pieces())
    assert alignment_errors(plan) == []
    block = lipdub_param_block(plan)
    assert block["pause_reset"] == "silence_crossfade"
    assert block["guide_available"] is False


def test_pause_reset_guide_path_records_optional_provenance_fields():
    plan = _pause_plan()
    resolve_pause_reset(plan, guide_available=True)
    block = lipdub_param_block(plan, audio_sha256="abc")
    assert block["pause_reset"] == "end_keyframe"
    assert block["guide_node"] == "LTXVAddGuide"
    assert block["guide_available"] is True
    payload = {"schema": CLIP_PROVENANCE_SCHEMA, "params": {"lipdub": block}}
    assert payload["schema"] == CLIP_PROVENANCE_SCHEMA
    assert set(block) >= {
        "anchor",
        "max_piece_s",
        "guide_node",
        "guide_available",
        "pause_reset",
        "pause_reset_strength",
        "pause_reset_min_s",
        "segments",
    }
    assert block["pause_reset_strength"] == 0.65
    assert block["pause_reset_min_s"] == 0.5


def test_workflow_patch_inserts_guide_on_both_stages():
    info = {GUIDE_CLASS: {}, CROP_CLASS: {}}
    base, _meta = load_and_patch_workflow(
        "ltx25_a2v",
        prompt="the ringmaster speaks",
        seed=42,
        duration_s=1.0,
        width=352,
        height=480,
        image_name="prev_last.png",
        audio_name="voice.wav",
        frames=25,
    )
    original = copy.deepcopy(base)
    patched, patch = apply_last_frame_guide(
        base, image_name="source_still.png", object_info=info
    )
    assert patch.applied
    assert patch.status == "applied"
    assert base == original
    guides = [n for n in patched.values() if isinstance(n, dict) and n.get("class_type") == GUIDE_CLASS]
    crops = [n for n in patched.values() if isinstance(n, dict) and n.get("class_type") == CROP_CLASS]
    assert len(guides) == 2
    assert len(crops) == 2
    assert all(node["inputs"]["frame_idx"] == -1 for node in guides)
    assert all(node["inputs"]["strength"] == 1.0 for node in guides)
    assert all(node["inputs"]["image"] == [patch.image_node, 0] for node in guides)
    still = patched[patch.image_node]
    assert still["class_type"] == "LoadImage"
    assert still["inputs"]["image"] == "source_still.png"
    # Frame 0 is still the previous-frame LoadImage, not the end still.
    first_images = [
        node["inputs"]["image"]
        for node in patched.values()
        if isinstance(node, dict) and node.get("class_type") == "LoadImage" and node is not still
    ]
    assert "prev_last.png" in first_images
    for node in patched.values():
        if isinstance(node, dict) and node.get("class_type") == "LTXVConcatAVLatent":
            src = node["inputs"]["video_latent"][0]
            assert patched[src]["class_type"] == GUIDE_CLASS
        if isinstance(node, dict) and node.get("class_type") == "CFGGuider":
            assert patched[node["inputs"]["positive"][0]]["class_type"] == GUIDE_CLASS
            assert patched[node["inputs"]["negative"][0]]["class_type"] == GUIDE_CLASS
        if isinstance(node, dict) and node.get("class_type") in ("LTXVLatentUpsampler", "VAEDecodeTiled"):
            src = node["inputs"]["samples"][0]
            assert patched[src]["class_type"] == CROP_CLASS
    softened, soft_patch = apply_last_frame_guide(
        base,
        image_name="source_still.png",
        object_info=info,
        frame_idx=8,
        strength=0.65,
    )
    assert soft_patch.applied
    soft_guides = [
        n for n in softened.values() if isinstance(n, dict) and n.get("class_type") == GUIDE_CLASS
    ]
    assert soft_guides
    assert all(node["inputs"]["frame_idx"] == 8 for node in soft_guides)
    assert all(node["inputs"]["strength"] == 0.65 for node in soft_guides)


def test_workflow_patch_without_guide_node_leaves_the_graph_and_says_why():
    base, _meta = load_and_patch_workflow(
        "ltx25_a2v",
        prompt="the ringmaster speaks",
        seed=42,
        duration_s=1.0,
        image_name="prev_last.png",
        audio_name="voice.wav",
        frames=25,
    )
    original = copy.deepcopy(base)
    returned, patch = apply_last_frame_guide(base, image_name="source_still.png", object_info={})
    assert returned is base
    assert base == original
    assert patch.status == "missing"
    assert not patch.applied
    assert "LTXVAddGuide" in patch.message
    assert "LTXVCropGuides" in patch.message
    assert "not a new download" in patch.message
    assert "inside the silence" in patch.message
    assert GUIDE_MISSING_MESSAGE.split(".")[0] in patch.message
    assert not any(
        isinstance(n, dict) and n.get("class_type") == GUIDE_CLASS for n in returned.values()
    )
    bare = {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}
    _same, bad = apply_last_frame_guide(
        bare, image_name="still.png", object_info={GUIDE_CLASS: {}, CROP_CLASS: {}}
    )
    assert bad.status == "infeasible"
    assert _same == bare


def test_shipped_a2v_graph_accepts_the_guide_against_cached_object_info():
    assert OBJECT_INFO_CACHE.is_file()
    info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
    assert guide_nodes_present(info)
    available, message = probe_pause_reset_guide(info)
    assert available, message
    assert "LTXVAddGuide" in message
    row = check_ltx_guide()
    assert row["name"] == "ltx-guide"
    assert row["ok"] is True
    assert "LTXVAddGuide" in row["detail"]
    missing_ok, missing_message = probe_pause_reset_guide({})
    assert missing_ok is False
    assert "not a new download" in missing_message
    assert "LTXVAddGuide" in missing_message

    wf, _meta = load_and_patch_workflow(
        "ltx25_a2v",
        prompt="ringmaster",
        seed=42,
        duration_s=1.0,
        width=352,
        height=480,
        image_name="prev.png",
        audio_name="voice.wav",
        frames=25,
    )
    patched, patch = apply_last_frame_guide(wf, image_name="still.png", object_info=info)
    assert patch.applied
    report = validate_workflow(patched, info, file_label="pause-reset")
    topology = [err for err in report.errors if _is_topology_error(err)]
    assert topology == [], topology

