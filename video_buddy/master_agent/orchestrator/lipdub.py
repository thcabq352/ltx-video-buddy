"""Long-audio segmentation for ``ltx25_a2v`` lipdub.

A still plus a voice wav is one talking clip. Short audio (at or under the
configurable threshold, default 6.5s) stays a single pass — the same render
as today. Longer audio is split on pauses, each slice is rendered at an
``8n+1`` frame count, and the pieces are stitched with the original wav
muxed back on.

Root cause of the 12s tower failure (main ``9494eb6``, mouth dead after
about 7.4s), checked against the graph and the LTX-2.5 architecture:

* There is no hard 7-second audio-attention mask. LTX-2.5 a2v uses
  bidirectional audio/video cross-attention with 1D temporal RoPE, and the
  cloud API accepts input audio well past 12s.
* Locally, ``frames_for_duration`` caps every LTX 2.5 latent at 193 frames
  (``LTX25_SEGMENT_MAX_S`` ≈ 8.04s) while ``Trim Audio Duration`` is still
  given the full wav. The latent ends, the audio keeps going, and the mouth
  holds. 7.4s × 24 fps lands on the legal count **177 frames = 7.375s**,
  which is *before* that 193-frame cap, so the distilled latent also stops
  tracking audio inside the window. VAEDecodeTiled (``temporal_size`` 64)
  does not explain it: a 12s clip is ~37 video latents, under one tile.
* The safe default is 6.5s. With an 8-frame overlap and ``8n+1`` snap the
  render stays at or under 169 frames (7.04s), under the observed cliff.

Continuity: the a2v graph has one ``LoadImage`` and no guide / keyframe
input, so a second conditioning node is not added. Default ``--anchor hybrid``
puts the original still on that image slot (identity and framing). The
previous piece's last frame is only a continuity guide: a low-weight blend
into the conditioning image, and a crossfade over the trimmed overlap so
the head does not snap back to the still. ``--anchor previous`` is the #34
chain (each piece starts from the previous last frame). ``--anchor source``
is the still alone, with the same overlap crossfade.

Silence (≥ ``silence_min_s``, default 250ms), ``--silence-mode``:

* ``idle`` (default): the pause is an a2v closed-mouth idle (breathing,
  blink, micro head motion) fed the real pause slice of the wav. Holds
  longer than a few frames are not used. A pause shorter than 9 frames
  stays a trimmed mouth bridge.
* ``hold``: #34 behaviour. Leading silence is a still plate. A longer pause
  is a 9-frame close-mouth bridge plus a hold of that frame.
* ``bridge``: every pause is that 9-frame bridge (leading included). The
  tail past 9 frames is still a hold.

``--reframe`` (default on for a segmented lipdub, off for a single short
pass) measures zoom and shift against the source still and scales the
piece back. ``--tripod`` is unchanged: prompt, negative, and stage-1
``LTXVImgToVideoInplace`` strength ``0.85``. The graph has no camera-motion
input. Stage 2 stays pinned at 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from master_agent.config import (
    DEFAULT_FPS,
    LIPDUB_OVERLAP_FRAMES,
    LIPDUB_SEGMENT_MAX_S,
    LIPDUB_SILENCE_MIN_S,
)

# 7.4s at 24 fps sits on this legal count. Renders above the safe cap warn.
OBSERVED_CLIFF_FRAMES = 177
SAFE_RENDER_FRAMES = 169
BRIDGE_FRAMES = 9
# Idle mode never leaves a frozen plate longer than this. Snap leftovers
# are trimmed from the render, not held.
MAX_FROZEN_HOLD = 3
TRIPOD_I2V_STRENGTH = 0.85
BRIDGE_I2V_STRENGTH = 0.80
IDLE_I2V_STRENGTH = 0.85

SILENCE_IDLE = "idle"
SILENCE_HOLD = "hold"
SILENCE_BRIDGE = "bridge"
ANCHOR_SOURCE = "source"
ANCHOR_PREVIOUS = "previous"
ANCHOR_HYBRID = "hybrid"

CLOSED_MOUTH_CLAUSE = (
    "Lips tightly synced to this audio. Lips press fully closed on every m, b, and p. "
    "The mouth is closed at the first and last moments of this slice."
)
TRIPOD_POSITIVE = (
    "Static locked-off tripod camera, no push-in, no zoom, no dolly, no camera drift. "
    "The head stays planted. Coat buttons and the cane stay rigid."
)
TRIPOD_NEGATIVE = (
    "camera push-in, zoom in, dolly in, camera drift, handheld, head bob, "
    "morphing buttons, warping cane, warping coat"
)
SILENCE_NEGATIVE = "open mouth during silence, talking with no audio, frozen open jaw"
BRIDGE_PROMPT = (
    "The same person holds this exact pose. The mouth closes and rests fully closed, "
    "lips together, no speech, no head turn, no camera move."
)
IDLE_PROMPT = (
    "The same person as the source still. Mouth fully closed, lips together, no speech. "
    "Subtle breathing in the chest and shoulders, one slow blink, and tiny natural "
    "head micro-motion. Face paint, costume, and portrait framing stay locked to the still."
)
IDLE_NEGATIVE = (
    "talking, speaking, open mouth, parted lips, teeth, jaw moving, lip sync, "
    "camera push-in, zoom in, dolly in, camera drift, handheld, head turn, morphing face paint"
)
ANCHOR_CLAUSE = (
    "Same face, face paint, costume, and portrait framing as the source still. "
    "No push-in, no zoom, no change in the face paint."
)
DEFAULT_NEGATIVE = "blurry, low quality, distorted face, watermark, text overlay"

CONTINUITY_STILL = "still"
CONTINUITY_PREV = "previous_last_frame"
CONTINUITY_SOURCE = "source_still"
CONTINUITY_HYBRID = "hybrid"
KIND_SPEECH = "speech"
KIND_PLATE = "silence_plate"
KIND_BRIDGE = "mouth_bridge"
KIND_IDLE = "silence_idle"
JOIN_CROSSFADE = "overlap_crossfade"
JOIN_TRIM = "trim_overlap"
JOIN_NONE = "none"


def lipdub_segment_max_s() -> float:
    return float(LIPDUB_SEGMENT_MAX_S)


def snap_ltx_frames_at_least(n: int) -> int:
    """Smallest legal LTX count (8k+1, minimum 9) that is >= ``n``."""
    raw = max(int(n), 9)
    k = int(math.ceil((raw - 1) / 8.0))
    return k * 8 + 1


def timeline_frames(duration_s: float, fps: int = DEFAULT_FPS) -> int:
    """Frame count whose duration matches the audio within half a frame."""
    frames = int(round(float(duration_s) * int(fps)))
    return max(frames, 1)


@dataclass
class WordSpan:
    text: str
    start: float
    end: float


@dataclass
class LipdubPiece:
    index: int
    kind: str
    start_s: float
    end_s: float
    place_start: int
    place_end: int
    spoken_line: str = ""
    prompt: str = ""
    negative: str = ""
    continuity: str = CONTINUITY_STILL
    render_frames: int = 0
    drop_leading: int = 0
    generated_keep: int = 0
    hold_frames: int = 0
    audio_start_s: float = 0.0
    audio_duration_s: float = 0.0
    i2v_strength: Optional[float] = None
    seed: Optional[int] = None
    crossfade_frames: int = 0
    scale: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    audio_feed: str = "source_wav_slice"

    @property
    def keep_frames(self) -> int:
        return int(self.place_end) - int(self.place_start)

    def to_record(self) -> dict:
        """One segment inside ``params.lipdub.segments`` (existing schema)."""
        end_inclusive = self.place_end - 1
        return {
            "index": self.index,
            "kind": self.kind,
            "seed": self.seed,
            "attempt": 1,
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "place_frames": [self.place_start, end_inclusive],
            "render_frames": self.render_frames,
            "keep_frames": self.keep_frames,
            "drop_leading_frames": self.drop_leading,
            "hold_frames": self.hold_frames,
            "audio_start_s": round(self.audio_start_s, 4),
            "audio_duration_s": round(self.audio_duration_s, 4),
            "source_frame": self.continuity,
            "continuity": self.continuity,
            "spoken_line": self.spoken_line,
            "crossfade_frames": int(self.crossfade_frames),
            "scale": self.scale,
            "dx": self.dx,
            "dy": self.dy,
            "audio_feed": self.audio_feed,
            "prompt_id": None,
            "output_path": None,
            "hash": None,
        }


@dataclass
class LipdubPlan:
    segmented: bool
    duration_s: float
    fps: int
    max_segment_s: float
    silence_min_s: float
    overlap_frames: int
    tripod: bool
    timeline_frame_count: int
    pieces: list[LipdubPiece] = field(default_factory=list)
    split_points_s: list[float] = field(default_factory=list)
    continuity_method: str = "none"
    silence_handling: str = "none"
    silence_mode: str = SILENCE_IDLE
    anchor: str = ANCHOR_HYBRID
    reframe: bool = False
    join_method: str = JOIN_NONE
    scale_drift: list = field(default_factory=list)
    warning: Optional[str] = None
    full_audio_mux: bool = True

    def comfy_pieces(self) -> list[LipdubPiece]:
        return [p for p in self.pieces if p.kind in (KIND_SPEECH, KIND_BRIDGE, KIND_IDLE)]

    def idle_pieces(self) -> list[LipdubPiece]:
        return [p for p in self.pieces if p.kind == KIND_IDLE]

    def speech_pieces(self) -> list[LipdubPiece]:
        return [p for p in self.pieces if p.kind == KIND_SPEECH]


def load_words(path: str) -> list[WordSpan]:
    """Read ``[{w,s,e}, ...]`` or ``{"words": [...]}`` (faster-whisper style)."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("words") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"word file {path} must be a list or {{\"words\": [...]}}")
    words: list[WordSpan] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("w") or row.get("word") or row.get("text") or "").strip()
        start = float(row.get("s") if row.get("s") is not None else row.get("start"))
        end = float(row.get("e") if row.get("e") is not None else row.get("end"))
        if end < start:
            start, end = end, start
        words.append(WordSpan(text=text, start=start, end=end))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def tripod_conditioning(request: str) -> tuple[str, str, float]:
    """Prompt, negative, and first-frame strength for a locked-off talking head."""
    prompt = (request or "").strip()
    if TRIPOD_POSITIVE not in prompt:
        prompt = (prompt + " " + TRIPOD_POSITIVE).strip()
    negative = f"{DEFAULT_NEGATIVE}, {TRIPOD_NEGATIVE}"
    return prompt, negative, TRIPOD_I2V_STRENGTH


def _segment_negative(*, tripod: bool) -> str:
    parts = [DEFAULT_NEGATIVE, SILENCE_NEGATIVE]
    if tripod:
        parts.append(TRIPOD_NEGATIVE)
    return ", ".join(parts)


def _speech_prompt(base: str, spoken: str, *, tripod: bool, anchor: str = ANCHOR_PREVIOUS) -> str:
    parts = [(base or "").strip()]
    if spoken:
        parts.append(f"This slice says: {spoken}")
    parts.append(CLOSED_MOUTH_CLAUSE)
    if anchor in (ANCHOR_SOURCE, ANCHOR_HYBRID):
        parts.append(ANCHOR_CLAUSE)
    if tripod:
        parts.append(TRIPOD_POSITIVE)
    return " ".join(p for p in parts if p)


def _bridge_prompt(*, tripod: bool) -> str:
    if tripod:
        return f"{BRIDGE_PROMPT} {TRIPOD_POSITIVE}"
    return BRIDGE_PROMPT


def _idle_prompt(*, tripod: bool, anchor: str) -> str:
    parts = [IDLE_PROMPT]
    if anchor in (ANCHOR_SOURCE, ANCHOR_HYBRID):
        parts.append(ANCHOR_CLAUSE)
    if tripod:
        parts.append(TRIPOD_POSITIVE)
    return " ".join(parts)


def _idle_negative(*, tripod: bool) -> str:
    parts = [DEFAULT_NEGATIVE, IDLE_NEGATIVE]
    if tripod:
        parts.append(TRIPOD_NEGATIVE)
    return ", ".join(parts)


def _norm_silence_mode(value: Optional[str]) -> str:
    if value is None or str(value).strip() == "":
        return SILENCE_IDLE
    mode = str(value).strip().lower()
    if mode not in (SILENCE_IDLE, SILENCE_HOLD, SILENCE_BRIDGE):
        raise ValueError("silence_mode must be idle, hold, or bridge")
    return mode


def _norm_anchor(value: Optional[str]) -> str:
    if value is None or str(value).strip() == "":
        return ANCHOR_HYBRID
    anchor = str(value).strip().lower()
    if anchor not in (ANCHOR_SOURCE, ANCHOR_PREVIOUS, ANCHOR_HYBRID):
        raise ValueError("anchor must be source, previous, or hybrid")
    return anchor


def energy_silence_spans(
    pcm,
    sample_rate: int,
    duration_s: float,
    *,
    min_silence_s: float = LIPDUB_SILENCE_MIN_S,
) -> list[tuple[float, float]]:
    """RMS pauses. A flat signal (no dynamic range) is not marked silent."""
    import numpy as np

    samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    duration = float(duration_s)
    if samples.size == 0 or sr <= 0 or duration <= 0:
        return []
    win = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.01))
    limit = max(1, samples.size - win + 1)
    rms: list[float] = []
    times: list[float] = []
    for i in range(0, limit, hop):
        chunk = samples[i : i + win]
        rms.append(math.sqrt(float(np.dot(chunk, chunk)) / float(chunk.size) + 1e-12))
        times.append(i / float(sr))
    if not rms:
        return []
    values = np.asarray(rms, dtype=np.float64)
    # 5th percentile, not the 20th: a 12s line can be ~85% speech, so the
    # quieter fifth still has to be able to see a 0.5–0.9s pause.
    floor = float(np.percentile(values, 5))
    peak = float(np.percentile(values, 90))
    if peak <= 1e-4:
        if duration >= min_silence_s:
            return [(0.0, duration)]
        return []
    # A steady tone has no floor/peak gap. Speech against a pause does.
    if peak <= floor * 1.8 + 1e-6:
        return []
    thresh = floor + (peak - floor) * 0.22
    silent = values < thresh
    # Swallow sub-50ms spikes so one click does not split a pause.
    hop_s = hop / float(sr)
    min_speech_hops = max(1, int(round(0.05 / hop_s)))
    i = 0
    n = len(silent)
    while i < n:
        if silent[i]:
            i += 1
            continue
        j = i
        while j < n and not silent[j]:
            j += 1
        if (j - i) < min_speech_hops:
            silent[i:j] = True
        i = j
    spans: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        start = times[i]
        end = min(duration, times[j - 1] + hop_s)
        if i == 0:
            start = 0.0
        if j >= n:
            end = duration
        if end - start >= float(min_silence_s) - 1e-6:
            spans.append((round(start, 4), round(end, 4)))
        i = j
    return _merge_spans(spans)


def _merge_spans(spans: Sequence[tuple[float, float]], gap: float = 0.04) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    if not ordered:
        return []
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(round(a, 4), round(b, 4)) for a, b in out]


def word_gap_silences(
    words: Sequence[WordSpan],
    duration_s: float,
    min_silence_s: float,
) -> list[tuple[float, float]]:
    if not words:
        return []
    spans: list[tuple[float, float]] = []
    if words[0].start >= min_silence_s:
        spans.append((0.0, float(words[0].start)))
    for left, right in zip(words, words[1:]):
        gap = float(right.start) - float(left.end)
        if gap >= min_silence_s:
            spans.append((float(left.end), float(right.start)))
    tail = float(duration_s) - float(words[-1].end)
    if tail >= min_silence_s:
        spans.append((float(words[-1].end), float(duration_s)))
    return spans


def _punch_words(
    words: Sequence[WordSpan],
    silences: Sequence[tuple[float, float]],
    min_silence_s: float,
) -> list[WordSpan]:
    """Drop the interior of a word only when a real pause sits inside it.

    Whisper sometimes paints a pause as part of the neighbouring word
    (the ringmaster "so" covers 5.52–6.34s, but the pause is ~5.7–6.25s).
    A short overlap is not a license to cut the word.
    """
    atoms: list[WordSpan] = []
    for word in words:
        pieces: list[tuple[float, float]] = [(float(word.start), float(word.end))]
        for s0, s1 in silences:
            overlap = min(s1, word.end) - max(s0, word.start)
            if overlap < float(min_silence_s) - 1e-6:
                continue
            nxt: list[tuple[float, float]] = []
            for a, b in pieces:
                if s1 <= a or s0 >= b:
                    nxt.append((a, b))
                    continue
                if a < s0:
                    nxt.append((a, min(s0, b)))
                if b > s1:
                    nxt.append((max(s1, a), b))
            pieces = [(a, b) for a, b in nxt if b - a >= 0.04]
        if not pieces:
            continue
        longest = max(range(len(pieces)), key=lambda i: pieces[i][1] - pieces[i][0])
        for i, (a, b) in enumerate(pieces):
            atoms.append(WordSpan(text=word.text if i == longest else "", start=a, end=b))
    atoms.sort(key=lambda w: (w.start, w.end))
    return atoms


def _merge_runs(atoms: Sequence[WordSpan], min_silence_s: float) -> list[list[WordSpan]]:
    if not atoms:
        return []
    runs: list[list[WordSpan]] = [[atoms[0]]]
    for atom in atoms[1:]:
        gap = float(atom.start) - float(runs[-1][-1].end)
        if gap < float(min_silence_s):
            runs[-1].append(atom)
        else:
            runs.append([atom])
    return runs


def _split_run(atoms: Sequence[WordSpan], max_s: float) -> list[list[WordSpan]]:
    """Split a speech run on word edges. Never inside an atom.

    An atom longer than ``max_s`` stays whole (cutting it would be mid-word).
    """
    if not atoms:
        return []
    start = float(atoms[0].start)
    end = float(atoms[-1].end)
    if end - start <= float(max_s) + 1e-6:
        return [list(atoms)]
    limit = start + float(max_s)
    split_at: Optional[int] = None
    for i, atom in enumerate(atoms[:-1]):
        if float(atom.end) <= limit + 1e-6:
            split_at = i
    if split_at is None:
        return [list(atoms[:1]), *_split_run(atoms[1:], max_s)]
    head = list(atoms[: split_at + 1])
    return [head, *_split_run(atoms[split_at + 1 :], max_s)]


def _hard_atoms(duration_s: float, max_s: float) -> list[WordSpan]:
    atoms: list[WordSpan] = []
    cursor = 0.0
    step = max(float(max_s), 0.5)
    while cursor < float(duration_s) - 1e-6:
        end = min(float(duration_s), cursor + step)
        atoms.append(WordSpan(text="", start=cursor, end=end))
        cursor = end
    return atoms


def _line_for(words: Sequence[WordSpan], start: float, end: float) -> str:
    bits: list[str] = []
    for word in words:
        if not word.text:
            continue
        mid = (float(word.start) + float(word.end)) / 2.0
        if start - 1e-4 <= mid < end - 1e-6:
            bits.append(word.text)
    return " ".join(bits)


def _word_interior(sec: float, words: Sequence[WordSpan]) -> bool:
    for word in words:
        if not (word.text or "").strip():
            continue
        if word.start + 1e-3 < sec < word.end - 1e-3:
            return True
    return False


def _edge_frame(
    t: float,
    fps: int,
    total: int,
    words: Sequence[WordSpan],
    *,
    prefer: str,
    silences: Sequence[tuple[float, float]] = (),
) -> int:
    """Frame index at ``t`` that is not strictly inside a word.

    Silence ends snap down so the next speech slice keeps the word onset.
    Speech ends snap up so the cut lands in the following gap when there is one.
    A pause is a legal cut even when a word timestamp was smeared across it.
    """

    def inside(frame: int) -> bool:
        sec = frame / float(fps)
        if any(s0 - 1e-3 <= sec <= s1 + 1e-3 for s0, s1 in silences):
            return False
        return _word_interior(sec, words)

    if prefer == "ceil":
        raw = int(math.ceil(float(t) * fps - 1e-9))
    else:
        raw = int(math.floor(float(t) * fps + 1e-9))
    raw = max(0, min(total, raw))
    if not words or not inside(raw):
        return raw
    for delta in range(1, int(fps) + 1):
        order = (raw + delta, raw - delta) if prefer == "ceil" else (raw - delta, raw + delta)
        for cand in order:
            if 0 <= cand <= total and not inside(cand):
                return cand
    return raw


def _allocate_frames(
    chunks: list[tuple[str, float, float, str]],
    duration_s: float,
    fps: int,
    words: Sequence[WordSpan] = (),
    silences: Sequence[tuple[float, float]] = (),
) -> tuple[int, list[tuple[int, int]]]:
    total = timeline_frames(duration_s, fps)
    if not chunks:
        return total, []
    edges = [0]
    for kind, _start, end, _text in chunks[:-1]:
        prefer = "floor" if kind in (KIND_PLATE, KIND_BRIDGE, KIND_IDLE) else "ceil"
        edges.append(
            _edge_frame(float(end), fps, total, words, prefer=prefer, silences=silences)
        )
    edges.append(total)
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
        if edges[i] > total:
            edges[i] = total
    edges[-1] = total
    # Give an empty interior chunk one frame from the following chunk when it has spare.
    for i in range(1, len(edges) - 1):
        if edges[i] == edges[i - 1] and edges[i + 1] - edges[i] >= 2:
            edges[i] += 1
    spans = [(edges[i], edges[i + 1]) for i in range(len(chunks))]
    return total, spans


def _silence_handling(pieces: Sequence[LipdubPiece], *, segmented: bool, silence_mode: str) -> str:
    kinds = {p.kind for p in pieces}
    if silence_mode == SILENCE_IDLE:
        if KIND_IDLE in kinds and KIND_BRIDGE in kinds:
            return "idle_breath+short_bridge"
        if KIND_IDLE in kinds:
            return "idle_breath"
        if KIND_BRIDGE in kinds:
            return "mouth_bridge"
    if KIND_BRIDGE in kinds and (KIND_PLATE in kinds or any(p.hold_frames > MAX_FROZEN_HOLD for p in pieces)):
        return "mouth_bridge+still_hold"
    if KIND_BRIDGE in kinds and KIND_PLATE in kinds:
        return "mouth_bridge+still_hold"
    if KIND_BRIDGE in kinds:
        return "mouth_bridge"
    if KIND_PLATE in kinds:
        return "still_hold"
    if segmented:
        return "prompt_only"
    return "none"


def _continuity_method(pieces: Sequence[LipdubPiece], *, segmented: bool, anchor: str) -> str:
    if not segmented:
        return "none"
    if anchor == ANCHOR_HYBRID:
        return "hybrid_source_still"
    if anchor == ANCHOR_SOURCE:
        return "source_still"
    if any(p.continuity == CONTINUITY_PREV for p in pieces):
        return CONTINUITY_PREV
    return CONTINUITY_STILL


def _generated_before(prev_kind: Optional[str]) -> bool:
    return prev_kind in (KIND_SPEECH, KIND_BRIDGE, KIND_IDLE)


def plan_lipdub(
    duration_s: float,
    *,
    words: Optional[Sequence[WordSpan]] = None,
    silences: Optional[Sequence[tuple[float, float]]] = None,
    pcm=None,
    sample_rate: Optional[int] = None,
    audio_path: Optional[str] = None,
    max_segment_s: Optional[float] = None,
    silence_min_s: Optional[float] = None,
    overlap_frames: Optional[int] = None,
    fps: int = DEFAULT_FPS,
    tripod: bool = False,
    base_prompt: str = "",
    seed: Optional[int] = None,
    silence_mode: Optional[str] = None,
    anchor: Optional[str] = None,
    reframe: Optional[bool] = None,
) -> LipdubPlan:
    """Plan slices. ``segmented`` is false when audio is within the threshold."""
    duration = max(float(duration_s), 0.0)
    max_s = float(lipdub_segment_max_s() if max_segment_s is None else max_segment_s)
    min_sil = float(LIPDUB_SILENCE_MIN_S if silence_min_s is None else silence_min_s)
    overlap = int(LIPDUB_OVERLAP_FRAMES if overlap_frames is None else overlap_frames)
    overlap = max(0, overlap)
    fps_i = max(int(fps), 1)
    mode = _norm_silence_mode(silence_mode)
    anc = _norm_anchor(anchor)
    will_segment = duration > max_s + 1e-3
    do_reframe = will_segment if reframe is None else bool(reframe)
    warning: Optional[str] = None
    plan = LipdubPlan(
        segmented=False,
        duration_s=duration,
        fps=fps_i,
        max_segment_s=max_s,
        silence_min_s=min_sil,
        overlap_frames=overlap,
        tripod=bool(tripod),
        timeline_frame_count=timeline_frames(duration, fps_i) if duration > 0 else 0,
        full_audio_mux=True,
        silence_mode=mode,
        anchor=anc,
        reframe=do_reframe,
        join_method=JOIN_NONE,
    )
    if duration <= max_s + 1e-3:
        if duration > (OBSERVED_CLIFF_FRAMES / float(fps_i)) + 0.02:
            plan.warning = (
                f"audio {duration:.2f}s is under --lipdub-max-s ({max_s:.2f}s) but past "
                f"the observed lip-sync cliff (~{OBSERVED_CLIFF_FRAMES / fps_i:.2f}s, "
                f"{OBSERVED_CLIFF_FRAMES} frames). One pass will be used because the "
                "threshold was raised."
            )
        return plan

    energy: list[tuple[float, float]] = list(silences or [])
    if pcm is not None and sample_rate:
        energy.extend(
            energy_silence_spans(pcm, int(sample_rate), duration, min_silence_s=min_sil)
        )
    elif audio_path and silences is None and pcm is None:
        try:
            from master_agent.music.beats import decode_audio

            decoded, sr = decode_audio(audio_path, sr=16000)
            energy.extend(energy_silence_spans(decoded, sr, duration, min_silence_s=min_sil))
        except Exception as exc:
            warning = f"energy VAD skipped ({exc}); splitting on word gaps or the max length"

    word_list = list(words or [])
    gap_sil = word_gap_silences(word_list, duration, min_sil)
    sil_all = _merge_spans([*energy, *gap_sil])
    sil_all = [(max(0.0, a), min(duration, b)) for a, b in sil_all if b - a >= min_sil - 1e-6]
    sil_all = _merge_spans(sil_all)

    if word_list:
        atoms = _punch_words(word_list, sil_all, min_sil)
    elif sil_all:
        cursor = 0.0
        atoms = []
        for s0, s1 in sil_all:
            if s0 - cursor >= 0.04:
                atoms.append(WordSpan(text="", start=cursor, end=s0))
            cursor = max(cursor, s1)
        if duration - cursor >= 0.04:
            atoms.append(WordSpan(text="", start=cursor, end=duration))
    else:
        atoms = _hard_atoms(duration, max_s)
        if warning is None and not word_list:
            warning = (
                "no word timestamps or silence spans; splitting on the max length only"
            )

    runs = _merge_runs(atoms, min_sil)
    speech_chunks: list[tuple[float, float, str]] = []
    overflow = False
    for run in runs:
        for chunk in _split_run(run, max_s):
            c0 = float(chunk[0].start)
            c1 = float(chunk[-1].end)
            text = " ".join(a.text for a in chunk if a.text).strip()
            if c1 - c0 > max_s + 0.05:
                overflow = True
            speech_chunks.append((c0, c1, text))

    # Absorb sub-threshold gaps into the following speech so frames are not dropped.
    # Real pauses stay plates, bridges, or idle renders.
    def pause_kind(span_s: float, *, leading: bool) -> str:
        if mode == SILENCE_HOLD:
            return KIND_PLATE if leading else KIND_BRIDGE
        if mode == SILENCE_BRIDGE:
            return KIND_BRIDGE
        frames = int(round(float(span_s) * fps_i))
        if frames >= BRIDGE_FRAMES:
            return KIND_IDLE
        return KIND_BRIDGE

    timed: list[tuple[str, float, float, str]] = []
    cursor = 0.0
    for c0, c1, text in speech_chunks:
        if c0 - cursor >= min_sil - 1e-6:
            leading = not any(item[0] == KIND_SPEECH for item in timed)
            timed.append((pause_kind(c0 - cursor, leading=leading), cursor, c0, ""))
        elif c0 > cursor:
            c0 = cursor
        timed.append((KIND_SPEECH, c0, c1, text))
        cursor = c1
    if duration - cursor >= min_sil - 1e-6:
        leading = not any(item[0] == KIND_SPEECH for item in timed)
        timed.append((pause_kind(duration - cursor, leading=leading), cursor, duration, ""))
    elif cursor < duration and timed:
        kind, s0, _e, text = timed[-1]
        timed[-1] = (kind, s0, duration, text)

    if not timed:
        timed = [(pause_kind(duration, leading=True), 0.0, duration, "")]

    total, spans = _allocate_frames(timed, duration, fps_i, word_list, sil_all)
    pieces: list[LipdubPiece] = []
    for (kind, _s, _e, text), (f0, f1) in zip(timed, spans):
        if f1 <= f0:
            continue
        pieces.append(
            LipdubPiece(
                index=len(pieces),
                kind=kind,
                start_s=f0 / float(fps_i),
                end_s=f1 / float(fps_i),
                place_start=f0,
                place_end=f1,
                spoken_line=text,
            )
        )

    if mode == SILENCE_IDLE:
        for piece in pieces:
            if piece.kind == KIND_BRIDGE and piece.keep_frames >= BRIDGE_FRAMES:
                piece.kind = KIND_IDLE
            elif piece.kind == KIND_PLATE and piece.keep_frames >= BRIDGE_FRAMES:
                piece.kind = KIND_IDLE
            elif piece.kind == KIND_PLATE:
                piece.kind = KIND_BRIDGE

    prev_kind: Optional[str] = None
    negative = _segment_negative(tripod=tripod)
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
    if overflow:
        warnings.append(
            "a word is longer than --lipdub-max-s; that word was kept whole (not cut mid-word)"
        )
    for piece in pieces:
        piece.seed = seed
        before = _generated_before(prev_kind)
        _assign_continuity(piece, before=before, anchor=anc)
        drop = _overlap_drop(piece, prev_kind, anchor=anc, overlap=overlap)
        piece.drop_leading = drop
        piece.crossfade_frames = drop if anc != ANCHOR_PREVIOUS and before else 0
        if piece.kind == KIND_SPEECH:
            _finalize_speech(
                piece,
                drop=drop,
                fps=fps_i,
                words=word_list,
                base_prompt=base_prompt,
                negative=negative,
                tripod=tripod,
                anchor=anc,
                warnings=warnings,
            )
        elif piece.kind == KIND_IDLE:
            _finalize_idle(
                piece,
                drop=drop,
                fps=fps_i,
                tripod=tripod,
                anchor=anc,
                warnings=warnings,
            )
        elif piece.kind == KIND_BRIDGE:
            _finalize_bridge(
                piece,
                drop=drop,
                fps=fps_i,
                tripod=tripod,
                negative=negative,
                allow_hold=mode != SILENCE_IDLE,
            )
        else:
            piece.continuity = CONTINUITY_STILL
            piece.hold_frames = piece.keep_frames
            piece.generated_keep = 0
            piece.render_frames = 0
            piece.drop_leading = 0
            piece.crossfade_frames = 0
            piece.prompt = ""
            piece.negative = ""
            piece.audio_feed = "none"
        prev_kind = piece.kind

    for i, piece in enumerate(pieces):
        piece.index = i
    plan.segmented = True
    plan.timeline_frame_count = total
    plan.pieces = pieces
    plan.split_points_s = [round(p.end_s, 4) for p in pieces[:-1]]
    plan.continuity_method = _continuity_method(pieces, segmented=True, anchor=anc)
    plan.silence_handling = _silence_handling(pieces, segmented=True, silence_mode=mode)
    if anc == ANCHOR_PREVIOUS:
        plan.join_method = JOIN_TRIM if any(p.drop_leading for p in pieces) else JOIN_NONE
    elif any(p.crossfade_frames for p in pieces):
        plan.join_method = JOIN_CROSSFADE
    else:
        plan.join_method = JOIN_NONE
    plan.warning = "; ".join(warnings) if warnings else None
    return plan


def _assign_continuity(piece: LipdubPiece, *, before: bool, anchor: str) -> None:
    if piece.kind == KIND_PLATE or not before:
        piece.continuity = CONTINUITY_STILL
        return
    if anchor == ANCHOR_SOURCE:
        piece.continuity = CONTINUITY_SOURCE
    elif anchor == ANCHOR_HYBRID:
        piece.continuity = CONTINUITY_HYBRID
    else:
        piece.continuity = CONTINUITY_PREV


def _overlap_drop(
    piece: LipdubPiece,
    prev_kind: Optional[str],
    *,
    anchor: str,
    overlap: int,
) -> int:
    if piece.kind == KIND_PLATE or not _generated_before(prev_kind):
        return 0
    if anchor == ANCHOR_PREVIOUS:
        # #34: overlap context only on a speech-to-speech seam. A bridge already
        # starts on the previous frame, so it is not rendered twice.
        if piece.kind == KIND_SPEECH and prev_kind == KIND_SPEECH:
            return min(overlap, piece.place_start)
        return 0
    return min(int(overlap), piece.place_start)


def _finalize_speech(
    piece: LipdubPiece,
    *,
    drop: int,
    fps: int,
    words: Sequence[WordSpan],
    base_prompt: str,
    negative: str,
    tripod: bool,
    anchor: str,
    warnings: list[str],
) -> None:
    piece.generated_keep = piece.keep_frames
    piece.hold_frames = 0
    piece.render_frames = snap_ltx_frames_at_least(piece.keep_frames + drop)
    piece.audio_start_s = (piece.place_start - drop) / float(fps)
    piece.audio_duration_s = piece.render_frames / float(fps)
    piece.audio_feed = "source_wav_slice"
    spoken = piece.spoken_line or _line_for(words, piece.start_s, piece.end_s)
    piece.spoken_line = spoken
    piece.prompt = _speech_prompt(base_prompt, spoken, tripod=tripod, anchor=anchor)
    piece.negative = negative
    piece.i2v_strength = TRIPOD_I2V_STRENGTH if tripod else None
    if piece.render_frames > SAFE_RENDER_FRAMES:
        warnings.append(
            f"speech slice {piece.index} renders {piece.render_frames} frames, "
            f"past the safe cap {SAFE_RENDER_FRAMES} (cliff {OBSERVED_CLIFF_FRAMES})"
        )


def _finalize_idle(
    piece: LipdubPiece,
    *,
    drop: int,
    fps: int,
    tripod: bool,
    anchor: str,
    warnings: list[str],
) -> None:
    piece.generated_keep = piece.keep_frames
    piece.hold_frames = 0
    piece.render_frames = snap_ltx_frames_at_least(piece.keep_frames + drop)
    piece.audio_start_s = (piece.place_start - drop) / float(fps)
    piece.audio_duration_s = piece.render_frames / float(fps)
    # The graph trims the original wav. A pause slice is silence or room tone.
    piece.audio_feed = "source_pause_slice"
    piece.prompt = _idle_prompt(tripod=tripod, anchor=anchor)
    piece.negative = _idle_negative(tripod=tripod)
    piece.i2v_strength = IDLE_I2V_STRENGTH
    if piece.render_frames > SAFE_RENDER_FRAMES:
        warnings.append(
            f"idle slice {piece.index} renders {piece.render_frames} frames, "
            f"past the safe cap {SAFE_RENDER_FRAMES}"
        )


def _finalize_bridge(
    piece: LipdubPiece,
    *,
    drop: int,
    fps: int,
    tripod: bool,
    negative: str,
    allow_hold: bool,
) -> None:
    piece.i2v_strength = BRIDGE_I2V_STRENGTH
    piece.prompt = _bridge_prompt(tripod=tripod)
    piece.negative = negative
    piece.audio_feed = "source_pause_slice"
    keep = piece.keep_frames
    if allow_hold and keep > BRIDGE_FRAMES:
        piece.generated_keep = BRIDGE_FRAMES
        piece.hold_frames = keep - BRIDGE_FRAMES
        piece.render_frames = snap_ltx_frames_at_least(BRIDGE_FRAMES + drop)
        if piece.render_frames < drop + piece.generated_keep:
            piece.render_frames = snap_ltx_frames_at_least(drop + piece.generated_keep)
    else:
        piece.generated_keep = keep
        piece.hold_frames = 0
        piece.render_frames = snap_ltx_frames_at_least(keep + drop)
    piece.audio_start_s = (piece.place_start - drop) / float(fps)
    piece.audio_duration_s = piece.render_frames / float(fps)


def alignment_errors(plan: LipdubPlan) -> list[str]:
    """Frame/audio invariants. Empty when the timeline matches the wav."""
    errors: list[str] = []
    if not plan.segmented:
        return errors
    cursor = 0
    for piece in plan.pieces:
        if piece.place_start != cursor:
            errors.append(
                f"piece {piece.index} starts at frame {piece.place_start}, expected {cursor}"
            )
        if piece.place_end <= piece.place_start:
            errors.append(f"piece {piece.index} has an empty frame span")
        if piece.generated_keep + piece.hold_frames != piece.keep_frames:
            errors.append(
                f"piece {piece.index} keep {piece.keep_frames} != "
                f"generated {piece.generated_keep} + hold {piece.hold_frames}"
            )
        if plan.silence_mode == SILENCE_IDLE and piece.hold_frames > MAX_FROZEN_HOLD:
            errors.append(
                f"piece {piece.index} holds {piece.hold_frames} frames; "
                f"idle mode allows at most {MAX_FROZEN_HOLD}"
            )
        if piece.kind in (KIND_SPEECH, KIND_IDLE):
            if piece.render_frames < piece.drop_leading + piece.generated_keep:
                errors.append(f"piece {piece.index} render is shorter than the frames it keeps")
            if (piece.render_frames - 1) % 8 != 0 or piece.render_frames < 9:
                errors.append(f"piece {piece.index} render_frames {piece.render_frames} is not 8n+1")
            expect_start = (piece.place_start - piece.drop_leading) / float(plan.fps)
            if abs(piece.audio_start_s - expect_start) > 1e-6:
                errors.append(f"piece {piece.index} audio_start drifted from the frame grid")
        if piece.kind == KIND_BRIDGE and piece.render_frames:
            if piece.render_frames < piece.drop_leading + piece.generated_keep:
                errors.append(f"bridge {piece.index} render is shorter than the frames it keeps")
            if (piece.render_frames - 1) % 8 != 0 or piece.render_frames < 9:
                errors.append(f"bridge {piece.index} render_frames {piece.render_frames} is not 8n+1")
            expect_start = (piece.place_start - piece.drop_leading) / float(plan.fps)
            if abs(piece.audio_start_s - expect_start) > 1e-6:
                errors.append(f"bridge {piece.index} audio_start drifted from the frame grid")
        cursor = piece.place_end
    if cursor != plan.timeline_frame_count:
        errors.append(
            f"placed frames {cursor} != timeline {plan.timeline_frame_count}"
        )
    expect = timeline_frames(plan.duration_s, plan.fps)
    if plan.timeline_frame_count != expect:
        errors.append(f"timeline {plan.timeline_frame_count} != round(duration*fps) {expect}")
    return errors


def cuts_inside_words(
    plan: LipdubPlan,
    words: Sequence[WordSpan],
    *,
    allowed_silences: Sequence[tuple[float, float]] = (),
) -> list[str]:
    """Speech boundaries that land inside a word and not inside a known pause."""
    bad: list[str] = []
    if not plan.segmented:
        return bad
    boundaries = [p.end_s for p in plan.pieces[:-1]]
    for edge in boundaries:
        for word in words:
            if word.start + 1e-3 < edge < word.end - 1e-3:
                if any(s0 - 1e-3 <= edge <= s1 + 1e-3 for s0, s1 in allowed_silences):
                    continue
                bad.append(
                    f"cut at {edge:.3f}s is inside {word.text!r} ({word.start:.3f}-{word.end:.3f})"
                )
    return bad


def format_lipdub_plan(plan: LipdubPlan) -> str:
    if not plan.segmented:
        lines = [
            f"lipdub: single pass {plan.duration_s:.3f}s "
            f"(threshold {plan.max_segment_s:.3f}s) — same render as a short lipdub"
        ]
        if plan.tripod:
            lines.append("tripod: on (prompt, negative, first-frame strength)")
        else:
            lines.append("tripod: off")
        lines.append(
            f"silence_mode: {plan.silence_mode} (single pass, idle pieces not used)"
        )
        lines.append(f"anchor: {plan.anchor} (single pass, not applied)")
        lines.append(f"reframe: {'on' if plan.reframe else 'off'}")
        lines.append("idle pieces: 0")
        if plan.warning:
            lines.append(f"warn: {plan.warning}")
        return "\n".join(lines)
    lines = [
        f"lipdub: {plan.duration_s:.3f}s -> {len(plan.pieces)} piece(s), "
        f"{len(plan.speech_pieces())} speech render(s), "
        f"{len(plan.idle_pieces())} idle render(s), "
        f"timeline {plan.timeline_frame_count} frames @ {plan.fps}fps, "
        f"max {plan.max_segment_s:.3f}s",
        f"continuity: {plan.continuity_method} overlap_frames={plan.overlap_frames}",
        f"silence: {plan.silence_handling} mode={plan.silence_mode} min={plan.silence_min_s:.3f}s",
        f"anchor: {plan.anchor} join={plan.join_method}",
        f"reframe: {'on' if plan.reframe else 'off'}",
        f"idle pieces: {len(plan.idle_pieces())} comfy_jobs: {len(plan.comfy_pieces())}",
        f"tripod: {'on' if plan.tripod else 'off'}",
        "mux: original full audio (untouched wav, aac 192k)",
    ]
    if plan.split_points_s:
        lines.append("splits_s: " + ", ".join(f"{s:.3f}" for s in plan.split_points_s))
    for piece in plan.pieces:
        lines.append(
            f"  [{piece.index}] {piece.kind} {piece.start_s:.3f}-{piece.end_s:.3f}s "
            f"frames {piece.place_start}-{piece.place_end - 1} "
            f"render={piece.render_frames} keep={piece.keep_frames} "
            f"drop_lead={piece.drop_leading} hold={piece.hold_frames} "
            f"source={piece.continuity} line={piece.spoken_line!r}"
        )
    if plan.warning:
        lines.append(f"warn: {plan.warning}")
    return "\n".join(lines)


def _hybrid_weight(plan: LipdubPlan) -> Optional[float]:
    if plan.anchor != ANCHOR_HYBRID or not plan.segmented:
        return None
    from master_agent.orchestrator.lipdub_reframe import HYBRID_PREV_WEIGHT

    return float(HYBRID_PREV_WEIGHT)


def lipdub_param_block(
    plan: LipdubPlan,
    *,
    audio_sha256: Optional[str] = None,
    segments: Optional[list[dict]] = None,
) -> dict:
    """Optional ``params.lipdub`` on ``buddy.clip.provenance/v1``.

    Same pattern as ``params.voice_sample``: extra key, same schema id.
    Rust ``ClipProvenance`` does not have this field yet — see the PR note.
    """
    records = segments if segments is not None else [p.to_record() for p in plan.pieces]
    return {
        "segmented": bool(plan.segmented),
        "max_segment_s": plan.max_segment_s,
        "silence_min_s": plan.silence_min_s,
        "fps": plan.fps,
        "timeline_frames": plan.timeline_frame_count,
        "split_points_s": list(plan.split_points_s),
        "continuity": plan.continuity_method,
        "overlap_frames": plan.overlap_frames,
        "silence_handling": plan.silence_handling,
        "silence_mode": plan.silence_mode,
        "anchor": plan.anchor,
        "reframe": bool(plan.reframe),
        "join": plan.join_method,
        "scale_drift": list(plan.scale_drift),
        "hybrid_prev_weight": _hybrid_weight(plan),
        "tripod": bool(plan.tripod),
        "tripod_i2v_strength": TRIPOD_I2V_STRENGTH if plan.tripod else None,
        "full_audio_mux": True,
        "audio_codec": "aac",
        "audio_sha256": audio_sha256,
        "segments": records,
    }


def attach_lipdub_params(payload: dict, block: dict) -> dict:
    params = payload.setdefault("params", {})
    if not isinstance(params, dict):
        params = {}
        payload["params"] = params
    params["lipdub"] = block
    return payload
