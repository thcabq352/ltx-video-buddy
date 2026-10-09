# Judge

The judge is a coherent-take retry ladder. It is not an album curator. Admiral or human eyes beat it for aesthetic album lock. `album_lock` from the judge is always false.

Code: `master_agent/judge/judge.py`, `master_agent/judge/quality_bar.py`, `orchestrator/machine.py`.

## Legs

Three legs merge when they are available. Weights `JUDGE_HEURISTIC_WEIGHT`, `JUDGE_LLM_WEIGHT`, `JUDGE_VISION_WEIGHT` renormalize over the legs that ran.

| Leg | What it sees |
|---|---|
| Heuristics | File size, duration vs request, frame-diff motion via ffmpeg |
| Text LLM | Verdict on the brief |
| Vision | `qwen3-vl-heretic` on extracted frames (temporal consistency, subject lock, artifacts). `VISION_ENABLED=0` disables it |

Payload fields: `look_score` (craft and motion), `health_score` (probe: size, frames, duration), `combined_score` (back-compat). The retry ladder uses **look only**. Junk under 100KB or under 3 frames is a health fail even when the tracker says `DONE`. Low `brief_adherence` with a high look score is `human_veto`, not an automatic rewrite.

Below threshold the judge may rewrite the prompt or retune a whitelist: `steps`, `cfg`, `stg_scale`, `stg_blocks`, `sampler_name`, `seed`. Values are clamped. Budget is `--max-judge-rounds` / `MAX_JUDGE_ROUNDS`.

`--no-judge` is one render: no judge and no quality-bar revise.

An opt-in Rainey1 rubric (`PANEL_JUDGE_RUBRIC=rainey1`) adds look dimensions and hard-fail gates on the same payload: [Rainey1](RAINEY1.md#rubric).

## Quality bar

Cheap rules. No GPU. No invented face scores. A fail always produces a `RevisePlan` even when the LLM is silent.

| id | code | Behavior |
|---|---|---|
| `a` | `missing_music_bed` | Music-video intent and no bed (`audio_name`, `audio_path`, muxed audio, or `music_bed_attached`) |
| `b` | `face_similarity` | Not evaluated. Listed under `skipped` |
| `c` | `thin_still_i2v` | Still→I2V with no shot plan (camera, action, visuals, or continuity) |
| `d` | `unused_control_pack` | A control pack is present and no channel was used |

Attach bookkeeping uses the same c/d ids. See [Comfy](COMFY.md#attach-contract).

## Loop

1. Generate, or `--self-improve-dry` with context only.
2. Judge.
3. On fail, apply the revise plan (prompt deltas, param deltas, attach or shot-plan usage).
4. Re-run unless `--self-improve-dry`.
5. Re-judge.
6. Stop on pass or on the attempt cap.

Terminal `loop_status` on the run JSON is `passed`, `exhausted`, `human_veto`, or `error`. Machine state `DONE` is not the same as `accept`. Exhaustion is not recorded as `accept`.

`--dry-run` is curriculum L3: plan and lint only. It does not close this loop. `--self-improve-dry` does close it, with no Comfy queue.

```bash
cd video_buddy
python -m master_agent run "music video for a synthwave track" \
  --self-improve-dry --no-interview --max-judge-rounds 3
```

The sidecar is read before a revise is applied. See [Provenance](PROVENANCE.md).

Live mux of a music bed still happens in `music` / `mv`. A live `run "music video…"` without audio exhausts on rule `a` unless a track is provided. Kling and Partner graphs are out of scope. Pack C renders on local Comfy only: [Features](FEATURES.md#pack-c-seedance).

Tests (no GPU): `python -m pytest tests/test_self_improvement_loop.py tests/test_judge_split.py tests/test_clip_provenance.py -q`.
