# Self-improvement loop (live Python orchestrator)

The intended loop lives in this repo’s `master_agent` orchestrator — not in
sibling `your-video-buddy`. That sibling owns the Rust iteration controller /
judge scaffold; this tree owns the **live** judge → revise → re-run → re-judge
path on the tower.

## Intended loop

1. Run / generate an asset (or dry-run with context only)
2. Judge: heuristics + text LLM + optional vision
3. On FAIL → structured revise plan (prompt deltas + param deltas, and/or
   attach / control / shot-plan usage) from fail reasons
4. Re-run generation with revised inputs (skipped under `--self-improve-dry`)
5. Re-judge
6. Stop on PASS or max attempts / budget

Terminal `loop_status` in the run JSON is one of: `passed`, `exhausted`,
`human_veto`, `error`. `DONE` is the machine state; exhaustion is **not**
recorded as `accept`.

## What was already wired

| Piece | Where | Notes |
|---|---|---|
| Director variant routing | `orchestrator/director.py` | LLM + keyword rules; attach preference |
| Multi-segment pipeline | `orchestrator/pipeline.py` | Storyboard → per-seg `Orchestrator.run` → stitch → full-video judge → weak-shot regen |
| State machine | `orchestrator/machine.py` | PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE |
| OOM downscale | `machine._handle_job_error` | Walks `DOWNSCALE_LADDER` / `vram_policy.downscale_ladder_for` |
| LoRA retry ladder | `lora/validate.py` | Separate from clip judge |
| Judge legs | `judge/judge.py` | Heuristic + text LLM + optional vision; look vs health split |
| LLM rewrite / retune | `machine._judge` (old) | Only if the LLM returned `prompt_rewrite` / `param_hints` |
| Full-video weak-shot regen | `pipeline.run_pipeline` | Outer loop; not quality-bar aware |
| Attach bookkeeping (c)/(d) | `comfy/attach.py` | `previs_source` + `control_pack_used` on attach records |
| `--dry-run` | `pipeline.dry_run_pipeline` | Plan + lint only (curriculum L3). Does **not** close the judge loop |
| Shift budget | `control/budget.py` | HOLD is not a judge fail |

## What was broken / missing (now closed)

| Gap | Fix |
|---|---|
| Quality Bar rules a/c/d never consulted | `judge/quality_bar.py` — cheap fails, no GPU, no face scores |
| Judge fail with no LLM rewrite stopped the loop (“no actionable feedback”) | Fail reasons always produce a `RevisePlan` (prompt + param deltas) |
| still→I2V with no shot plan did not fail | Rule **c** (`thin_still_i2v`) |
| Control pack present but unused did not fail / revise | Rule **d** (`unused_control_pack`) — same id as attach bookkeeping |
| Missing music bed on an MV brief did not fail | Rule **a** (`missing_music_bed`) |
| Rule **b** face similarity | **Skipped** — not ported, no invented scores |
| Max-retry looked like `accept` | `loop_status=exhausted`, `judge_decision=exhausted` |
| No dry-run closed loop for CI | `Orchestrator.run(dry_run=True)` and `--self-improve-dry` |
| Prompts / provenance not stored with clips | `master_agent/provenance.py` — sidecar + run-row ClipProvenance |

## Quality Bar rule ids (buddy-core strings)

These are plain strings / JSON — not a parallel iteration-controller API.

| id | code | Cheap? | Behavior here |
|---|---|---|---|
| `a` | `missing_music_bed` | yes | MV intent (`music video` / `song` / `music bed` / `kind=music_video`) and no bed (no `audio_name` / `audio_path` / muxed audio / `music_bed_attached`) |
| `b` | `face_similarity` | no (vision) | **Not evaluated.** Payload lists it under `skipped` |
| `c` | `thin_still_i2v` | yes | Still→I2V (`--image` / i2v wording) with no shot plan (camera/action/visuals/continuity). Attach records still use (c) as “previs_source recorded” |
| `d` | `unused_control_pack` | yes | `control_pack_present` and no used channel (`openpose` / `depth` / `edges` / `camera`) |

## How to invoke

Closed loop, no Comfy / no GPU (CI and local):

```bash
cd video_buddy
python -m master_agent run "music video for a synthwave track" \
  --self-improve-dry --no-interview --max-judge-rounds 3
python -m master_agent run "i2v from this still, neon alley" \
  --image still.png --self-improve-dry --no-interview
python -m master_agent run "rain on a window" \
  --attach path/to/export-patch.json --self-improve-dry --no-interview
```

Tests (no GPU, no live Comfy):

```bash
cd video_buddy
python -m pytest tests/test_self_improvement_loop.py tests/test_clip_provenance.py tests/test_judge_split.py tests/test_comfy_attach.py -q
```

Live (Comfy on `:8188`) — same `run` path; after a quality-bar / judge fail the
orchestrator applies the revise plan, re-patches, re-queues, and re-judges
until pass or `max_judge_rounds` (`--max-judge-rounds` / `MAX_JUDGE_ROUNDS`).

`--dry-run` stays plan + lint only (L3). It does not run the closed loop.

## Run JSON

`state/runs/<ts>_<id>.json` now includes:

- `loop_status` — `passed` / `exhausted` / `human_veto` / `error`
- `attempt` / `max_judge_rounds`
- `quality_bar` — `{evaluated, skipped, fails, pass}`
- `revise_history` — per-attempt prompt/param deltas
- `provenance` / `provenance_history` / `provenance_sidecar` — ClipProvenance
  (`buddy.clip.provenance/v1`) keyed by `output_path` / `hash`. Same object
  is written next to the planned clip as `{output_dir}/shot-N.buddy.json`.
  Sidecar is read before each revise. See [CLIP_PROVENANCE.md](CLIP_PROVENANCE.md).

## Remaining holes

- Rule **b** (face similarity) stays unported — no invented scores.
- Live mux of a music bed still happens in `python -m master_agent music`.
  The orchestrator dry-run treats a revise plan as “bed attached”; a live
  `run "music video…"` without `--audio` will exhaust on rule **a** unless a
  track is provided (auto-route then muxes).
- Vision / text-LLM legs are unchanged and optional. Cheap rules do not need them.
- Kling / Partner Node is out of scope.
- `--dry-run` (curriculum L3) is still plan/lint only.
