# ClipProvenance (`buddy.clip.provenance/v1`)

Prompts and full provenance are stored with **every** video clip.

- Sidecar: `{output_dir}/shot-N.buddy.json` next to planned `shot-N.mp4`
- Run row: the same object on the orchestrator / pipeline JSON, keyed by
  `output_path` / `hash` (`state/runs/<ts>_<id>.json`)

The live Python orchestrator reads and writes this contract on every
generate and revise. **The sidecar is the source of truth across attempts**
— it is read before a revise is applied.

## Wire compatibility

Schema id and field names match buddy-core **ClipProvenance** from sibling
`your-video-buddy` PR #7 (`cursor/close-improve-loop-1c25`, agent
`bc-39743643`). That repo / PR was **not fetchable** from this environment
(404). This is **not** a second schema — it is the PR #7 shape Scott
listed. If the Rust struct lands extra keys, add them without dropping
these.

Do not write `<stem>.provenance.json` for new clips. A read-only fallback
still opens a legacy `.provenance.json` if a `.buddy.json` is missing.
The public H3 BMX still ships the original showcase sidecar as
[`docs/demo/H3-SHOWCASE-BMX-8s.provenance.json`](../../docs/demo/H3-SHOWCASE-BMX-8s.provenance.json).

## Required fields (every attempt)

```json
{
  "schema": "buddy.clip.provenance/v1",
  "prompts": {
    "brief": "music video for a synthwave track",
    "positive": "neon rain alley, handheld",
    "negative": "blur, text",
    "additives": ["slow camera push-in"]
  },
  "engine": {
    "backend": "comfy",
    "workflow_id": "ltx25_t2v_i2v",
    "variant": "ltx25_t2v_i2v"
  },
  "params": {
    "seed": 42,
    "steps": 8,
    "cfg": 3.5,
    "size": [768, 512],
    "fps": 24,
    "duration": 3.0,
    "refs": ["still.png"]
  },
  "lineage": {
    "shot_id": "shot-1",
    "attempt_id": "shot-1.a2",
    "iteration": 2,
    "parent_shot_id": "shot-1",
    "parent_attempt_id": "shot-1.a1"
  },
  "judge": {
    "score": 0.81,
    "fail_reasons": [
      {
        "kind": "cpu_fail_rules",
        "id": "c",
        "code": "thin_still_i2v",
        "detail": "still→I2V with no shot plan"
      }
    ]
  },
  "revise_notes": "quality_bar.c: plan the still→I2V shot; prompt: slow push-in",
  "created_at": "2026-09-18T04:59:00Z",
  "output_path": "outputs/<run_id>/shot-1.mp4",
  "hash": null
}
```

| Group | Fields |
|---|---|
| Prompts | `brief` / `positive` / `negative` / `additives` |
| Engine | `backend=comfy` + `workflow_id` / `variant` |
| Params | `seed`, `steps`, `cfg`, `size` `[w,h]`, `fps`, `duration`, `refs` |
| Lineage | `shot_id`, `attempt_id`, `iteration`, `parent_shot_id`, `parent_attempt_id` |
| Judge | `score` (honest) + `fail_reasons` (`kind=cpu_fail_rules` for quality-bar a/c/d) |
| Other | `revise_notes` that produced this attempt, `created_at`, `output_path`, `hash` |

`hash` is SHA-256 **only if the clip file exists**. Otherwise `null` — never
invented.

## When it is written / read

| Event | Action |
|---|---|
| Run start | Plan `{output_dir}/shot-N.mp4` and write sidecar (hash null until generate) |
| Generate (`RESOLVE`) | Copy Comfy output onto planned `shot-N.mp4`; persist sidecar + hash |
| Judge | Persist updated `judge.score` / `fail_reasons` onto the same sidecar |
| Before revise | **Read sidecar** and restore prompts/params/parent lineage |
| After revise | Increment attempt; persist sidecar with `revise_notes` + parent ids |
| Re-run | Overwrite `shot-N.mp4`; persist new hash |
| `--self-improve-dry` | Same read/write; hash stays null if no file |
| Stitch / music mux | `inherit_clip_provenance` onto the derived file's `<stem>.buddy.json` |

Sidecar dest name stays `shot-N.buddy.json` across retries. History lives on
the run row (`provenance_history`).

## Code

- `master_agent/provenance.py` — build / read / write / inherit / apply
- `orchestrator/machine.py` — plan paths; persist on start/resolve/judge/revise; read before revise
- `orchestrator/pipeline.py` — `shot_index=i+1`; inherit on stitch
- `music/pipeline.py` — `shot_index=i+1`; inherit on mux

Tests (no GPU): `python -m pytest tests/test_clip_provenance.py -q`
