# Clip provenance

Every clip stores schema `buddy.clip.provenance/v1`.

- Sidecar: `{output_dir}/shot-N.buddy.json` next to the planned `shot-N.mp4`
- Run row: the same object on `state/runs/<ts>_<id>.json`, keyed by `output_path` / `hash`

The sidecar is the source of truth across attempts. The orchestrator reads it before a revise. Do not write `<stem>.provenance.json` for new clips. A read-only fallback still opens a legacy `.provenance.json` when the `.buddy.json` is missing. The public H3 still uses that older name: [`docs/demo/H3-SHOWCASE-BMX-8s.provenance.json`](demo/H3-SHOWCASE-BMX-8s.provenance.json).

Field names match the buddy-core ClipProvenance shape. If a sibling struct adds keys, add them here without dropping these and without a second schema id.

## Required shape

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
      {"kind": "cpu_fail_rules", "id": "c", "code": "thin_still_i2v", "detail": "still→I2V with no shot plan"}
    ]
  },
  "revise_notes": "quality_bar.c: plan the still→I2V shot",
  "created_at": "2026-09-18T04:59:00Z",
  "output_path": "outputs/<run_id>/shot-1.mp4",
  "hash": null
}
```

`hash` is SHA-256 only when the clip file exists. Otherwise `null`. Never invent it.

Optional blocks stay on this schema id and are omitted when empty:

- `prompts.spoken_line` and `params.voice_sample` for an H3 sample (`original_duration_s`, `start_s`, `end_s`, `method`, `trimmed`)
- `params.lipdub` for a stitched long `ltx25_a2v` clip. Flag names and defaults: [Features](FEATURES.md#lipdub)
- `params.heartmula` for a generate or transcribe block. Fields: [Features](FEATURES.md#heartmula)

Paths in anything you commit are repo-relative or role-relative (`gguf/…`, `loras/…`). No absolute paths, API keys, or `.env` values. The same rule is in [`knowledge/AGENTS.md`](../knowledge/AGENTS.md).

## When it is written

| Event | Action |
|---|---|
| Run start | Plan `shot-N.mp4` and write the sidecar (`hash` null) |
| Generate | Copy the Comfy output onto that path and store the hash |
| Judge | Update `judge.score` and `fail_reasons` on the same sidecar |
| Before revise | Read the sidecar and restore prompts, params, and parent lineage |
| After revise | Increment the attempt. Record `revise_notes` and parent ids |
| `--self-improve-dry` | Same read and write. Hash stays null if there is no file |
| Stitch or music mux | `inherit_clip_provenance` onto the derived `<stem>.buddy.json` |

The sidecar name stays `shot-N.buddy.json` across retries. History lives on the run row (`provenance_history`).

Code: `master_agent/provenance.py`. Tests: `python -m pytest tests/test_clip_provenance.py -q`.
