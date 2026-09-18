# ClipProvenance (`buddy.clip.provenance/v1`)

Prompts and full provenance are stored with **every** video clip.

- Sidecar: `<clip_stem>.provenance.json` next to the file
- Run row: the same object on the orchestrator / pipeline JSON, keyed by
  `path` / `hash` (`state/runs/<ts>_<id>.json`)

The live Python orchestrator on this repo reads and writes that contract on
every generate, revise, and re-run.

## Wire compatibility

Field names match buddy-core **ClipProvenance** (Rust sibling
`your-video-buddy` / agent `bc-39743643`). That schema was **not fetchable**
from this environment (sibling repo 404; agent not in this workspace). The
shape below is the minimal compatible contract. **Do not rename these
fields.** If the Rust struct lands extra keys, add them here without
dropping these.

Schema token: `buddy.clip.provenance/v1` (same family as
`buddy.comfy.attach/v1`).

## Required fields

| Field | Type | Meaning |
|---|---|---|
| `prompt` | string | Prompt text used for this clip |
| `model` | string | Checkpoint / pack id (`workflow_meta.checkpoint` else variant) |
| `workflow_id` | string | Catalog variant / workflow slug |
| `seed` | int \| null | Generation seed |
| `params` | object | Sampler / size / duration / negative (see below) |
| `attempt` | int | 1-based attempt index |
| `iteration` | int | Same as `attempt` (Rust alias; keep both) |
| `judge_score` | number | Combined judge score for this attempt |
| `judge_reasons` | string[] | Judge + quality-bar fail reasons |
| `revise_notes` | string | What changed this attempt |

Always present extras (not in the Rust-required list, but keyed on the
run row):

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `buddy.clip.provenance/v1` |
| `path` | string | Clip path this object describes |
| `hash` | string | SHA-256 of the clip bytes (empty if missing) |
| `run_id` | string | Orchestrator run id |

`params` keys written today: `steps`, `cfg`, `width`, `height`,
`duration_s`, `stg_scale`, `stg_blocks`, `sampler_name`, `quality`,
`negative_prompt`. Extra keys are allowed.

## When it is written

| Event | Writer | Sidecar | Run JSON |
|---|---|---|---|
| Generate (`RESOLVE` copy into `outputs/`) | `persist_clip_provenance` | yes (clip exists) | later `_finish` |
| Judge (every attempt) | `persist_clip_provenance` | in-place update | later `_finish` |
| Revise + re-run | same path; `attempt`/`iteration` increment; `revise_notes` from the plan | in-place (dest name unchanged) | `provenance_history` appends |
| `--self-improve-dry` | judge persist only | only if a clip path exists | always embedded |
| Stitch | `inherit_clip_provenance` from first segment | next to master | pipeline record |
| Music mux | `inherit_clip_provenance` from first shot | next to muxed file | music record |

Sidecar dest name stays `<clip_stem>.provenance.json` across retries because
the orchestrator copies onto the same output name. History lives on the run
row (`provenance_history`).

## Example

```json
{
  "schema": "buddy.clip.provenance/v1",
  "prompt": "neon rain alley, handheld",
  "model": "ltx-2.5.safetensors",
  "workflow_id": "ltx25_t2v_i2v",
  "seed": 42,
  "params": {"steps": 8, "cfg": 3.5, "width": 768, "height": 512, "duration_s": 3.0},
  "attempt": 2,
  "iteration": 2,
  "judge_score": 0.81,
  "judge_reasons": ["quality_bar.c: thin still"],
  "revise_notes": "quality_bar.c: plan the still→I2V shot; prompt: slow push-in",
  "path": "outputs/abc123_clip.mp4",
  "hash": "…64 hex…",
  "run_id": "abc123"
}
```

## Code

- `master_agent/provenance.py` — build / read / write / inherit
- `orchestrator/machine.py` — persist on resolve, judge, finish
- `orchestrator/pipeline.py` — copy onto `PipelineResult`; inherit on stitch
- `music/pipeline.py` — inherit on mux

Tests (no GPU): `python -m pytest tests/test_clip_provenance.py -q`
