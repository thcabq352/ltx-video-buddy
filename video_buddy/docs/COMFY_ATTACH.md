# Comfy attach (`buddy.comfy.attach/v1`)

Python Buddy consumes a previs **WorkflowPatchPlan** / `buddy.comfy.attach/v1`
recipe and patches OpenPose / depth / edges / camera prompt additives onto a
compatible LTX or Wan **API** graph. It does not synthesize topology.

## Contract (source of truth)

The apply-recipe format is vendored in this tree:

- [`COMFY_ATTACH_CONTRACT.md`](COMFY_ATTACH_CONTRACT.md)
- `buddy.comfy.attach/v1` schema consumed by `master_agent/comfy/attach.py`

If the contract adds fields, extend `master_agent/comfy/attach.py` — do not
invent a second schema.

## Dry-run (default)

Patch + validate against live or cached `/object_info`. **No** `/prompt` queue.

```bash
cd video_buddy
python -m master_agent comfy attach \
  --recipe path/to/export-patch.json \
  --json path/to/workflow_api.json \
  --out patched.json
```

`--dry-run` is implied when `--submit` is absent. Missing control-pack nodes or
`object_info` classes fail honestly (non-zero exit). Nothing is claimed as a
Comfy success.

Director pipeline (plan + lint only):

```bash
python -m master_agent run "BRIEF" --attach path/to/export-patch.json --dry-run --no-interview
```

When the recipe includes a control pack and `director.preferred_variants`,
routing uses the first allowlisted preference (`source=attach`) unless
`--variant` or a source video forces another path.

## Live `/prompt` (explicit flag)

Requires Comfy on `:8188`. Does **not** invent a render success — it only POSTs
the patched graph and records the `prompt_id`.

```bash
python -m master_agent comfy attach \
  --recipe path/to/export-patch.json \
  --json path/to/workflow_api.json \
  --submit
```

Tower smoke on evilclownworld is still required to confirm live node packs
(OpenPose / depth / Fun Control) and that `/object_info` matches the graph.

## Run JSON (judge rules c / d)

Each attach writes `state/runs/<ts>_<id>_attach.json` (or the orchestrator run
record) with:

| Field | Judge |
|---|---|
| `previs_source` | (c) must be recorded when a recipe was applied |
| `control_pack_present` | pack had media or prompt additives |
| `control_pack_used` | per-channel `openpose` / `depth` / `edges` / `camera` |

`evaluate_attach_judge_rules()` is bookkeeping only — no face scores.

## Compatible graphs

Existing widgets only:

- titled `LoadImage` / `LoadVideo` (`OpenPose`, `Depth`, `Edges`, …)
- positive `CLIPTextEncode` / Wan Fun Control prompt fields

If the pack names a channel and the graph has no matching loader, attach
errors. Optional accelerators still bypass via the existing validator.

## HeartMuLa classes (not a control channel)

Tower Comfy Desk (2026-09-30) registered exactly two classes from
`benjiyaya/HeartMuLa_ComfyUI` @ `fdb53c4`:

| `class_type` | Display name |
|---|---|
| `HeartMuLa_Generate` | HeartMuLa Music Generator |
| `HeartMuLa_Transcribe` | HeartMuLa Lyrics Transcriber |

Attach does not patch their widgets and does not treat them as OpenPose /
depth / edges / camera. They are payload nodes, not optional accelerators,
so a missing class is not a TeaCache-style bypass. A recipe that names any
other class containing `HeartMuLa` raises `AttachError`. Do not invent
class names. Weights were not downloaded (empty model dirs). Buddy's own
generate/transcribe path is heartlib, documented in
[HEARTMULA.md](HEARTMULA.md). `HeartCodec-oss` returned 401; prefer
`HeartCodec-oss-20260123`. Check disk before a multi-GB pull
(F: ~28GB free on that tower check).
