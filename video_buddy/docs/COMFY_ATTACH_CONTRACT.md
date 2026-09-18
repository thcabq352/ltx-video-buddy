# `buddy.comfy.attach/v1` contract

Vendored here so public Video Buddy docs do not link a private sibling
repo. This is the apply-recipe format Python Buddy consumes. Do not invent
a second schema in `master_agent/comfy/attach.py`.

## Schema tokens

Accepted on a recipe object (top-level or nested under
`workflow_patch_plan` / `recipe`):

- `buddy.comfy.attach/v1` (canonical)
- `buddy.comfy.attach`
- `https://buddy.video/schema/comfy.attach/v1`

## Recipe shape

```json
{
  "schema": "buddy.comfy.attach/v1",
  "previs_source": "path-or-id",
  "preferred_variants": ["ltx25_t2v_i2v"],
  "required_nodes": ["LoadImage"],
  "control_pack": {
    "openpose": {"image": "pose.png"},
    "depth": {"image": "depth.png"},
    "edges": {"image": "edges.png"},
    "camera": {"prompt": "slow push-in"}
  },
  "patches": []
}
```

| Field | Role |
|---|---|
| `schema` | Must be a v1 token above |
| `previs_source` / `source` / `previs` | Recorded on the run JSON (judge rule c) |
| `preferred_variants` / `director.preferred_variants` | First allowlisted slug wins when a control pack is present |
| `required_nodes` / `required_class_types` | Extra class names that must exist in `/object_info` |
| `control_pack` / `channels` | `openpose`, `depth`, `edges`, `camera` |
| `patches` / `ops` | Optional graph ops (`set_widget`, `rewire`, …) |

Channels accept `image` / `video` media and a `prompt` additive. Camera is
prompt-only. Buddy patches existing `LoadImage` / `LoadVideo` widgets whose
title matches the channel; it does not invent topology.

## Consumer behavior

- Dry-run default: patch + `validate_workflow`. No `/prompt` queue.
- `--submit` POSTs the patched graph and records `prompt_id` only.
- Missing required class types fail closed (non-zero). Optional accelerators
  still soft-bypass via `graph_ops.is_optional_node`.
- Never serialize VRAM / slot / `weight_path` on this recipe.

Operator notes: [`COMFY_ATTACH.md`](COMFY_ATTACH.md).
