# Sulphur LTX 2.3 workflows

Phase 0 packaging for the Sulphur adult-studio graphs. The public repo
ships workflow JSON plus the path convention. LoRA weight blobs stay on
the tower. Buddy does not download them.

## Canonical source

Tower copies came from:

`ltx_director/models/checkpoints/sulphur-2/workflows/`

| Buddy file | Tower file | Format |
|---|---|---|
| `ltx23_i2v_base.json` | `ltx23_i2v base.json` | Comfy UI graph |
| `ltx23_i2v_distilled.json` | `ltx23_i2v distilled.json` | Comfy API graph |
| `ltx23_t2v_base.json` | `ltx23_t2v base.json` | Comfy UI graph |
| `ltx23_t2v_distilled.json` | `ltx23_t2v distilled.json` | Comfy UI graph |

Spaces in the tower names are underscores here so the paths are stable.
The four source files are the tower bytes. Do not hand-edit them.

## Queue format

Buddy queues API graphs (`class_type` + `inputs`). `ltx23_i2v_distilled.json`
is already API and is the runtime file for `ltx23_i2v_distilled`.

The other three are UI graphs. API siblings are generated with the repo
converter (cached `state/object_info.json`, no live Comfy required):

```bash
python state/convert_ui_to_api.py workflows/sulphur/ltx23_i2v_base.json workflows/sulphur/ltx23_i2v_base_api.json
python state/convert_ui_to_api.py workflows/sulphur/ltx23_t2v_base.json workflows/sulphur/ltx23_t2v_base_api.json
python state/convert_ui_to_api.py workflows/sulphur/ltx23_t2v_distilled.json workflows/sulphur/ltx23_t2v_distilled_api.json
```

Bypassed UI nodes (mode 4) are omitted from the API sibling, the same way
Comfy drops them at queue time. They remain in the UI source.

The cached object_info schema has no `ResizeImageResolution` or
`ImageScaleDownBy` class. On `ltx23_i2v_base_api.json` those two widgets
are filled from the tower API export of the same node types
(`resolution` / `method`, `scale_by`). That export is
`ltx23_i2v_distilled.json`.

| Buddy id | Runtime file |
|---|---|
| `ltx23_i2v_base` | `ltx23_i2v_base_api.json` |
| `ltx23_i2v_distilled` | `ltx23_i2v_distilled.json` |
| `ltx23_t2v_base` | `ltx23_t2v_base_api.json` |
| `ltx23_t2v_distilled` | `ltx23_t2v_distilled_api.json` |

These are heavy two-stage graphs, not the 16GB default. `base` is the
safer alternate. `eros_t2v_i2v.json` is a different pack and is unchanged.

```bash
python -m master_agent workflows
python -m master_agent comfy run --mode generate --variant ltx23_i2v_distilled --prompt "BRIEF" --prepare
```

A brief that says `sulphur` routes to `ltx23_i2v_distilled`. Say
`sulphur t2v` or `sulphur i2v base` to pick another graph.

Linked widgets stay links. Image-to-video width and height follow the
still. Text-to-video width, height, and length are scalar primitives, so
a generate run can set them. Positive prompt text is the `Prompt`
primitive. Negative text is the CLIP encode wired to
`LTXVConditioning.negative`. LoRA filename tokens inside the graphs are
left as authored.

## Where weights live

Sulphur and distill LoRAs stay on the tower. Buddy resolves a filename
token already written in a graph in this order:

1. `models/loras/sulphur/<filename>`
2. `models/loras/<filename>`

Either layout matches. The graphs use bare filenames, which is the Comfy
`models/loras/` combo form. This repo does not publish a folder inventory
and does not add those files to `download_models`.

Public Lightricks / Comfy-Org URLs already embedded in the UI graphs
(base LTX 2.3 checkpoint, Gemma, spatial upscaler, camera LoRA) stay in
those UI files. They are not Sulphur weight downloads.

Do not commit `.safetensors` or `.gguf` LoRA blobs. smut/ Smart Maker
packaging is out of scope for this pack.
