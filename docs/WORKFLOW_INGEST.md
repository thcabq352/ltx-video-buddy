# Workflow ingest

Phase A learns a one-off ComfyUI API graph and runs it through the same patcher, `vae_guard`, and Comfy client as a catalog variant. The graph stays on this machine under `state/ingested/`. It is not copied into `workflows/` and it is not added to `manifests.yaml`.

Catalog variants stay on `comfy run --variant`. Inpaint, sulphur, and lipsync are not detected or rerouted from an ingested graph. Keep using `--variant` for those.

No web UI. No custom-node install. No weight download. No LLM. Outputs are never deleted.

## Commands

From `video_buddy/`:

```bash
python -m master_agent comfy ingest path/to/workflow_api.json --slug demo
python -m master_agent comfy learn demo
python -m master_agent comfy dry-run demo --prompt "neon rain" --seed 42
python -m master_agent comfy run --ingested demo --prompt "neon rain" --seed 42 --frames 25
```

`ingest` writes the bundle and stops. The default next action is `dry-run`. It does not queue. `--queue` on `ingest` is the only ingest flag that queues, and it uses the same path as `run --ingested`.

`learn` rebuilds `learned.yaml` from the stored API graph and prints it.

`dry-run` prints the resolved fields, low-confidence warnings, and `vae_guard` notes. It does not POST `/prompt`.

`run --ingested` queues. A low-confidence role is a warning, not a prompt and not a hard failure. The run still uses the best guess.

## Files

```text
state/ingested/<slug>/
  workflow_api.json    # API graph that runs
  learned.yaml         # field map, same vocabulary as manifests.yaml
  provenance.json      # source path, time, sha256; comfy version stays empty offline
  workflow_ui.json     # only after a UI→API conversion (not Phase A)
```

`state/ingested/` is gitignored.

UI-format JSON is refused. Phase A does not call Comfy `/workflow/convert`. Supply API JSON (`node id` → `class_type` / `inputs`).

## What learn maps

Heuristics only. The field map uses `node_id` or `class_type` + `index`, plus `input`, matching `workflows/manifests.yaml`.

| Role | Where it looks |
|---|---|
| `prompt` | Positive-titled CLIP text, or the Primitive widget that text links to |
| `negative_prompt` | Negative-titled CLIP text, else the next unbound text widget |
| `seed` | `RandomNoise.noise_seed`, else a sampler `seed` |
| `width`, `height`, `frames` | `Empty*Latent*` widgets. `frames` writes `length` on LTX latents |
| `checkpoint` | `CheckpointLoaderSimple.ckpt_name` |
| `filename_prefix` | `SaveVideo` / `SaveImage` when that widget exists |

A linked `CLIPTextEncode.text` is not a tunable field. The value is written on the Primitive (or other source) widget. That is the same lesson as lipsync and LTX 2.5: writing the linked encoder breaks the graph.

`--frames` on an LTX `length` widget is snapped with `snap_ltx_frames` (8n+1, minimum 9). When the graph also has `LTXVEmptyLatentAudio`, that node's `frames_number` is set to the same count. Omitted flags leave the baked widget values, including the baked seed. There is no random seed on this path.

Confidence `low` means the title did not name the role, or more than one widget tied. Those lines are logged and printed in dry-run. The guess is still applied.

## Before queue

Every ingested run applies `vae_guard`. A baked `taeltx*` / `tae*` preview VAE that feeds tiled decode (`VAEDecodeTiled`, `LTXVTiledVAEDecode`, or any class whose name has both `Tiled` and `Decode`) is swapped to `LTX23_video_vae_bf16.safetensors`. `--vae` (or `--set` that writes `vae_name`) naming a tiny preview VAE fails closed with `TinyVAETiledDecodeError`. Plain `VAEDecode` is unchanged. The stored `workflow_api.json` is not rewritten; the swap is on the copy that is queued.

`run --ingested` loads `/object_info` and fails before queue when a `class_type` in the graph is absent. Dry-run does not contact Comfy; it fails when a class is absent from the bundled offline catalog (the shipped LTX 2.3 base classes are in that catalog). The error lists every missing node. Buddy does not install the pack.

`vram_class` on a learned file is `unknown`. Model files are not downloaded and are not substituted.

## Not in this phase

UI drag-and-drop, history ingest, URL ingest, promote-to-catalog, and fingerprint routing onto inoutpaint, sulphur, or lipsync.
