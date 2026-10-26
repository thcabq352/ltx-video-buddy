# Workflow ingest

Buddy learns a one-off ComfyUI graph and runs it through the same patcher, `vae_guard`, and Comfy client as a catalog variant. The graph stays on this machine under `state/ingested/` until an explicit `comfy promote` copies a draft into the local `workflows/` checkout. Promote does not change the catalog default.

Catalog variants stay on `comfy run --variant`. An ingested graph that matches inoutpaint, sulphur, or lipsync is routed through that family's existing helper. `--no-family-route` keeps it generic. Unmatched graphs stay generic.

No custom-node install. No weight download. Outputs are never deleted. Promote does not commit, push, or open a pull request. `--llm-assist` is off unless you pass it, and it stays on a local model.

## Commands

From `video_buddy/`:

```bash
python -m master_agent comfy ingest path/to/workflow_api.json --slug demo
python -m master_agent comfy ingest path/to/workflow_api.json --slug demo --llm-assist
python -m master_agent comfy ingest path/to/workflow_ui.json --slug demo
python -m master_agent comfy ingest --from history:PROMPT_ID --slug demo
python -m master_agent comfy learn demo
python -m master_agent comfy dry-run demo --prompt "neon rain" --seed 42
python -m master_agent comfy promote demo --variant-name demo-draft
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
  provenance.json      # source path, time, sha256; comfy version when history answered
  workflow_ui.json     # original UI export when ingest converted one
```

`state/ingested/` is gitignored.

## UI JSON and history

API JSON (`node id` → `class_type` / `inputs`) is stored as `workflow_api.json`.

UI-format JSON (a `nodes` list) is posted to Comfy `/workflow/convert` when Comfy is reachable. Both files are kept: `workflow_ui.json` is the export you dropped, `workflow_api.json` is the converted graph that runs. When Comfy is down, ingest refuses with `start Comfy or supply API JSON`. Buddy does not convert UI JSON on its own.

`--from history:PROMPT_ID` reads Comfy `/history/<prompt_id>` and stores the queued prompt graph. The same refusal is used when that request fails. No weights are downloaded.

## Readiness

`learned.yaml` `readiness` lists:

- `missing_nodes` — class types absent from live `/object_info`, or from the bundled offline catalog when Comfy was not asked. `missing_node_packs` adds a best-effort pack name from that class's object_info `python_module` (`custom_nodes.<pack>....`) or from a known class→pack map (`VHS_*` → Video Helper Suite, LTX nodes → `ComfyUI-LTXVideo`). Unknown packs stay unnamed. Buddy does not install the pack.
- `missing_models` — loader filenames that are not in the Comfy `/models/<folder>` lists and not under a `base_path` in Buddy `state/extra_model_paths.yaml`. The check runs only when one of those inventories exists (`models_checked`). A missing name is left exactly as written. Buddy does not substitute another file. The report points at `python -m master_agent doctor` and `python -m master_agent download-models`.

`run --ingested` refuses when `models_checked` is true and `missing_models` is not empty. Dry-run prints the list and does not queue.

## What learn maps

Heuristics only. The field map uses `node_id` or `class_type` + `index`, plus `input`, matching `workflows/manifests.yaml`.

| Role | Where it looks |
|---|---|
| `prompt` | Positive-titled CLIP text, or the Primitive widget that text links to |
| `negative_prompt` | Negative-titled CLIP text, else the next unbound text widget |
| `seed` | `RandomNoise.noise_seed`, else a sampler `seed` |
| `width`, `height`, `frames` | `Empty*Latent*` widgets. `frames` writes `length` on LTX latents |
| `checkpoint` | `CheckpointLoaderSimple.ckpt_name`, else `UnetLoaderGGUF.unet_name` on the shared `ltx23_av.json` graph. The text-projection `ckpt_name` on `LTXAVTextEncoderLoader` is not this role |
| `filename_prefix` | `SaveVideo` / `SaveImage` when that widget exists |

A linked `CLIPTextEncode.text` is not a tunable field. The value is written on the Primitive (or other source) widget. That is the same lesson as lipsync and LTX 2.5: writing the linked encoder breaks the graph.

`--frames` on an LTX `length` widget is snapped with `snap_ltx_frames` (8n+1, minimum 9). When the graph also has `LTXVEmptyLatentAudio`, that node's `frames_number` is set to the same count. Omitted flags leave the baked widget values, including the baked seed. There is no random seed on this path.

Confidence `low` means the title did not name the role, or more than one widget tied. Those lines are logged and printed in dry-run. The guess is still applied.

## Before queue

Every ingested run applies `vae_guard`. A baked `taeltx*` / `tae*` preview VAE that feeds tiled decode (`VAEDecodeTiled`, `LTXVTiledVAEDecode`, or any class whose name has both `Tiled` and `Decode`) is swapped to `LTX23_video_vae_bf16.safetensors`. `--vae` (or `--set` that writes `vae_name`) naming a tiny preview VAE fails closed with `TinyVAETiledDecodeError`. Plain `VAEDecode` is unchanged. The stored `workflow_api.json` is not rewritten; the swap is on the copy that is queued.

`run --ingested` loads `/object_info` and fails before queue when a `class_type` in the graph is absent. Dry-run does not contact Comfy; it fails when a class is absent from the bundled offline catalog (the shipped LTX 2.3 base classes are in that catalog). The error lists every missing node. Buddy does not install the pack.

`vram_class` on a learned file is `unknown`. Model files are not downloaded and are not substituted.

## Family route

A fingerprint picks at most one family:

| Family | Signal | Existing path |
|---|---|---|
| inoutpaint | `LTXVInpaintPreprocess` | `finalize_inoutpaint_graph` |
| lipsync | `LTXAddVideoICLoRAGuide` or `LTXVSetAudioRefTokens` | source-widget prompt; `prepare_queue_inputs` at queue |
| sulphur | `PathchSageAttentionKJ`, `LTX2SamplingPreviewOverride`, or a LoRA filename containing `sulphur` | `patch_sulphur_graph` |

inoutpaint is checked first. A 2.5 in/outpaint graph (`VHS_DuplicateMasks`, `LTXVImgToVideoInplace`, or `LTXVImgToVideoConditionOnly`) is finalized with `ltx25=True`, so the PR #44 mask repeat and `trim_to_shortest=false` still run. The learned file records `family` and `family_route` (`specialized` or `generic`). Dry-run prints the same line.

An unmatched graph stays on the generic patcher and dry-run warns `unmatched graph stays on the generic path`.

`--no-family-route` on ingest or learn stores `family_route: generic`. The same flag on dry-run or run forces generic for that call. `learn` without the flag turns routing back on when the graph still matches.

## Promote

`comfy promote SLUG [--variant-name NAME]` writes two local files and prints a unified diff:

- `workflows/<name>.json` — a copy of the stored API graph
- an appended block in `workflows/manifests.yaml` — `file`, `description`, `vram_class`, `fields`, `inputs`, `outputs`, `requires`, plus `draft: true`

The block uses the same field vocabulary as the rest of the manifest (`node_id` or `class_type` + `index`, plus `input`). A low-confidence role is a YAML comment above that field, not an extra key.

Promote refuses when `readiness` lists missing nodes or models. `--force` writes the draft anyway and does not install nodes or substitute weights. It refuses a `--variant-name` that is already a catalog default (`base`, or any id `default_variant_ids()` publishes) and refuses a name that already exists in `manifests.yaml`. It does not replace `base`. It does not run git, and it does not open a pull request. `comfy run --variant` does not pick the new draft unless you pass that name yourself.

## LLM assist

`--llm-assist` on `ingest` or `learn` is off by default. It asks a local model to name widgets that the heuristics already marked low confidence. High-confidence roles are not sent and are not overwritten.

The backend order is the repo's local order: llama.cpp when that server is already up, otherwise Ollama. The flag does not call `get_llm("auto")`, does not start llama.cpp, does not download a weight, and does not call Grok or any other cloud model. When neither local server is up, the command warns and keeps the heuristic names.

A proposal is stored on the field as `source: llm` plus a confidence, listed under `llm_proposals`, and printed in dry-run. A proposal that would take a role the graph already mapped is shown and not applied. Low confidence still warns and proceeds.

## Not in this phase

URL ingest.
