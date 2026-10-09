# Comfy

Two different features share the word "attach":

- `comfy attach` applies a previs **WorkflowPatchPlan** (`buddy.comfy.attach/v1`). It does not start a server.
- Managed / external mode decides who owns the Comfy process and where weights may be written.

Generate still uses HTTP. `ComfyClient` talks to `COMFYUI_URL` (default `http://127.0.0.1:8188`).

## Modes

| | Managed (`COMFY_MODE=managed`, the default) | External (`COMFY_MODE=external`) |
|---|---|---|
| Who runs Comfy | Buddy, via comfy-cli. Workspace `MANAGED_COMFY_ROOT` or `PROJECT_ROOT/ComfyUI` | You. Buddy never deletes processes or files in that tree. |
| `comfy start` / `stop` / `restart` | Launch, stop, or crash-restart the Buddy-owned server | **refuses start, stop, and restart** |
| `comfy status` | Health check | Health check only. Does not launch or stop |
| Weights | Downloads go only to Buddy `MODELS_DIR` | Same. Buddy never writes weights into the attached tree |

`COMFY_MODE` overrides `state/managed_comfy.json`. Accepted values are `managed` and `external`.

Point at a server you already run:

1. Set `COMFYUI_URL`.
2. Set `EXTERNAL_COMFY_ROOT` to the install that contains `models/`. `COMFYUI_ROOT` is the fallback when that root is not the managed workspace.
3. Set `COMFY_MODE=external`.

## Model paths YAML

Comfy sees extra folders only when it is started with `--extra-model-paths-config`. Buddy writes `video_buddy/state/extra_model_paths.yaml` and passes that path. The file is not copied into an attached tree unless you opt in.

- `buddy_models` is Buddy `MODELS_DIR` and the only section with `is_default: true`.
- `user_models_*` are read-only views of `EXTERNAL_COMFY_ROOT/models`, `COMFYUI_ROOT/models`, and `EXTRA_MODELS_DIRS`.

Search order stays `MODELS_DIR`, then Comfy `models/`, then the YAML and the Hugging Face cache. Consent for downloads: [Weights](WEIGHTS.md#consent).

```bash
python -m master_agent comfy status --write-yaml-into-external
```

That copies the same YAML to `EXTERNAL_COMFY_ROOT/extra_model_paths.yaml` (or `COMFYUI_ROOT`). It does not create a missing install and does not delete anything. It never writes weight files. The flag is the only consent to write into the attached tree.

If you start Comfy yourself:

```bash
--extra-model-paths-config /path/to/video_buddy/state/extra_model_paths.yaml
```

## Managed launch

`comfy start` asks comfy-cli for a local loopback server:

- `--where local` (no Comfy Cloud routing)
- after `--`: `--disable-auto-launch`, `--port 8188`, `--listen 127.0.0.1`
- `--extra-model-paths-config` pointing at the Buddy YAML, unless `--extra-model-paths` names an existing file
- `--use-sage-attention` is last, after `--listen` and after the extra-model-paths pair when that pair is present, when `sageattention` imports in the interpreter Comfy runs in. Buddy checks that with a subprocess `import sageattention` in `<workspace>/.venv` (or `<workspace>/venv`). Only when the workspace has no venv does it fall back to the interpreter running Buddy. If the import fails, Buddy logs a warning naming that interpreter and omits the flag so Comfy does not crash.

Once the workspace venv exists, comfy-cli calls run with `VIRTUAL_ENV` and `CONDA_PREFIX` removed. comfy-cli 1.20.0 would otherwise pick Buddy's activated venv over the workspace one.

Start, stop, status, and restart do not run `comfy install` or `comfy update`. Installing is the job of [automatic install](GETTING_STARTED.md#automatic-install), which runs `comfy --workspace=<root> --where local --skip-prompt install <gpu flag>` (`--nvidia`, `--amd`, `--m-series`, or `--cpu`; `--restore` added to repair a half-finished install), then `comfy … node install ComfyUI-LTXVideo ComfyUI-GGUF`. It refuses `COMFY_MODE=external`. `--no-watch` skips the crash-restart watchdog. `--no-wait` returns before `/system_stats` is ready. If `:8188` is already bound, do not kill a running render. Attach or wait.

`comfy update` only reports stale packs until `--yes`. `COMFY_MODE=external` refuses core and node updates. `--cli --yes` pip-installs the pin in `requirements.txt` (`comfy-cli==1.20.0`). It does not float to latest. Before a core or node update, Buddy saves `state/comfy_snapshots/pre-update-….json` via `comfy node save-snapshot`. That snapshot is not a weight rollback. Keep/wipe of LTX packs is `models select`, not this snapshot.

## Triton and SageAttention

Automatic install puts both into the Comfy venv, never Buddy's `.venv` or the system Python. It runs `<comfy-python> -m pip install …`, Triton first, then SageAttention, and prints the Comfy Python, torch, and CUDA versions.

| Platform | Triton | SageAttention |
|---|---|---|
| Linux, NVIDIA | `triton`, pinned to torch's own `triton==` requirement when torch declares one | `sageattention` (PyPI) |
| Windows, NVIDIA | `triton-windows` matched to torch (2.6 → `>=3.2,<3.3` … 2.10 → `>=3.6,<3.7`; newer torch is extrapolated and marked unverified) | Prebuilt wheel from `woct0rdho/SageAttention` releases for the CUDA tag and torch version. If none matches, or it fails to install or import, `sageattention` from PyPI. |
| macOS, CPU-only, AMD, non-CUDA torch, other CPUs | skipped | skipped |

After each install Buddy runs `import triton` / `import sageattention` in the Comfy Python. Every failure is `FAILED-NONFATAL` with a plain message, for example "No Triton build could be installed for Python 3.12 / CUDA 12.8 on this machine; continuing without Sage Attention. Comfy will run, just slower." A failed Triton skips SageAttention. Already-installed packages are skipped. `--skip-sage` skips the step.

## Attach contract

`python -m master_agent comfy attach` consumes `buddy.comfy.attach/v1` (also `buddy.comfy.attach` or `https://buddy.video/schema/comfy.attach/v1`). Code: `master_agent/comfy/attach.py`. Do not invent a second schema.

```bash
python -m master_agent comfy attach --recipe previs.json --json workflow.json --out patched.json
python -m master_agent comfy attach --recipe previs.json --json workflow.json --submit
```

Dry-run is the default: patch plus `validate_workflow`. No `/prompt`. `--submit` POSTs the patched graph and records `prompt_id` only. It does not invent a render success. Missing required classes fail closed. Optional accelerators still soft-bypass.

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

Buddy patches existing `LoadImage` / `LoadVideo` widgets and positive prompt fields. It does not invent topology. Camera is prompt-only. The recipe does not carry VRAM, slot, or a weight path.

Run JSON records `previs_source` (quality-bar bookkeeping for rule c), `control_pack_present`, and `control_pack_used` per channel (`openpose`, `depth`, `edges`, `camera`). No face scores.

`HeartMuLa_Generate` and `HeartMuLa_Transcribe` are payload nodes, not control channels. Attach does not treat them as optional accelerators. Any other class name containing `HeartMuLa` raises `AttachError`.

One-off graphs (not catalog defaults) are ingested, learned, and dry-run before queue. The Comfy tab drop zone uses that path. Procedure: [Workflow ingest](WORKFLOW_INGEST.md).

## Driving a graph

```bash
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent comfy run --mode template --template base --set 12.steps=8
python -m master_agent comfy run --mode raw --json workflow.json
python -m master_agent fetch-object-info
python -m master_agent validate --all --offline
```

`--prepare` stops before the queue. Validator checks class types, required inputs, widget values, link types (`COMFY_MATCHTYPE_V3` counts as a wildcard), and that loader filenames exist in the inventory.

LTX frame law: lengths are `8n+1`, minimum 9. The patcher writes `EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number` together. Illegal counts warn and auto-correct unless `--strict`.
