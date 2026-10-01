# Managed Comfy and external mode

Buddy can own a local ComfyUI process, or leave a user-owned server alone.
The generate path does not change: `ComfyClient` still talks HTTP to
`COMFYUI_URL` (default `http://127.0.0.1:8188`).

`comfy attach` is still a previs **WorkflowPatchPlan** (`buddy.comfy.attach/v1`).
It is not this feature. External mode does not call `comfy/attach.py`.

## Modes

| | Managed (`COMFY_MODE=managed`, the default) | External (`COMFY_MODE=external`) |
|---|---|---|
| Who runs Comfy | Buddy, via comfy-cli, workspace `MANAGED_COMFY_ROOT` or `PROJECT_ROOT/ComfyUI` | You. Buddy never deletes processes or files in that tree. |
| `comfy start` / `stop` / `restart` | Launch, stop, or crash-restart the Buddy-owned server | **refuses start, stop, and restart** |
| `comfy status` | Health check | Health check only. Does not launch or stop. |
| Weights | Downloads go only to Buddy `MODELS_DIR` | Same. Buddy never writes weights into the attached tree. |

`COMFY_MODE` overrides `state/managed_comfy.json`. Accepted values are
`managed` and `external`.

## Pointing at a server you already run

1. Set `COMFYUI_URL` to the server the HTTP client should use
   (default `http://127.0.0.1:8188`).
2. Set `EXTERNAL_COMFY_ROOT` to the attached install directory
   (the folder that contains `models/`). `COMFYUI_ROOT` is the fallback when
   `EXTERNAL_COMFY_ROOT` is unset and that root is not the managed workspace.
3. Set `COMFY_MODE=external` so Buddy will not start, stop, or restart it.

Buddy never deletes files in that tree and never writes weights there.
`download-models` and other pack pulls write only under `MODELS_DIR`.
`HEARTMULA_MODELS_DIR`, when set, must still be inside `MODELS_DIR`
(the default is `MODELS_DIR/heartmula`).

## Model paths YAML

ComfyUI only sees extra model folders when it is started with
`--extra-model-paths-config`. Buddy writes that file under its own state
directory:

`video_buddy/state/extra_model_paths.yaml`

Managed `comfy start` and `comfy restart` generate it and pass that path.
The file is not written into the managed workspace or into an attached
install unless you opt in.

Sections:

- `buddy_models` — Buddy `MODELS_DIR`. This is the only section with
  `is_default: true`, so a Comfy-side download targets Buddy's folder.
- `user_models_*` — read-only views of `EXTERNAL_COMFY_ROOT/models`,
  `COMFYUI_ROOT/models`, and `EXTRA_MODELS_DIRS`. No `is_default`.

Inventory search order stays **`MODELS_DIR`, then Comfy `models/`, then
YAML (including this state file) and the Hugging Face cache**. A file in
`MODELS_DIR` wins over the same name in an attached tree.

If Comfy is already running and Buddy cannot relaunch it (`COMFY_MODE=external`
refuses start), pass the Buddy file yourself:

```bash
# on the ComfyUI process you start
--extra-model-paths-config /path/to/video_buddy/state/extra_model_paths.yaml
```

`comfy status` refreshes the Buddy-owned file without launching Comfy.

### Writing the YAML into an attached tree

Default is the Buddy-owned file only. To also copy that same YAML onto the
attached install (still not a weight file):

```bash
python -m master_agent comfy status --write-yaml-into-external
```

The copy lands at `EXTERNAL_COMFY_ROOT/extra_model_paths.yaml` (or
`COMFYUI_ROOT` when that is the attached root). Buddy does not create a
missing install and does not delete anything already there. The flag is the
only consent to write into the attached tree.

## Managed launch flags

`comfy start` always asks comfy-cli for a local loopback server:

- `--where local` (no Comfy Cloud routing)
- after `--`: `--disable-auto-launch`, `--port 8188`, `--listen 127.0.0.1`
- `--extra-model-paths-config` pointing at the Buddy YAML, unless you passed
  `--extra-model-paths` with an existing file

This module does not run `comfy install` or `comfy update`.

## Try this on the tower after merge

Do not treat a checkout of this change as a reason to download weights or to
run `comfy install` / `comfy update`. After the branch is merged, and only
when a Comfy tree is already on the machine:

```bash
cd video_buddy
python -m master_agent comfy status
# read state/extra_model_paths.yaml — user sections are read-only

# managed, existing workspace only (does not install or update ComfyUI)
python -m master_agent comfy start --no-watch
python -m master_agent health
python -m master_agent comfy stop

# external: your server, your tree
export COMFY_MODE=external
export COMFYUI_URL=http://127.0.0.1:8188
export EXTERNAL_COMFY_ROOT=/path/to/your/ComfyUI
python -m master_agent comfy status
python -m master_agent comfy start   # must fail; Buddy does not touch that process
```

`python -m master_agent download-models` still lists missing files and does
not fetch until `--yes`. Those bytes go to `MODELS_DIR` only.
