> **Archived 2026-10-02.** Superseded by [`docs/INDEX.md`](../../../INDEX.md). If this copy disagrees with `docs/`, `docs/` wins.

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

Managed start, stop, status, and restart do not run `comfy install` or `comfy update`.
Startup prints the hardware sentence and any stale pins. It does not update.

## Model selector (CLI)

Phase C is the checklist. The studio Models tab calls the same functions.
This command does not run `comfy install` or `comfy update`.

The catalog is grouped by capability: video generation, soundtrack, and
local studio. `models manifest` prints that JSON. The 2.3 / 2.5 radio swaps
the video-generation rows. Required rows stay on and read **required for
generation**. Optional rows show a size when the repo already attests one.

```bash
cd video_buddy
python -m master_agent models manifest
python -m master_agent models select --version 2.5
python -m master_agent models select --version 2.3
python -m master_agent download-models --selector --version 2.5 --scan-only
```

`--scan-only` prints the running download total, the ETA (default 50 Mbit/s,
override with `--mbps`), and free space on `MODELS_DIR`. It does not fetch
and does not write `state/model_selector.json`.

A fetch still needs `--yes`, and only after the free-space check passes.
Files already on disk are skipped. Bytes go to `MODELS_DIR` through the
existing download guards.

```bash
python -m master_agent models select --version 2.5 --yes
python -m master_agent models select --version 2.5 --soundtrack --yes
```

`--soundtrack` is **Enable Soundtrack Studio**. It calls the existing
`download-models --heartmula` consent path (`download_missing_slots`). It
does not add a new music runtime.

Switching the radio asks whether to keep the previous pack or wipe it from
`MODELS_DIR`. Non-interactive runs pass `--keep` or `--wipe`. Keeping both
turns on per-job version: an LTX 2.3 job uses 2.3 files and an LTX 2.5 job
uses 2.5 files. If the requested version is missing, Buddy reports that and
offers `models select --version …`. It does not silently run the other pack.

Sulphur GGUF, Sulphur LoRA (`models/loras/sulphur/`), and the EROS checkpoint
are local detect flags. The selector does not download them and does not
store a Hub URL for them.

LTX 2.3 rows whose byte size is not attested in this repo are listed and are
not fetched. LTX 2.5 rows and Soundtrack Studio are the download paths.

## Model selector (web)

The studio Models tab is that same checklist. Start the dashboard, then open
Models:

```bash
cd video_buddy
python -m master_agent ui --port 8189
```

The page calls `catalog_document`, `assess`, and `apply_selection` in
`master_agent/models/selector.py`. It does not keep a second catalog.
Generate still uses the Comfy HTTP client on `:8188`. This tab does not
queue a job and does not run `comfy update`.

What you see:

- LTX 2.3 / 2.5 radio. Required rows stay locked and read **required for generation**.
- Optional rows with the size this repo already attests.
- **Enable Soundtrack Studio** (HeartMuLa). One toggle, the existing `--heartmula` path.
- A live download total and ETA. The ETA uses the Mbit/s field (default 50).
- Free space on `MODELS_DIR`. A short disk disables download.
- Files already on disk are marked skipped.
- Sulphur GGUF, Sulphur LoRA, and the EROS checkpoint are local detect flags.
- LTX 2.3 rows without an attested fetch stay scan-only.

**Scan only** posts `scan_only: true`. That does not download and does not
write `state/model_selector.json`. **Download selected** stays idle until
the agreement checkbox is checked. That sends `yes: true`, the same consent
as `--yes`. Switching the radio still asks you to keep or wipe the previous
pack under `MODELS_DIR`.

Scan-only smoke (no weight download):

```bash
curl -s 'http://127.0.0.1:8189/api/models/selector?version=2.5'
curl -s -X POST http://127.0.0.1:8189/api/models/selector/plan \
  -H 'content-type: application/json' \
  -d '{"version":"2.3","optional_ids":[],"soundtrack":false,"mbps":50}'
curl -s -X POST http://127.0.0.1:8189/api/models/selector/apply \
  -H 'content-type: application/json' \
  -d '{"version":"2.5","optional_ids":[],"soundtrack":false,"mbps":50,"scan_only":true,"yes":false}'
```

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

# report only — no comfy install, no comfy update, no weight download
python -m master_agent doctor
python -m master_agent comfy update
```

`python -m master_agent download-models` still lists missing files and does
not fetch until `--yes`. Those bytes go to `MODELS_DIR` only.

## Hardware scan

`doctor` and managed `comfy start` / `comfy status` / `comfy restart` print one
routing sentence. The row is `hardware`. It stays OK, and it does not stop
`comfy start` or `setup --fix`.

| What the scan sees | Sentence |
|---|---|
| NVIDIA, 12GB and above | Full checkpoints are in range. |
| NVIDIA, above 8GB and below 12GB | Use GGUF. Sulphur GGUF can run on less. |
| NVIDIA, 8GB and below | Use GGUF or CPU, or upgrade. Sulphur GGUF can run on less. |
| AMD with ROCm | Slower, and it works. The same VRAM bands pick the pack. |
| AMD without ROCm | Install ROCm first, or use GGUF. |
| No GPU | Use GGUF or CPU, or upgrade. The scan does not block install. |

`VRAM_GB`, when set, is the number in that sentence. The `vram-policy` row
still prefers an on-disk GGUF when one is present.

## Stale packs

Startup compares installed `comfy-cli` with the pin in `requirements.txt`
(`comfy-cli==1.20.0`) and compares custom nodes with
`master_agent/comfy/pack_pins.json`. A missing optional node is not stale.
ComfyUI core has no pin in this repo, so a core checkout is reported and is
not marked stale.

`doctor`, `setup --fix`, `comfy start`, and `comfy status` do not run
`comfy update` or `comfy node update`.

```bash
python -m master_agent comfy update
python -m master_agent comfy update --yes
python -m master_agent comfy update --core --yes
python -m master_agent comfy update --nodes --yes
python -m master_agent comfy update --cli --yes
```

`--yes` or a `y` at the prompt is the opt-in. A non-interactive run without
`--yes` prints the stale list and stops. `COMFY_MODE=external` refuses core
and node updates.

`--cli` pip-installs the requirements pin. It does not run `comfy update cli`.

A commit pin that does not match HEAD is reported. Bare `--yes` does not run
`comfy node update` for that pin. `--nodes --yes` moves custom nodes forward
after a snapshot, and it does not check out the pinned commit.

## Snapshots and weight keep/wipe

Before `comfy update comfy` or `comfy node update`, Buddy runs:

```bash
comfy node save-snapshot --output video_buddy/state/comfy_snapshots/pre-update-….json
```

That file records custom nodes and Python dependencies. It does not roll back
files under `MODELS_DIR`. Restore nodes with `comfy node restore-snapshot`
on the same managed workspace. That restore is not a weight rollback.

Switching LTX 2.3 and 2.5 still uses `models select --keep` or `--wipe`.
That prompt keeps or deletes files under `MODELS_DIR`. It is a different
action from the node snapshot.

Selector checklist (no fetch). Leave off `--yes` until you mean to pull:

```bash
python -m master_agent models manifest
python -m master_agent models select --version 2.5 --scan-only
python -m master_agent models select --version 2.3 --scan-only
python -m master_agent download-models --selector --version 2.5 --scan-only
```
