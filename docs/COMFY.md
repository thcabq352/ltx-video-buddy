# Comfy

Two different features share the word "attach":

- `comfy attach` applies a previs **WorkflowPatchPlan** (`buddy.comfy.attach/v1`). It does not start a server.
- Managed / external mode decides who owns the Comfy process and where weights may be written.

Generate still uses HTTP. `ComfyClient` talks to `COMFYUI_URL` (default `http://127.0.0.1:8188`).

## Modes

| | Managed (`COMFY_MODE=managed`, the default) | External (`COMFY_MODE=external`) |
|---|---|---|
| Who runs Comfy | Buddy. Installed with comfy-cli, started by Buddy itself. Workspace: see [Which folder](#which-folder) | You. Buddy never deletes processes or files in that tree. |
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

## Which folder

Every command (automatic install, doctor, `comfy start` / `stop` / `status` / `restart` / `update`, model search, `download-models`) asks `master_agent/comfy/root.py` where Buddy's ComfyUI lives. First match wins:

1. `comfy start --workspace <dir>` (remembered in `state/managed_comfy.json` for later runs)
2. `MANAGED_COMFY_ROOT` in the environment or `.env`
3. the remembered `--workspace`, when it still has `main.py`. A remembered folder without `main.py` is stale: Buddy ignores it, prints `INFO  ignored stale state/managed_comfy.json workspace … (no main.py there)`, and clears it at the next `comfy start`.
4. `video_buddy/ComfyUI`, when it has `main.py`
5. a Windows portable build inside `video_buddy` (`ComfyUI_windows_portable*`, or `COMFYUI_PORTABLE_ROOT`), shown as `… (auto-detected portable)`
6. `video_buddy/ComfyUI` anyway, so the "not installed" error names it

Only `--workspace` is remembered. `MANAGED_COMFY_ROOT`, the default and an auto-detected portable are worked out again every run.

`comfy status`, the automatic-install pre-flight, and doctor's `comfyui` row print the folder and where it came from, for example `E:\VideoBuddy\ComfyUI (from MANAGED_COMFY_ROOT)`. `MANAGED_COMFY_ROOT` is Buddy's own install. `COMFYUI_ROOT` / `EXTERNAL_COMFY_ROOT` name a ComfyUI you own, which Buddy only reads.

### Windows portable build

`MANAGED_COMFY_ROOT` (or `--workspace`) may name the portable top folder (`ComfyUI_windows_portable`, the one with `python_embeded\` and `run_*.bat`) or its inner `ComfyUI` folder. Both resolve to the inner folder, which is what custom nodes, models and comfy-cli use. When `python_embeded\python.exe` sits next to that folder (or inside it), `comfy start` launches the way the portable's own `.bat` does: cwd is the top folder and the command is

```text
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-auto-launch --port 8188 --listen 127.0.0.1 …
```

`--disable-auto-launch` comes after `--windows-standalone-build`, so no browser opens. The sageattention check runs in `python_embeded`. Automatic install never runs `comfy install` over a portable build, and refuses to install a second ComfyUI while one is found: it tells you to set `MANAGED_COMFY_ROOT` to it instead. A portable whose `python_embeded` cannot import torch is reported, not repaired; use the portable's own `update\` scripts.

The tower's `.env`:

```ini
MANAGED_COMFY_ROOT=F:\CircusMoved\ltx2.3_agent\video_buddy\ComfyUI_windows_portable
COMFY_EXTRA_ARGS=--preview-method auto --use-sage-attention
```

## Managed launch

`comfy start` runs `<workspace>/.venv` python (`venv` also works; a portable build uses `python_embeded`, see [Windows portable build](#windows-portable-build); with none of them, the interpreter running Buddy) as `main.py` in the workspace folder, detached, with a local loopback server:

- `--disable-auto-launch`, `--port 8188`, `--listen 127.0.0.1`
- `--extra-model-paths-config` pointing at the Buddy YAML, unless `--extra-model-paths` names an existing file
- `--use-sage-attention`, after `--listen` and after the extra-model-paths pair when that pair is present, when `sageattention` imports in the interpreter Comfy runs in. Buddy checks that with a subprocess `import sageattention` in `<workspace>/.venv` (or `<workspace>/venv`, or a portable's `python_embeded`). Only when the workspace has no interpreter of its own does it fall back to the interpreter running Buddy. If the import fails, Buddy logs a warning naming that interpreter and omits the flag so Comfy does not crash.
- `COMFY_EXTRA_ARGS`, last. Empty by default, so the command is unchanged. It is split like a command line (quote paths with spaces) and applies to venv and portable layouts alike. A repeated switch (`--use-sage-attention`, `--windows-standalone-build`, `--disable-auto-launch`) is dropped, so `COMFY_EXTRA_ARGS=--preview-method auto --use-sage-attention` is safe whether or not the sageattention check passed. A flag that needs sageattention still crashes Comfy when it is missing; the launch error shows the log tail.

### A ComfyUI Buddy did not start

When something already answers on the port and Buddy did not start it (for example `run_api_8188.bat`), `comfy status` reports it as up with `foreign: true` and `owner` (pid, command line, cwd, the `main.py` it runs, and `matches_workspace`). The text output says `up (not started by Buddy)` and `owner pid=…`.

`comfy start` and `comfy restart` refuse it by default, before stopping anything, and say what to do:

- It runs this workspace's `main.py` (cmdline path, resolved against its cwd): `python -m master_agent comfy restart --adopt` (`--force` is the same) stops that process tree and starts Buddy's managed ComfyUI. Agents call `comfy_restart` with `{"adopt": true}`. Or keep using it as is, or set `COMFY_MODE=external`.
- It runs a different ComfyUI: Buddy will not stop it, even with `--adopt`. Stop it yourself (`taskkill /PID <pid> /T /F`), point `MANAGED_COMFY_ROOT` at that ComfyUI, or set `COMFY_MODE=external`.
- Buddy cannot tell who owns the port (psutil missing, or access denied): stop it yourself, then `comfy start`.

ComfyUI's output goes to `video_buddy/state/logs/comfyui_8188.log` (new file each start). `comfy start` then polls `/system_stats` until it answers (180 s). If ComfyUI exits first, `comfy start` fails with its exit code, the command, the folder, the python, and the last 40 lines of that log. `--as-json` puts the same fields under `launch`, and `health` (agent API) repeats them under `comfyui.managed`. `stop` ends the process Buddy started; a server an older Buddy started through `comfy launch --background` is still stopped with `comfy stop`.

Agents use the agent tools `comfy_status`, `comfy_start` and `comfy_restart` (`python -m master_agent agent comfy_start`). They run the same code and return the same JSON, plus `next`. `comfy_start` returns `starting` instead of an error when ComfyUI is still loading at `timeout_s`. An agent must never start ComfyUI by running `main.py` or `run_*.bat` in a foreground terminal: it never returns, so the tool time limit kills it. See [Agents](AGENTS.md#starting-comfyui).

Once the workspace venv (or a portable's `python_embeded`) exists, ComfyUI and comfy-cli run with `VIRTUAL_ENV` and `CONDA_PREFIX` removed, so Buddy's activated venv is not used for Comfy.

Start, stop, status, and restart do not run `comfy install` or `comfy update`. Installing is the job of [automatic install](GETTING_STARTED.md#automatic-install), which runs `comfy --workspace=<root> --where local --skip-prompt install <gpu flag>` (`--nvidia`, `--amd`, `--m-series`, or `--cpu`; `--restore` added to repair a half-finished install), then `comfy … node install ComfyUI-LTXVideo ComfyUI-GGUF`, then the [Comfy Python packages](#comfy-python-packages) and the [ComfyUI-Manager channel](#comfyui-manager-channel). It refuses `COMFY_MODE=external`. `--no-watch` skips the crash-restart watchdog. `--no-wait` returns about 3 s after launch, before `/system_stats` is ready (a crash in those seconds is still reported). If `:8188` is already bound, do not kill a running render. Attach or wait.

`comfy update` only reports stale packs until `--yes`. `COMFY_MODE=external` refuses core and node updates. `--cli --yes` pip-installs the pin in `requirements.txt` (`comfy-cli==1.20.0`). It does not float to latest. Before a core or node update, Buddy saves `state/comfy_snapshots/pre-update-….json` via `comfy node save-snapshot`. That snapshot is not a weight rollback. Keep/wipe of LTX packs is `models select`, not this snapshot.

## Triton and SageAttention

Automatic install puts both into the Comfy venv, never Buddy's `.venv` or the system Python. It runs `<comfy-python> -m pip install …`, Triton first, then SageAttention, and prints the Comfy Python, torch, and CUDA versions.

| Platform | Triton | SageAttention |
|---|---|---|
| Linux, NVIDIA | `triton`, pinned to torch's own `triton==` requirement when torch declares one | `sageattention` (PyPI) |
| Windows, NVIDIA | `triton-windows` matched to torch (2.6 → `>=3.2,<3.3` … 2.10 → `>=3.6,<3.7`; newer torch is extrapolated and marked unverified) | Prebuilt wheel from `woct0rdho/SageAttention` releases for the CUDA tag and torch version. If none matches, or it fails to install or import, `sageattention` from PyPI. |
| macOS, CPU-only, AMD, non-CUDA torch, other CPUs | skipped | skipped |

After each install Buddy runs `import triton` / `import sageattention` in the Comfy Python. Every failure is `FAILED-NONFATAL` with a plain message, for example "No Triton build could be installed for Python 3.12 / CUDA 12.8 on this machine; continuing without Sage Attention. Comfy will run, just slower." A failed Triton skips SageAttention. Already-installed packages are skipped. `--skip-sage` skips the step.

## Comfy Python packages

Automatic install runs `<comfy-python> -m pip install --upgrade -r video_buddy/comfy-requirements.txt` (`gguf==0.19.0`, `colour==0.1.5`) in the Comfy venv, or in `python_embeded` with `-s` for a portable build. The file holds exact pins only, so `--upgrade` keeps those versions. A pip failure is `FAILED-NONFATAL`; run `automatic-install --yes` again to retry.

ComfyUI-LTXVideo also installs `colour-science`, which provides the same `colour` import name. When both are installed, `import colour` loads `colour-science`.

## Install CI

`.github/workflows/fresh-install.yml` runs `install.py --automatic-install --yes --skip-weights --skip-sage --skip-llm --gpu cpu` from a clean checkout. It then starts ComfyUI with `COMFY_EXTRA_ARGS="--cpu --enable-manager"` through `comfy start`. Manager only loads with `--enable-manager`, and Buddy's own launch does not pass that flag. Each launch waits for `/system_stats`. It fails on `InvalidChannel`, `ModuleNotFoundError` or `IMPORT FAILED` in the log, and on Manager's `/v2/manager/channel_url_list` not selecting the Comfy-Org channel. The job also:

- installs the Klein stills packs and checks the pins after `pip install --upgrade` of other packages and of ComfyUI's own requirements (torch excluded);
- checks `260615_VIDEO-BUDDY_FLUX_KLEIN_9B_V01_api.json` against the live `/object_info` and the offline validator;
- deletes Manager's `config.ini`, launches, repairs with `automatic-install`, launches again, then repairs a second time and checks that nothing changed.

Linux runs on pushes to `main` and PRs that touch the install path, nightly, and on demand. Windows runs nightly and on demand. The checks live in `video_buddy/scripts/ci/fresh_install_check.py`, and each subcommand also runs by hand. The `comfy-pins` job in `pytest.yml` checks the pins in a clean venv on every push.

## ComfyUI-Manager channel

`comfy install` adds ComfyUI-Manager, and Manager writes its own `config.ini` with `channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main`. Current Manager builds reject that and log `[InvalidChannel]` on every start ([Comfy-Org/ComfyUI-Manager#2709](https://github.com/Comfy-Org/ComfyUI-Manager/issues/2709), [#2561](https://github.com/Comfy-Org/ComfyUI-Manager/issues/2561)).

Automatic install points Manager at `https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main` (note the hyphen: `ComfyOrg` without it is a 404). Manager only accepts a `channel_url` listed in its `channels.list`, so both files are written, in `ComfyUI/user/__manager/` (or `ComfyUI/user/default/ComfyUI-Manager/` on older ComfyUI):

- `config.ini`: `[default] channel_url = <Comfy-Org URL>`. An existing file is edited in place: only an `ltdrdata` (or `ComfyOrg`) `channel_url` is replaced, or a missing one added. Every other key is kept, and a `channel_url` you set to another URL is left alone.
- `channels.list`: `default::<Comfy-Org URL>`, so the UI shows the `default` channel selected. Entries under the `ltdrdata` or `ComfyOrg` URL (`recent`, `legacy`, `forked`, `dev`, `tutorial`) move to the Comfy-Org URL with the same `node_db/...` path. Channels pointing anywhere else are kept.

Manager has no `channel` or `auto-updates` key in `config.ini`, so Buddy writes neither. Running automatic install again changes nothing once the files are fixed; on an existing ComfyUI it repairs them without reinstalling.

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

One-off graphs (not catalog defaults) are ingested, learned, and dry-run before queue. Procedure: [Workflow ingest](WORKFLOW_INGEST.md).

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
