# CLI reference

All commands are `python -m master_agent …` from `video_buddy/`. `python -m master_agent <cmd> --help` prints flags. This page names the commands that exist. It does not restate the loader order: [Weights](WEIGHTS.md#loader-policy).

## Setup and health

| Command | Role |
|---|---|
| `automatic-install` | Automatic install: pre-flight, then ffmpeg, comfy-cli, ComfyUI in its own folder, LTX 2.3 nodes, `gguf` + `colour` in the Comfy venv, ComfyUI-Manager channel fix (`config.ini` / `channels.list`), Triton + SageAttention in the Comfy venv, LTX 2.3 weights, llama.cpp (built from the pinned `LLAMACPP_REF`, official prebuilt as fallback; optional), LLM models (skipped with a note when no public GGUF source is defined). Never installs or calls Ollama. Plan only until `--yes`. `--preflight-only`, `--dry-run`, `--gpu auto\|nvidia\|amd\|m-series\|cpu`, `--skip-weights`, `--skip-sage`, `--skip-llm`. Also `python install.py --automatic-install …`. See [Getting started](GETTING_STARTED.md#automatic-install) |
| `curriculum` | L0→L5 card. `--json` |
| `about` | Studio card. Also `GET /api/about` |
| `doctor` / `setup` | Deps + weight scan. `--fix` installs deps, skips LLM GGUFs with no public source, and never pulls Ollama. `python install.py` runs `setup --fix`; `python install.py --check` runs `setup` (report only, needs an existing `.venv`). See [Weights](WEIGHTS.md#consent) |
| `inventory` | Discovered files and roles. `--json` |
| `workflows` | Default catalog. `--vram` prints the pack table. `--json` |
| `health` | Comfy `:8188`, GPU stats, and a one-line shift budget (`budget: used / cap VRAM-min used, remaining; paused runs`). `doctor` shows the same line as its `budget` row. See [Architecture](ARCHITECTURE.md) |
| `capabilities` | Pack vs wiring. `--offline` uses the cache. `--json` |
| `diagnose` | 9-frame hull, then `sec/step`. `--prepare` lints only. Does not spend shift budget |
| `budget status` / `budget reset-shift` | VRAM-minute ledger. Reset archives history |
| `fetch-object-info` | Cache `/object_info` to `state/` |
| `scan-models` | Scan `models/` into the inventory. `--strict` exits 1 if a bundle is not runnable |
| `validate [file]` | One workflow, or `--all`. `--offline`. `--strict` makes illegal LTX frames an error |

## Generate

```bash
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent comfy run --mode template --template wan22 --prepare
python -m master_agent comfy run --mode raw --json workflow.json
python -m master_agent run "rain on a window" --quality draft --duration 3 --no-interview
python -m master_agent run "she says the line" --image face.png --audio line.wav --no-interview
python -m master_agent brief "rough idea"
python -m master_agent power-tune "neon rain" --variant base --json
```

`comfy run` modes are `generate`, `template`, and `raw`. `--prepare` stops before the queue. `--set NODE.FIELD=VALUE` is the expert override.

`run` is the director pipeline. Useful flags: `--variant`, `--dry-run` (plan and lint, no GPU), `--self-improve-dry` (judge loop, no Comfy), `--no-judge`, `--no-interview`, `--storyboard`, `--llm-panel`, `--panel-judge`, `--attach`, `--upscale seedvr2|rtx`, `--image`, `--video`, `--audio`, `--mask`, `--outpaint`, `--aspect`, `--line`, and the lipdub flags in [Features](FEATURES.md#lipdub). H3 voice warning is stated once in [Features](FEATURES.md#default-catalog).

`comfy` subcommands: `run`, `attach`, `ingest`, `learn`, `dry-run`, `promote`, `start`, `stop`, `status`, `restart`, `update`. `run --ingested SLUG` queues a learned graph. `ingest` takes a JSON path or `--from history:PROMPT_ID`, stores under `state/ingested/`, and does not queue; the next command is dry-run. `--llm-assist` is optional and local. `promote` writes a local draft manifest entry. Procedure: [Workflow ingest](WORKFLOW_INGEST.md). Process ownership and the attach schema: [Comfy](COMFY.md).

## LLM provider

`LLM_PROVIDER` defaults to `auto` when unset. Order: llama.cpp, then Ollama. `auto` never calls a cloud model; Grok runs only with `LLM_PROVIDER=grok` or a panel that names it. llama.cpp is the preferred local backend. `ollama` or `llamacpp` pins that backend and does not hop.

| Env | Default |
|---|---|
| `LLAMACPP_URL` | `http://127.0.0.1:8080` |
| `LLAMACPP_MODEL` | `OLLAMA_MODEL` (`qwen3-vl-heretic`) |
| `LLAMACPP_BIN` | `llama-server` on `PATH`, then Buddy's own build in `LLAMACPP_ROOT` |
| `LLAMACPP_ROOT` | `video_buddy/llama.cpp` (automatic install builds or unpacks here) |
| `LLAMACPP_REF` | `b11389` (pinned ggml-org/llama.cpp release tag) |
| `LLAMACPP_AUTOSTART` | `1`. `0` (or `false`/`no`/`off`) never starts `llama-server`; Buddy only uses one already listening on `LLAMACPP_URL` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen3-vl-heretic` |

Buddy starts `llama-server --models-dir $MODELS_DIR` (default `video_buddy/models/`) and stops that process on shutdown. A missing binary warns and auto falls through to Ollama. `health` prints llama.cpp, Ollama, and Grok separately. See [Architecture](ARCHITECTURE.md#local-model).

## Models

```bash
python -m master_agent download-models --ltx23
python -m master_agent download-models --ltx25
python -m master_agent download-models --h3
python -m master_agent download-models --heartmula
python -m master_agent download-models --wan
python -m master_agent models manifest
python -m master_agent models select --version 2.5 --scan-only
python -m master_agent models register <file-or-folder> [--dry-run] [--mode auto|hardlink|symlink|copy] [--force] [--json]
python -m master_agent download-flux
```

Also `--vace`, `--krea`, `--qwen`, `--flux-pack`, `--bundle`, `--optional`, `--selector`. Nothing is fetched until you agree. See [Weights](WEIGHTS.md#consent). `--ltx23` is the LTX 2.3 pack that automatic install downloads (distilled 1.1 all-in-one, GGUF Q4_K_S, Gemma fp4 text encoder, distilled LoRA, video VAE, tiny VAE).

## Agent entry, identity, knowledge

| Command | Role |
|---|---|
| `agent <tool> --args JSON` | The gateway entry. One JSON document on stdout, logs on stderr. `agent list` prints the tools. Exit 1 on `status` error/busy, 2 on a bad call |
| `fallback-mcp` | Temporary, dormant MCP stdio fallback forwarding to the `agent` tools. Off unless `BUDDY_FALLBACK_MCP=1` and `requirements-fallback-mcp.txt` is installed. Removal after 2026-11-02: [Revert / fallback path](ARCHITECTURE.md#revert--fallback-path) |
| `persona list\|show\|set` | Interview voice |
| `soul list\|show\|set` | Standing values |
| `kb ingest\|search\|stats` | Local Chroma. `--knowledge` searches the git folder. `--workflows` searches digests |

### Hermes

```bash
cd video_buddy
python install_hermes_skill.py --profile ltx   # → ~/.hermes/profiles/ltx/skills/media/video-buddy/
python install_hermes_skill.py                 # → ~/.hermes/skills/video-buddy/   (--dest DIR for any other folder)
python -m master_agent agent list
python -m master_agent agent comfy_start       # detached; never run ComfyUI main.py in a foreground terminal
python -m master_agent agent create_video --args '{"request": "neon rain", "dry_run": true}'
```

The installer copies `SKILL.md`, `TOOLS.md`, `PROFILE.md` and `references/` only. Skill source: `video_buddy/skills/video-buddy/`. Agent tools (`master_agent/agent_api.py`): `about`, `health`, `comfy_status`, `comfy_start`, `comfy_restart`, `create_video`, `plan_storyboard`, `judge_asset`, `search_workflows`, `search_runs`, `search_knowledge`, `kb_ingest`, `list_runs`, `list_models`, `validate_workflow`, `create_character`, `train_lora`, `control_get`, `control_set`, `budget_status`, `budget_reset_shift`. Diagnose, curriculum, `comfy run`, doctor and ingest/promote are CLI commands.

`comfy_start` runs the same code as `comfy start --as-json` and returns within `timeout_s` (default 180 s): `started`, `starting` (poll `comfy_status`), or `error` with the launch log tail. Gateway setup and old-config cleanup: `video_buddy/skills/video-buddy/PROFILE.md`.

## Feature commands

| Command | Role |
|---|---|
| `fractal` | CPU zoom, inpaint, or outpaint |
| `music` | Beat-synced MV. `--visual shots\|fractal` |
| `mv plan` | `buddy.mv.beat_plan/v1` from `--audio` |
| `mv render` | Comfy/LTX burns → Remotion. `--dry-run` needs no GPU |
| `heartmula generate\|transcribe` | heartlib. `--dry-run` does not import it |
| `character create\|list` | CCC sheet. `--train` chains LoRA |
| `lora setup\|train\|validate` | Separate ai-toolkit venv |

Details: [Features](FEATURES.md).
