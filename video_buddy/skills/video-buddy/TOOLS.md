# Video Buddy — MCP + CLI inventory

Authoritative lists. Do not invent names. MCP tools come from
`master_agent/mcp_server.py`. CLI commands come from
`python -m master_agent` (`master_agent/__main__.py`).

Cwd for CLI: the `video_buddy/` directory. Use the project venv.

## MCP tools (`master-agent`)

Server: FastMCP `"master-agent"`. Hermes spawns
`<venv python> master_agent/mcp_server.py` over stdio. Stdout inside tools is
redirected to stderr so the protocol channel stays clean.

| Tool | Arguments | What it does |
|---|---|---|
| `health` | *(none)* | ComfyUI reachability + GPU VRAM, Ollama up, llama.cpp up (`local_llm`), KB counts, config hash. |
| `create_video` | `request: str`, `duration_s: float=5.0`, `quality: str="draft"`, `variant: str\|None=None`, `llm_panel: str\|None=None`, `dry_run: bool=False` | Full director pipeline (route, storyboard, per-segment judge, stitch, full judge). `dry_run=True` plans + validates, no GPU. |
| `plan_storyboard` | `request: str`, `duration_s: float=8.0`, `quality: str="draft"`, `llm_panel: str\|None=None` | Segment split + LLM-panel storyboard with KB recall. **No GPU, no validation.** |
| `judge_asset` | `video_path: str`, `request: str=""`, `full_video: bool=False` | Grade an existing file (heuristics + text LLM + vision). `full_video=True` uses the full-video judge. |
| `search_workflows` | `query: str`, `k: int=3` | Semantic search over the workflow KB. |
| `search_runs` | `query: str`, `k: int=3` | Semantic search over past run records. |
| `kb_ingest` | *(none)* | Bulk-load workflows + all run records into the KB. |
| `list_models` | *(none)* | Local inventory summary (LTX weights, bundles, runnable state). |
| `validate_workflow` | `path: str` | Validate one workflow JSON against the live/cached node registry + local models. |
| `create_character` | `description: str`, `name: str=""`, `shots: int=0`, `train: bool=False` | CCC: bible → Flux sheet → captioned dataset. `train=True` chains Flux LoRA (hours on 16GB). |
| `train_lora` | `character_name: str`, `steps: int=0`, `lr: float=0.0`, `rank: int=0`, `validate: bool=True` | Train Flux LoRA for an existing character (ai-toolkit, resumable). |

Latency: `health` / `search_*` / `validate_workflow` / `kb_ingest` ~seconds.
`judge_asset` and `create_video` can take minutes (cold local LLM). `train_lora`
is hours. Prefer ~900s MCP timeout.

## CLI (`python -m master_agent <cmd>`)

| Command | What it does |
|---|---|
| `about` | Studio identity card (`--json` ok). |
| `curriculum` | Print L0→L5 + Part 2 overnight gate (`--json` ok). |
| `setup` / `doctor` | Deps + LTX 2.5 / H3 scan. **Does not fetch weights.** `--fix` installs deps. `--fix-models` only after you agree. |
| `health` | Comfy `:8188` + GPU stats. |
| `workflows` | Default catalog (includes `ltx25_*` and `h3_*`). `--vram` prints 16GB pack table. |
| `capabilities` | Comfy pack vs Buddy wiring. `--offline` uses cache. |
| `diagnose` | 9-frame hull fire; prints `sec/step`; no shift-budget spend. `--prepare` lints only. |
| `comfy run` | Prepare, lint, queue, copy into `outputs/`. `--prepare` stops before GPU. Modes: `generate` / `template` / `raw`. |
| `run "BRIEF"` | Director pipeline. `--dry-run` plan+lint only. `--no-interview` for unattended. `--variant` forces a catalog slug. |
| `download-models` | List confirmed-missing slots. `--ltx25` / `--h3` / `--wan` / … Add `--yes` only after the ask. |
| `download-flux` | One-time Flux fp8 weights (~17GB). |
| `validate` | One file or `--all`. `--offline` / `--strict` (illegal LTX frames = ERROR). |
| `scan-models` | Scan `models/` → inventory. |
| `fetch-object-info` | Cache Comfy `/object_info` to `state/`. |
| `budget status` / `budget reset-shift` | VRAM-min shift ledger. Reset archives history; never wipes. |
| `hermes status` / `hermes register` | Discover ltx seat (Hermes / buddy-adapter / A2A fallback). Register seats `~/.hermes/profiles/ltx/` (no `.env`). |
| `ui` | Studio dashboard, default `:8189`. |
| `kb ingest` / `kb search` / `kb stats` | Local RAG. `search` defaults to runs; `--workflows` for templates. |
| `persona list\|show\|set` | Interview voice (`ara`, `exec`, `zod`). |
| `soul list\|show\|set` | Standing values (`studio`, `play`). |
| `brief "idea"` | Interview only. `--go` chains into generation. |
| `fractal` | CPU Mandelbrot/Julia. No Comfy. |
| `music "BRIEF" --audio FILE` | Beat-synced MV. `--visual fractal` is CPU-only. |
| `character create\|list` | CCC stage. `--train` chains LoRA. |
| `lora setup\|train\|validate` | ai-toolkit Flux LoRA. |
| `power-tune` | Dry-run power mode: patch + LLM graph ops + validate, no GPU. |

### `comfy run` examples

```bash
python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant ltx25_t2v_i2v
python -m master_agent comfy run --mode generate --prompt "BRIEF" --variant h3_t2v
python -m master_agent comfy run --mode template --template wan22 --prepare
python -m master_agent comfy run --mode template --template lipsync --prepare --out prepared.json
python -m master_agent comfy run --mode raw --json workflow.json
python -m master_agent comfy run --mode template --template base --set 12.steps=8
```

Templates: every `workflows/manifests.yaml` slug (`base`, `eros`, `directors`,
`lipsync`, `wan22`, `flux`, `vb_aivfx_adv`, `vb_movie_builder`, every
`ltx25_*` / research aliases, `h3_t2v` / `h3_i2v` / `h3_flf` / `h3_r2v` with
aliases `fl2va` / `ref2va`, or a path under `workflows/`).

### `run` examples

```bash
python -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
python -m master_agent run "BRIEF" --duration 8 --dry-run --no-interview
python -m master_agent run "LTX 2.5 alley" --variant ltx25_t2v_i2v --no-interview
python -m master_agent run "talking head" --video input.mp4 --variant lipsync --no-interview
python -m master_agent music "synthwave MV" --audio track.mp3 --no-interview
python -m master_agent fractal "title" --duration 20 --target seahorse
```

## Variants (default catalog)

LTX 2.5 (no env flag): `ltx25_t2v_i2v` (`t2v_i2v`), `ltx25_t2v_i2v_two_stage`,
`ltx25_flf2v` (`flf2v`), `ltx25_msr` (`msr`), `ltx25_v2v_ic_lora`,
`ltx25_a2v`, `ltx25_t2a`.

MiniMax H3: `h3_t2v` / `h3_i2v` / `h3_flf` (fl2va), `h3_r2v` (ref2va).
Aliases `fl2va` / `ref2va`. CFG 1.0. Do not break LTX 2.5 / WAN paths.

Still-working: `base`, `eros`, `directors`, `lipsync`, `wan22`, `flux`,
`vb_movie_builder`, CCC / renderer / AI-VFX slugs. List: `python -m master_agent workflows`.
