# Video Buddy — agent tools + CLI inventory

Authoritative lists. Do not invent names. Agent tools come from
`master_agent/agent_api.py` (`python -m master_agent agent list`). CLI commands
come from `python -m master_agent` (dispatcher `master_agent/__main__.py`,
commands in `master_agent/cli/`).

Cwd for CLI: the `video_buddy/` directory. Use the project venv.

## Agent tools (`python -m master_agent agent <tool> --args '<json>'`)

The LTX bot gateway is the only caller. stdout is one JSON document; logs go
to stderr. Exit 1 when the result `status` is `error` or `busy`; exit 2 for an
unknown tool or bad `--args`. Python callers: `from master_agent import agent_api`.

| Tool | Arguments | What it does |
|---|---|---|
| `about` | *(none)* | Studio identity card. |
| `health` | *(none)* | ComfyUI reachability + GPU VRAM, Ollama up, llama.cpp up (`local_llm`), opt-in Grok credentials (read-only), KB counts, config hash + values. |
| `create_video` | `request: str`, `duration_s: float\|None=None` (5 s, or follows the audio for photo + voice), `quality: str="draft"`, `variant: str\|None=None`, `llm_panel: str\|None=None`, `dry_run: bool=False`, `image_path`, `audio_path`, `video_path`, `line: str\|None=None`, `seed: int\|None=None`, `storyboard: str\|None=None` (smart\|always\|multi_only\|off), `upscale: str\|None=None` (rtx\|seedvr2) | Full director pipeline (route, storyboard, per-segment judge, stitch, full judge). `dry_run=True` plans + validates, no GPU. `status="busy"` when another render holds the GPU (`state/gpu.lock`); `status="paused"` when the render budget held it. |
| `plan_storyboard` | `request: str`, `duration_s: float=8.0`, `quality: str="draft"`, `llm_panel: str\|None=None` | Segment split + LLM-panel storyboard with KB recall. **No GPU, no validation.** |
| `judge_asset` | `video_path: str`, `request: str=""`, `full_video: bool=False` | Grade an existing file (heuristics + text LLM + vision). `full_video=True` uses the full-video judge. |
| `search_workflows` | `query: str`, `k: int=3` | Semantic search over the workflow KB. |
| `search_runs` | `query: str`, `k: int=3` | Semantic search over past run records. |
| `search_knowledge` | `query: str`, `k: int=3` | Semantic search over git-synced `knowledge/`. |
| `kb_ingest` | *(none)* | Bulk-load workflows, run records and `knowledge/` into the KB. |
| `list_runs` | `limit: int=50` | Newest run records: request, variant, status, judge score, video path. |
| `list_models` | *(none)* | Local inventory: bundles, files (name, size, dir), summary text. |
| `validate_workflow` | `path: str` | Validate one workflow JSON against the live/cached node registry + local models. |
| `create_character` | `description: str`, `name: str=""`, `shots: int=0`, `train: bool=False` | CCC: bible → Flux sheet → captioned dataset. `train=True` chains Flux LoRA (hours on 16GB). |
| `train_lora` | `character_name: str`, `steps: int=0`, `lr: float=0.0`, `rank: int=0`, `validate: bool=True` | Train Flux LoRA for an existing character (ai-toolkit, resumable). |
| `control_get` | *(none)* | Studio knobs (judge strictness/threshold, learning rate, VRAM cost gate, budget cap/used, persona, soul) + config history. |
| `control_set` | any of `judge_strictness`, `learning_rate`, `cost_vram_threshold_gb`, `render_budget_cap_vram_min`, `judge_score_threshold`, `persona`, `soul`; `reset_budget: bool=False`; `session: str="agent"` | Versioned, hash-stamped knob change. `reset_budget=True` zeroes used VRAM-minutes and resumes budget-paused generates. |
| `budget_status` | *(none)* | Shift budget snapshot: used, cap, paused, shift id, pending, ledger. |
| `budget_reset_shift` | *(none)* | Archive the shift ledger, zero used, resume budget-paused generates. |

Latency: `health` / `search_*` / `validate_workflow` / `kb_ingest` ~seconds.
`judge_asset` and `create_video` can take minutes (cold local LLM). `train_lora`
is hours. Allow ~900s per call.

## CLI (`python -m master_agent <cmd>`)

| Command | What it does |
|---|---|
| `agent <tool> --args '<json>'` | Call an agent tool (table above) and print one JSON document on stdout. `--args @file.json` reads a file. `agent list` prints every tool with its parameters. |
| `about` | Studio identity card (`--json` ok). |
| `curriculum` | Print L0→L5 + Part 2 overnight gate (`--json` ok). |
| `inventory` | List discovered weights (path + role) before doctor. `--json` optional. Consent rules: repo `docs/WEIGHTS.md`. |
| `setup` / `doctor` | Deps + LTX 2.5 / H3 / LTX 2.3 scan. Flags and the loader order: repo `docs/WEIGHTS.md`. `--fix` installs deps; Ollama pull skips models already in `ollama list` unless you pass `--yes`. |
| `health` | Comfy `:8188` + GPU stats. |
| `workflows` | Default catalog (includes `ltx25_*` and `h3_*`). `--vram` prints 16GB pack table. |
| `capabilities` | Comfy pack vs Buddy wiring. `--offline` uses cache. |
| `diagnose` | 9-frame hull fire; prints `sec/step`; no shift-budget spend. `--prepare` lints only. |
| `comfy run` | Prepare, lint, queue, copy into `outputs/`. `--prepare` stops before GPU. Modes: `generate` / `template` / `raw`. |
| `comfy attach` | Apply previs `buddy.comfy.attach/v1` / WorkflowPatchPlan JSON. Default dry-run (patch + `/object_info`). `--submit` POSTs `/prompt`. |
| `run "BRIEF"` | Director pipeline. `--dry-run` plan+lint only. `--self-improve-dry` closes judge→revise→rejudge (quality_bar a/c/d, no Comfy). `--attach RECIPE.json` patches a previs pack. `--no-interview` for unattended. `--variant` forces a catalog slug. Every clip writes `shot-N.buddy.json` + run-row ClipProvenance. |
| `download-models` | List confirmed-missing slots. `--ltx25` / `--h3` / `--heartmula` / `--wan` / … HeartMuLa codec id and consent flags: repo `docs/WEIGHTS.md`. |
| `download-flux` | One-time Flux fp8 weights (~17GB). |
| `validate` | One file or `--all`. `--offline` / `--strict` (illegal LTX frames = ERROR). |
| `scan-models` | Scan `models/` → inventory. |
| `fetch-object-info` | Cache Comfy `/object_info` to `state/`. |
| `budget status` / `budget reset-shift` | VRAM-min shift ledger. Reset archives history; never wipes. |
| `kb ingest` / `kb search` / `kb stats` | Local RAG. `search` defaults to runs; `--workflows` for templates; `--knowledge` for git-synced `knowledge/` (failures recalled first). |
| `persona list\|show\|set` | Interview voice (`ara`, `exec`, `zod`). |
| `soul list\|show\|set` | Standing values (`studio`, `play`). |
| `brief "idea"` | Interview only. `--go` chains into generation. |
| `fractal` | CPU Mandelbrot/Julia. No Comfy. |
| `music "BRIEF" --audio FILE` | Beat-synced MV (ffmpeg mux). `--visual fractal` is CPU-only. `--audio` wins over `--heartmula-lyrics` / `--heartmula-tags`. |
| `mv plan --audio FILE` | Write `buddy.mv.beat_plan/v1` (30 fps windows). |
| `mv render --audio FILE --out out/MV-FIXED.mp4` | Comfy/LTX unique burns → Remotion stitch. `--dry-run` = no GPU. `--image` = I2V. Optional HeartMuLa track when `--audio` is omitted. |
| `heartmula generate --lyrics TEXT --tags TAGS --out track.wav` | heartlib plan. `--dry-run` writes nothing and does not import heartlib. `--max-seq-len` / `--low-vram` size the backbone KV window (16GB cannot use heartlib's 8192). Wav save falls back to soundfile when torchaudio/torchcodec fails. |
| `heartmula transcribe --audio FILE --out words.json` | `{w,s,e}` for lipdub. Skipped when `run --words` is set. |
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
python -m master_agent run "music video for a synthwave track" --self-improve-dry --no-interview
python -m master_agent run "LTX 2.5 alley" --variant ltx25_t2v_i2v --no-interview
python -m master_agent run "talking head" --video input.mp4 --variant lipsync --no-interview
python -m master_agent run "she says the line" --image face.png --audio line.wav --no-interview
# H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.
python -m master_agent run "The ringmaster clown speaks directly to camera, lips synced to the voice" --image face.png --audio sample.wav --variant h3_r2v --line "Hey there, Keep Local AI runs on your own machine." --no-interview
python -m master_agent music "synthwave MV" --audio track.mp3 --no-interview
python -m master_agent mv render "I'm in love with a bot" --audio track.mp3 --out out/MV-FIXED.mp4
python -m master_agent mv render --audio track.mp3 --dry-run
python -m master_agent fractal "title" --duration 20 --target seahorse
```

## Variants (default catalog)

LTX 2.5 (no env flag): `ltx25_t2v_i2v` (`t2v_i2v`), `ltx25_t2v_i2v_two_stage`,
`ltx25_flf2v` (`flf2v`), `ltx25_msr` (`msr`), `ltx25_v2v_ic_lora`,
`ltx25_a2v`, `ltx25_t2a`.

MiniMax H3: `h3_t2v` / `h3_i2v` / `h3_flf` (fl2va), `h3_r2v` (ref2va).
Aliases `fl2va` / `ref2va`. CFG 1.0. Do not break LTX 2.5 / WAN paths.
H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.

Still-working: `base`, `eros`, `directors`, `lipsync`, `wan22`, `flux`,
`vb_movie_builder`, CCC / renderer / AI-VFX slugs. List: `python -m master_agent workflows`.

## Pack C (hard local-only)

Local-only is a hard requirement. Generate on `http://127.0.0.1:8188`.

`seedance25_draft_t2v` / `seedance25_draft_i2v` / `seedance25_draft_r2v` are
field-shape records of `api_seedance2_5_draft_{t2v,i2v,r2v}`. They are not
executable. A Seedance brief fail-closes onto `ltx25_t2v_i2v`, `ltx25_flf2v`,
or `ltx25_msr`. No comfy.org, BytePlus, ModelArk, or KIE call.
Doc: repo `docs/FEATURES.md` (Pack C).
