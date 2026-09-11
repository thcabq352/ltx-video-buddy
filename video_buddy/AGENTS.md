# Video Buddy — agent notes

Local ComfyUI video studio. Code lives in this directory. Git root is the parent `ltx2.3_agent` repo (`thcabq352/ltx-video-buddy`).

## Install

From this folder, one command on any OS:

```powershell
python install.py
```

Windows: `install.bat`. macOS / Linux: `./install.sh`. Re-check with `python -m master_agent setup`; install missing pieces with `--fix`.

## About

`python -m master_agent about` prints the studio card (also `GET /api/about` and the studio About tab). VIDEO BUDDY is the local ComfyUI studio; package `master_agent`; MCP `master-agent`.

## Run

Drive Comfy from the CLI first. Studio `:8189` is optional.

```powershell
.\.venv\Scripts\python.exe -m master_agent health
.\.venv\Scripts\python.exe -m master_agent diagnose --variant base --prompt "garden proof"
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
.\.venv\Scripts\python.exe -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
.\.venv\Scripts\python.exe -m master_agent ui --port 8189
```

`comfy run` prepares, lints, queues, and copies into `outputs/`. `--prepare` stops before the GPU queue. Comfy portable is `:8188` (`ComfyUI_windows_portable\run_api_8188.bat`). Studio is `:8189`. Do not treat a live studio tab as proof Comfy is up.

Diagnose the hull before any scale. This is a 9-frame short fire (steps 6–8, fixed seed), not a 121-frame burn:

```powershell
.\.venv\Scripts\python.exe -m master_agent diagnose --variant base --prompt "garden proof"
```

It always prepares + lints first, then queues the short fire, prints wall time and `sec/step`, copies into `outputs/`, and ffprobes the file. Junk (`<100KB` or `<3` frames) is FAIL. Diagnose does **not** spend shift budget. `--prepare` stops after lint. Do not walk `DOWNSCALE_LADDER` or raise resolution until the hull has recorded `sec/step`. `--dry-run` on `run` stays plan-only (no GPU, no used increment).

```powershell
.\.venv\Scripts\python.exe -m master_agent budget status
.\.venv\Scripts\python.exe -m master_agent budget reset-shift
```

## Stop-lines (Jason Rainey fleet)

- **LTX frame law.** Valid lengths are `8n+1`, minimum **9**. `snap_ltx_frames()` never returns 8. The patcher writes `EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number` together. Validator WARNs and auto-corrects unless `--strict` (ERROR). `DEFAULT_FRAMES`, diagnose, and draft start at 9. Long-run templates keep their length but still snap illegal counts. `DOWNSCALE_LADDER` first rungs are 9/17/25/33 — not 121.
- **Diagnose before scale.** Measure `sec/step` on a 9-frame hull before raising res/frames. Scale is refused until `state/control/diagnose_hull.json` has `sec_per_step`.
- **TeaCache bypass.** Missing optional accelerators (`TeaCache`, `WanVideoTeaCache`, `WanVideoTeaCacheKJ`, …) are WARNING + bypass: MODEL (or the typed upstream link) is rewired past the node and the node is dropped. Validation still PASSes. Do not auto-install packs. Live Comfy may only expose the Wan TeaCache aliases; a generic LTX `TeaCache` must still bypass.
- **Judge look vs health.** Payload has `look_score`, `health_score`, and `combined_score` (back-compat). Retry ladder uses **look only**. Low `brief_adherence` + high look is `human_veto` / `HUMAN_VETO` — no album-lock. Probe `min_frames=3`; junk `<100KB` is a health fail.
- **Shift reset.** `RenderBudget.reset_shift()` archives `{event:shift_reset, previous_used, previous_shift_id}` then `used=0` `paused=False`. Never wipe the ledger. Diagnose/dry-run do not increment `used`. HOLD is A2A `input-required`, not `failed`.
- **LoRA A/B.** When comparing LoRAs, lock encoder + seed family; only LoRA name/strength may change. Do not swap `gemma_3_12B_it_fp8_scaled` / the default TE for Heretic mid-A/B. Windows folder-prefixed checkpoint names stay backslash style.

## Identity

Persona is the interview voice. Soul is standing studio values.

- Bundled personas: `ara`, `exec`, `zod` — override in `state/personas/<slug>.md`
- Bundled souls: `studio`, `play` — override in `state/souls/<slug>.md`
- Switch: `python -m master_agent persona set <slug>` / `soul set <slug>`, Create-tab selects, or `PERSONA` / `SOUL` env
- Changes go through versioned config (`state/control/config.json` + history)

## Local model

Default Ollama / vision model is `qwen3-vl-heretic`. Auto LLM chain is `ollama → grok`. Do not add a Kimi provider.

## Control layer

Render budget (VRAM-minutes) can pause the queue over cap. Each shift has `shift_id`; `budget reset-shift` archives the previous used total and zeros the counter without wiping history. `/api/control` exposes knobs, budget, and last-10 config history. A2A lives at `GET /.well-known/agent.json` and `POST /a2a` (studio port **8189**). Budget-held pipeline jobs map to A2A `input-required`, not `failed`.

## Tests

Use the project venv. Prefer one file at a time for GPU-adjacent work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ltx_frames.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_accelerator_bypass.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_judge_split.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_shift_budget.py -q
```

Do not commit models, `state/runs`, Comfy portable, or sibling trees (`ltx_director/`, `SOS/`, `lot/`).
