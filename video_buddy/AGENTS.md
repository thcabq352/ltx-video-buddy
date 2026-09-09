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
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --prompt "BRIEF" --variant base
.\.venv\Scripts\python.exe -m master_agent run "BRIEF" --quality draft --duration 3 --no-interview
.\.venv\Scripts\python.exe -m master_agent ui --port 8189
```

`comfy run` prepares, lints, queues, and copies into `outputs/`. `--prepare` stops before the GPU queue. Comfy portable is `:8188` (`ComfyUI_windows_portable\run_api_8188.bat`). Studio is `:8189`. Do not treat a live studio tab as proof Comfy is up.

## Identity

Persona is the interview voice. Soul is standing studio values.

- Bundled personas: `ara`, `exec`, `zod` — override in `state/personas/<slug>.md`
- Bundled souls: `studio`, `play` — override in `state/souls/<slug>.md`
- Switch: `python -m master_agent persona set <slug>` / `soul set <slug>`, Create-tab selects, or `PERSONA` / `SOUL` env
- Changes go through versioned config (`state/control/config.json` + history)

## Local model

Default Ollama / vision model is `qwen3-vl-heretic`. Auto LLM chain is `ollama → grok`. Do not add a Kimi provider.

## Control layer

Render budget (VRAM-minutes) can pause the queue over cap. `/api/control` exposes knobs, budget, and last-10 config history. A2A lives at `GET /.well-known/agent.json` and `POST /a2a` (studio port **8189**). Budget-held pipeline jobs map to A2A `input-required`, not `failed`.

## Tests

Use the project venv. Prefer one file at a time for GPU-adjacent work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_persona.py -q
```

Do not commit models, `state/runs`, Comfy portable, or sibling trees (`ltx_director/`, `SOS/`, `lot/`).
