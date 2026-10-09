# Getting started

From `video_buddy/`. Python 3.10+. ComfyUI listens on `:8188`.

Weights, consent, and the loader order have one home: [Weights](WEIGHTS.md#loader-policy). This page does not restate them.

## Install

`install.py` creates `.venv`, installs pip deps, Playwright Chromium, copies `.env`, and tries ffmpeg plus the Ollama models `qwen3-vl-heretic` and `nomic-embed-text`. It does not download video weights.

| OS | Command |
|---|---|
| Windows | `install.bat` or `python install.py` |
| macOS / Linux | `./install.sh` or `python3 install.py` |

```bash
cd video_buddy
python install.py
python -m master_agent doctor
python -m master_agent workflows
python -m master_agent health
```

`setup` is `doctor`. `setup --fix` installs missing deps (ffmpeg via winget, brew, or apt-get when that tool exists). Ollama pulls only models missing from `ollama list`, and only after `y` or `--yes`. Ollama itself is installed from https://ollama.com/download.

Local LLM order for `LLM_PROVIDER=auto` (the default when unset): llama.cpp, then Ollama. `auto` never calls a cloud model; Grok is opt-in with `LLM_PROVIDER=grok`. llama.cpp is preferred. Buddy starts `llama-server` against `MODELS_DIR` when `LLAMACPP_BIN` or `llama-server` is available, and stops it on shutdown. If that binary is missing, Buddy warns and uses Ollama. Pin `LLM_PROVIDER=llamacpp` or `LLM_PROVIDER=ollama` to force one backend. See [Architecture](ARCHITECTURE.md#local-model).

Windows portable Comfy: `ComfyUI_windows_portable\run_api_8188.bat`. Elsewhere start Comfy with `--port 8188`, or set `COMFYUI_URL`.

```bash
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent agent health
```

Hermes: `python install_hermes_skill.py` copies `skills/video-buddy/` to `~/.hermes/skills/video-buddy/`. Nothing else is written. The LTX bot gateway calls `python -m master_agent agent <tool>`. Buddy binds no gateway port and never 8642. See [CLI reference](CLI_REFERENCE.md#hermes).

## Curriculum

Print the card: `python -m master_agent curriculum`.

### Part 1 — `LESSON_BUDDY_WORKS_HERE` (L0→L5)

Do these in order. Do not jump to Part 2.

| Lesson | Name | Do |
|---|---|---|
| L0 | tree | This tree is `video_buddy` / `master_agent`. |
| L1 | about | `python -m master_agent about` before any GPU claim. |
| L2 | health | `python -m master_agent health`. Comfy **:8188** up. A gateway reply is not proof. |
| L3 | dry-run | `python -m master_agent run "BRIEF" --dry-run`. Plan and lint only. No queue. No shift-budget spend. |
| L4 | diagnose | `python -m master_agent diagnose --variant base --prompt "garden proof"`. 9-frame hull. Print `sec/step`. |
| L5 | short proof | A real file in `outputs/`. `ffprobe` frames and size. Junk under 100KB or under 3 frames is FAIL. |

### Part 2 — `LESSON_BUDDY_DURATION_LADDER`

Gated on Part 1 L5 and a human OK. This is the 1s→10s ladder (25→241 frames, `8n+1`). Do not start it until a 9-frame proof exists.

Diagnose before any scale. The hull is steps 6–8, not a 121-frame burn. Do not walk `DOWNSCALE_LADDER` or raise resolution until `state/control/diagnose_hull.json` has `sec_per_step`. Diagnose does not spend shift budget. `--prepare` stops after lint.

```bash
python -m master_agent budget status
python -m master_agent budget reset-shift
```

`budget reset-shift` archives the previous used total. It does not wipe history.

## First generate

```bash
python -m master_agent inventory
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain" --prepare
python -m master_agent run "cinematic close-up of rain on a window" --quality draft --duration 3 --no-interview
```

`--prepare` lints and writes JSON. It does not queue the GPU. Photo plus a voice file defaults to `ltx25_a2v`. Catalog ids and the H3 voice warning: [Features](FEATURES.md).

12GB cards (RTX 4000 Ada) and fully local LLM pins are in [Weights](WEIGHTS.md#12gb-and-local-llm).
