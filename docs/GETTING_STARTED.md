# Getting started

From `video_buddy/`. Python 3.10+. ComfyUI listens on `:8188`.

Weights, consent, and the loader order have one home: [Weights](WEIGHTS.md#loader-policy). This page does not restate them.

## Automatic install

New to all of this? Use automatic install. It is one command. It takes an empty machine to a working LTX 2.3 video generate. You only need Python 3.10+ and an internet connection.

| OS | Command (from the `video_buddy` folder) |
|---|---|
| Windows | `install.bat --automatic-install --yes` |
| macOS / Linux | `./install.sh --automatic-install --yes` |

Already have `.venv`? `python -m master_agent automatic-install --yes` does the same thing.

What it installs:

- ffmpeg
- comfy-cli at the pin in `requirements.txt` (`comfy-cli==1.20.0`)
- ComfyUI in its own folder (`MANAGED_COMFY_ROOT`, default `video_buddy/ComfyUI`), with the PyTorch build for your GPU. Buddy only ever uses this ComfyUI.
- The two custom node packs the LTX 2.3 graphs need: `ComfyUI-LTXVideo` and `ComfyUI-GGUF`
- Triton and SageAttention inside ComfyUI's own venv (optional; see [Comfy](COMFY.md#triton-and-sageattention))
- The LTX 2.3 weights, about 76.5 GB, into `MODELS_DIR` only (never into the ComfyUI folder)
- llama.cpp for the local LLM (optional; see [llama.cpp](#llamacpp) below)

LTX 2.3 is the only video model automatic install sets up. Other packs stay opt-in through `download-models`. Automatic install never installs Ollama or pulls anything through it.

### Pre-flight

The pre-flight runs first, before anything is downloaded or installed. It prints one PASS, FAIL, or INFO line per check in plain English. It stops at the first FAIL, says how to fix it, and changes nothing.

| Check | Kind | What it looks at |
|---|---|---|
| Disk space | stops on FAIL | Free space on each drive that will hold the LTX 2.3 weights (`MODELS_DIR`), the ComfyUI folder, and llama.cpp (`LLAMACPP_ROOT`). Uses the real file sizes of what is still missing, plus 5 GB headroom. On an empty machine that is about 97–99 GB (llama.cpp takes 0.3–5 GB depending on build or prebuilt, CPU or CUDA). Prints free vs needed, and how to free space or move a folder to a bigger drive. |
| Hugging Face access | stops on FAIL | The LTX 2.3 files are public, so no token is needed. If `HF_TOKEN` is set, it is checked with a lightweight sign-in call, because a bad token breaks even public downloads. The token is never printed. |
| GPU / VRAM | information only | Reads `nvidia-smi`. Under 14 GB VRAM, the GGUF Q4 loader is used. No GPU means CPU mode: it works, but is very slow. |
| llama.cpp | information only | One of: found (`LLAMACPP_BIN`, `PATH`, or Buddy's own copy), will build from source, will use the official prebuilt release (and which build tool is missing), or unavailable (with the command that installs the build tools). Also lists each local LLM model that will be skipped. |
| ComfyUI folder | stops on FAIL | The folder must be writable and either empty, a ComfyUI, or missing. Notes an existing install (that step is then skipped). Refuses when `COMFY_MODE=external`. |

```bash
python -m master_agent automatic-install --preflight-only   # pre-flight only, changes nothing
python -m master_agent automatic-install                    # pre-flight + the plan, changes nothing
python -m master_agent automatic-install --yes              # pre-flight, then install
```

Other flags: `--gpu auto|nvidia|amd|m-series|cpu` picks the ComfyUI PyTorch build (default: detect). `--skip-weights` installs everything but the weights. `--skip-sage` skips Triton and SageAttention. `--skip-llm` skips llama.cpp and the local LLM models.

### If something fails

- A pre-flight FAIL stops the run before any change. Fix the item and run the same command again.
- Running it again is safe. Finished steps are skipped. A half-finished ComfyUI is repaired with `comfy install --restore`.
- Buddy never uses `sudo` silently. On Linux, when ffmpeg needs administrator rights and `sudo` would ask for a password, the step shows `NEEDS-YOU` with the exact command to run yourself. The same goes for the llama.cpp build tools.
- Optional steps (Playwright, Triton, SageAttention, llama.cpp, LLM models) never stop the install. When they fail they show `FAILED-NONFATAL` (or `NEEDS-YOU` with a command) and video generation still works.
- The summary ends with doctor's own read-only scan, then either "Ready for generate (LTX 2.3)" or the list of items that still need you (exit 1).

When it says ready:

```bash
python -m master_agent comfy start
python -m master_agent comfy run --mode generate --variant base --prompt "a test shot"
```

### llama.cpp

The local LLM (brief, director, judge, knowledge-base embeddings) runs on llama.cpp. Automatic install puts it in `LLAMACPP_ROOT` (default `video_buddy/llama.cpp`), pinned to the official ggml-org/llama.cpp release tag `LLAMACPP_REF` (default `b11389`):

1. With git, cmake, and a C++ compiler, it builds `llama-server` from that tag: CUDA with an NVIDIA GPU and the CUDA toolkit, Metal on Apple Silicon, CPU otherwise.
2. If a tool is missing or the build fails, it downloads the official prebuilt release for your platform instead.
3. If neither works, it says which tools are missing and the command that installs them, and carries on.

Running it again is safe: a finished copy at the pinned tag is skipped, and a half-done checkout or download is resumed. Buddy finds that `llama-server` by itself; `LLAMACPP_BIN` and `PATH` still win.

The LLM models are GGUF files llama-server serves from `MODELS_DIR`: `LLAMACPP_MODEL` (`qwen3-vl-heretic`) and `KB_EMBED_MODEL` (`nomic-embed-text`). Buddy has no public download source for them, so automatic install skips each one with a line like "Skipped qwen3-vl-heretic: no public GGUF source is defined; continuing." It does not stop or ask, and the run can still end "Ready for generate (LTX 2.3)". The summary lists them on a "Skipped" line. To use them, put the GGUF files in `MODELS_DIR`.

## Install

`install.py` without `--automatic-install` is the lighter install. It creates `.venv`, installs pip deps, Playwright Chromium, copies `.env`, and tries ffmpeg. It does not install ComfyUI or download video weights. It handles the LLM models the same way automatic install does: each one not already in `MODELS_DIR` is skipped with "Skipped qwen3-vl-heretic: no public GGUF source is defined; continuing." and the install carries on. It never installs Ollama or runs `ollama pull`.

`python install.py --check` only reports: it runs `setup` (the doctor) with the existing `.venv` and creates, installs, and downloads nothing. With no `.venv` yet it says so and exits 1.

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

`setup` is `doctor`. `setup --fix` installs missing deps (ffmpeg via winget, brew, or apt-get when that tool exists) and skips LLM GGUFs with no public source, as above. It never pulls Ollama models. Doctor lists them on an informational `llm-models` row that never fails the check. If you run Ollama yourself, install it from https://ollama.com/download and pull your own models; Buddy does not.

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
