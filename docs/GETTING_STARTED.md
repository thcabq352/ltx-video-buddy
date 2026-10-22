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
- Each node pack's own `requirements.txt`, then `gguf` and `colour-science` (`video_buddy/comfy-requirements.txt`), installed into the Python ComfyUI runs with (its own venv), not Buddy's `.venv`. Already-importable packages are skipped. Both are required: ComfyUI-GGUF and ComfyUI-LTXVideo do not load without them
- ComfyUI-Manager's channel: `config.ini` and `channels.list` point at the Comfy-Org channel, and an existing `config.ini` with the old `ltdrdata` URL is repaired (see [Comfy](COMFY.md#comfyui-manager-channel))
- Triton and SageAttention inside ComfyUI's own venv (optional; see [Comfy](COMFY.md#triton-and-sageattention))
- The LTX 2.3 weights, about 76.5 GB, into `MODELS_DIR` only (never into the ComfyUI folder)
- llama.cpp for the local LLM (optional runtime; see [llama.cpp](#llamacpp) below)
- The local LLM files, about 2.6 GB, into `MODELS_DIR`: the text brain `qwen3-vl-4b-heretic-Q4_K_M.gguf` and the embedding file `nomic-embed-text-v1.5.Q4_K_M.gguf`. If a vision pair is already on disk (a model file plus a projector, looked up in `MODELS_DIR/gguf_vl_heretic/` first), that pair is the brain and the 4B file is not downloaded. If `nomic-embed-text` already exists as an Ollama file, that file is used and the 84 MB download is skipped

LTX 2.3 is the only video model automatic install sets up. Other packs stay opt-in through `download-models`. Automatic install never installs Ollama or pulls anything through it.

### Pre-flight

The pre-flight runs first, before anything is downloaded or installed. It prints one PASS, FAIL, or INFO line per check in plain English. It stops at the first FAIL, says how to fix it, and changes nothing.

| Check | Kind | What it looks at |
|---|---|---|
| Disk space | stops on FAIL | Free space on each drive that will hold the LTX 2.3 weights (`MODELS_DIR`), the local LLM GGUFs (about 2.6 GB on an empty machine, less when a vision pair or an embedding file is already present, same drive), the ComfyUI folder, and llama.cpp (`LLAMACPP_ROOT`). Uses the real file sizes of what is still missing, plus 5 GB headroom. On an empty machine that is about 100–102 GB (llama.cpp takes 0.3–5 GB depending on build or prebuilt, CPU or CUDA). Prints free vs needed, and how to free space or move a folder to a bigger drive. |
| Hugging Face access | stops on FAIL | The LTX 2.3 files are public, so no token is needed. If `HF_TOKEN` is set, it is checked with a lightweight sign-in call, because a bad token breaks even public downloads. The token is never printed. |
| GPU / VRAM | information only | Reads `nvidia-smi`. Under 14 GB VRAM, the GGUF Q4 loader is used. No GPU means CPU mode: it works, but is very slow. |
| llama.cpp | information only | One of: found (`LLAMACPP_BIN`, `PATH`, or Buddy's own copy), will build from source, will use the official prebuilt release (and which build tool is missing), or unavailable (with the command that installs the build tools). Also lists each local LLM file it will download, with the link. |
| ComfyUI folder | stops on FAIL | The folder must be writable and either empty, a ComfyUI, or missing. Notes an existing install (that step is then skipped). Refuses when `COMFY_MODE=external`. |

```bash
python -m master_agent automatic-install --preflight-only   # pre-flight only, changes nothing
python -m master_agent automatic-install                    # pre-flight + the plan, changes nothing
python -m master_agent automatic-install --yes              # pre-flight, then install
```

Other flags: `--gpu auto|nvidia|amd|m-series|cpu` picks the ComfyUI PyTorch build (default: detect). `--skip-weights` installs everything but the weights; the install and later `doctor` runs report them as INFO, and `models register <file-or-folder>` adopts files you already have ([Weights](WEIGHTS.md#skipping-or-registering-the-ltx-23-pack)). `--skip-sage` skips Triton and SageAttention. `--skip-llm` skips llama.cpp and the local LLM models.

### If something fails

- A pre-flight FAIL stops the run before any change. Fix the item and run the same command again.
- Running it again is safe. Finished steps are skipped. A half-finished ComfyUI is repaired with `comfy install --restore`.
- Buddy never uses `sudo` silently. On Linux, when ffmpeg needs administrator rights and `sudo` would ask for a password, the step shows `NEEDS-YOU` with the exact command to run yourself. The same goes for the llama.cpp build tools.
- A pip failure while installing the node packs' Python packages is a `FAILED` step with pip's error and the command to retry: the install ends `Install FAILED`, because video generation cannot work without them.
- Optional steps (Playwright, ComfyUI-Manager channel, Triton, SageAttention, llama.cpp) never stop the install. When they fail they show `FAILED-NONFATAL` (or `NEEDS-YOU` with a command) and video generation still works.
- The LLM file step is required, the same way the Comfy Python packages step is: already-complete files are skipped, a finished download is checked for size, and a failure is loud. The summary, `state/LAUNCH_NOTES.txt`, and `doctor` name the direct link and the exact folder to copy the file into. The install then exits with a problem to fix, so it does not say "Ready" while the local LLM is missing. `--skip-llm` is the only way to leave that step out.
- The summary ends with doctor's own read-only scan, then `Install OK: N ok, M non-fatal note(s)` and "Ready for generate (LTX 2.3)", or the list of items that still need you and `Install FAILED: ...` (exit 1). Every `INFO` or `FAILED-NONFATAL` line carries a "safe to ignore because ..." note.

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

The LLM files are GGUFs. llama-server serves each one from a folder under `MODELS_DIR` whose name is the model id (`LLAMACPP_MODEL`, default `qwen3-vl-heretic`, and `KB_EMBED_MODEL`, default `nomic-embed-text`). Automatic install downloads them. A file that is already the full size is skipped. A half-downloaded file is fetched again (the Hub download resumes). Running it again is safe.

| What it is for | Folder | File | Direct link |
|---|---|---|---|
| Brief, director, storyboard, text judge | `MODELS_DIR/qwen3-vl-heretic/` | `qwen3-vl-4b-heretic-Q4_K_M.gguf` (about 2.5 GB) | https://huggingface.co/DreamFast/Qwen3-VL-4b-Heretic-GGUF/resolve/main/qwen3-vl-4b-heretic-Q4_K_M.gguf |
| Knowledge-base embeddings | `MODELS_DIR/nomic-embed-text/` | `nomic-embed-text-v1.5.Q4_K_M.gguf` (about 84 MB) | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf |

Buddy uses a vision pair you already have before it downloads the 4B file. It looks in `MODELS_DIR/gguf_vl_heretic/` first, then in `MODELS_DIR/qwen3-vl-heretic/` if that folder has a projector, then in any other folder directly under `MODELS_DIR` that contains both a model `.gguf` and a file whose name starts with `mmproj`. When it finds that pair, it writes `state/llamacpp-models.ini` with `model` and `mmproj` pointing at those files (llama-server loads the projector with `--mmproj`), the served name stays `qwen3-vl-heretic`, and the vision judge can see images. The files stay where they are. Buddy does not move, rename, or delete them, and it does not download the 4B file.

The 4B repo does not publish an `mmproj` file (checked on the Hub file list). If that 4B file is the only brain, text works and the vision judge is text-only: it cannot see images. To turn vision on, put a model `.gguf` and an `mmproj*.gguf` in the same folder (for example `MODELS_DIR/gguf_vl_heretic/model.Q4_K_M.gguf` and `MODELS_DIR/gguf_vl_heretic/mmproj-Q8_0.gguf`) and run automatic install again. Two optional 8B packs include a projector. Automatic install does not download them. You can also put the model plus the mmproj in `MODELS_DIR/qwen3-vl-heretic/`:

- [SergiusFlavius/Qwen3-VL-8B-Instruct-heretic-gguf](https://huggingface.co/SergiusFlavius/Qwen3-VL-8B-Instruct-heretic-gguf) — `Qwen3-VL-8B-Instruct-heretic-Q4_K_M.gguf` and `mmproj-Qwen3-VL-8B-Instruct-heretic-F16.gguf` (the mmproj name already starts with `mmproj`, which llama.cpp expects).
- [mradermacher/Qwen-3-VL-8B-Instruct-heretic-GGUF](https://huggingface.co/mradermacher/Qwen-3-VL-8B-Instruct-heretic-GGUF) — `Qwen-3-VL-8B-Instruct-heretic.Q4_K_M.gguf` and `Qwen-3-VL-8B-Instruct-heretic.mmproj-f16.gguf`. That mmproj name does not start with `mmproj`. Rename it so it does, or keep the `mmproj =` line automatic install writes in `state/llamacpp-models.ini`.

If a download fails, or you set `LLAMACPP_MODEL` to a name this repo does not know, the install prints a warning box, writes the same text to `video_buddy/state/LAUNCH_NOTES.txt`, and `doctor` shows `WARN  llm-models` with the link. Copy the file into the folder named above and run the same command again.

### Which program answers the LLM

Two programs can run the local model. Buddy never installs the second one.

- **llama.cpp** (the default this installer sets up). It is a program on this computer. It reads the GGUF files above. You do not install anything else. When `.env` has no `LLM_PROVIDER` line and the GGUF files are in place, automatic install adds `LLM_PROVIDER=llamacpp`. A failed download leaves the line alone, so `auto` can still try Ollama.
- **Ollama**. A separate app you install yourself from https://ollama.com/download. Buddy does not install it and does not run `ollama pull`. Set `LLM_PROVIDER=ollama` only if that is the app you want. An existing `LLM_PROVIDER=ollama` line is left as you wrote it. If `nomic-embed-text:latest` is already in Ollama's files, automatic install points llama.cpp at that blob and does not copy it or download a second embedding file. If it is not there, the 84 MB embedding file is still downloaded for llama.cpp.
- **auto** (only if you write `LLM_PROVIDER=auto`). Buddy tries llama.cpp first, then Ollama. It does not call a cloud model. Cloud Grok runs only if you set `LLM_PROVIDER=grok`.

`doctor` prints the value it will use on the `llm-models` row, which brain, projector, and embed files are in use, and whether vision is enabled.

## Install

`install.py` without `--automatic-install` is the **light** install. It creates `.venv`, installs pip deps, Playwright Chromium, copies `.env`, tries ffmpeg, and creates the empty model folders under `MODELS_DIR`. It does not install ComfyUI or download video weights, so nothing renders and nothing answers on `:8188` afterwards. A `THIS IS THE LIGHT INSTALL` banner at the start and the end says so and gives the full-setup command. Its report marks ComfyUI and the weight rows `INFO` ("safe to ignore because ...") and ends with `Install OK: N ok, M non-fatal note(s)`; only real failures (venv, pip, ffmpeg, ...) end `Install FAILED` with exit 1. It does not download the local LLM GGUFs (about 2.6 GB). If those files are missing it prints a warning with the direct link, writes `video_buddy/state/LAUNCH_NOTES.txt`, and `doctor` shows `WARN  llm-models`. That warning does not fail the light install. It never installs Ollama or runs `ollama pull`.

`python install.py --check` only reports: it runs `setup` (the doctor) with the existing `.venv` and creates, installs, and downloads nothing. With no `.venv` yet it says so and exits 1.

| OS | Command |
|---|---|
| Windows | `install.bat` or `python install.py` |
| macOS / Linux | `./install.sh` or `python3 install.py` |

```bash
cd video_buddy
python install.py                                   # light install only
python install.py --automatic-install --yes         # then the full setup: ComfyUI + LTX 2.3
python -m master_agent doctor
python -m master_agent comfy start
python -m master_agent health
```

`setup` is `doctor`. `setup --fix` installs missing deps (ffmpeg via winget, brew, or apt-get when that tool exists) and warns out loud when the LLM GGUFs are not in `MODELS_DIR`, as above. It never pulls Ollama models. `doctor` prints `WARN  llm-models` (with the download link) when those files are missing. That row does not by itself make `doctor` exit 1, because video generation can still run; it is not a quiet info line. If you run Ollama yourself, install it from https://ollama.com/download and set `LLM_PROVIDER=ollama`. Buddy does not install Ollama.

When `LLM_PROVIDER` is unset and the GGUF files are in place, automatic install writes `LLM_PROVIDER=llamacpp`. `auto` (only if you set it) tries llama.cpp, then Ollama, and never a cloud model. Grok is opt-in with `LLM_PROVIDER=grok`. Buddy starts `llama-server` against `MODELS_DIR` when `LLAMACPP_BIN` or `llama-server` is available, and stops it on shutdown. If that binary is missing and the provider is `auto`, Buddy warns and uses Ollama. Pin `LLM_PROVIDER=llamacpp` or `LLM_PROVIDER=ollama` to force one backend. See [Architecture](ARCHITECTURE.md#local-model).

Windows portable Comfy: `ComfyUI_windows_portable\run_api_8188.bat`. Elsewhere start Comfy with `--port 8188`, or set `COMFYUI_URL`.

```bash
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
python -m master_agent agent health
```

Hermes: `python install_hermes_skill.py --profile ltx` copies `skills/video-buddy/` (SKILL.md, TOOLS.md, PROFILE.md, references/) to `~/.hermes/profiles/ltx/skills/media/video-buddy/`; without `--profile` it goes to `~/.hermes/skills/video-buddy/`, and `--dest DIR` picks any folder. Nothing else is written. Agents start ComfyUI only with `python -m master_agent agent comfy_start` (detached), never a foreground `main.py`. The LTX bot gateway calls `python -m master_agent agent <tool>`. Buddy binds no gateway port and never 8642. See [CLI reference](CLI_REFERENCE.md#hermes).

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

### Browser dashboard

No terminal agent needed. Once the install above works (`health` shows ComfyUI up, or use the dashboard's **Start ComfyUI** button), from `video_buddy/`:

```bash
pip install -r requirements-ui.txt
streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501, type a prompt, pick duration and quality, and press **Generate**. The page shows `health` at the top, polls the render, and plays the MP4 when it is done. **Advanced** has the model variant (blank lets the director pick), storyboard mode, upscale, seed, and **Plan only** (a dry run that uses no GPU). A budget HOLD shows as paused, not failed; `budget reset-shift` continues it. One render runs at a time, and a page refresh finds it again.

`streamlit_app.py` is a thin layer over `master_agent.agent_api` (`health`, `comfy_start`, `create_video`). Keep `--server.address 127.0.0.1`: the page can queue GPU renders and has no login.

`--prepare` lints and writes JSON. It does not queue the GPU. Photo plus a voice file defaults to `ltx25_a2v`. Catalog ids and the H3 voice warning: [Features](FEATURES.md).

12GB cards (RTX 4000 Ada) and fully local LLM pins are in [Weights](WEIGHTS.md#12gb-and-local-llm).
