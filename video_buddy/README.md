# VIDEO BUDDY

Python package for the local ComfyUI studio. Git root is the parent repo. The operator manual is [`../docs/INDEX.md`](../docs/INDEX.md).

- Code: `master_agent/` (`python -m master_agent`)
- Graphs: `workflows/` (LTX 2.5, MiniMax H3, curated library)
- Tests: `tests/`
- Skill: `skills/video-buddy/`

## Install and first commands

Need Python 3.10+. Full steps and the L0→L5 curriculum: [Getting started](../docs/GETTING_STARTED.md).

Empty machine: `python install.py --automatic-install --yes` (or `python -m master_agent automatic-install --yes` once `.venv` exists). Automatic install runs a pre-flight check, then installs ComfyUI in its own folder, LTX 2.3, and llama.cpp. See [Getting started](../docs/GETTING_STARTED.md#automatic-install).

```bash
python install.py
python -m master_agent doctor
python -m master_agent workflows
python -m master_agent health
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "a test shot"
```

Windows: `install.bat`. macOS / Linux: `./install.sh`. `python install.py` never installs Ollama or pulls models; LLM GGUFs with no public source are skipped and the install continues. `python install.py --check` only reports. Agent entry: `python -m master_agent agent list`. Comfy is `:8188`.

## Where each fact lives

| Topic | Doc |
|---|---|
| Loader order and consent | [Weights](../docs/WEIGHTS.md#loader-policy) — the only [GGUF Q4 loader policy](../docs/WEIGHTS.md#loader-policy) |
| Commands | [CLI reference](../docs/CLI_REFERENCE.md) |
| Managed Comfy and attach | [Comfy](../docs/COMFY.md) |
| Judge | [Judge](../docs/JUDGE.md) |
| Catalog, lipdub, HeartMuLa, Pack C | [Features](../docs/FEATURES.md) |
| Stop-lines | [Agents](../docs/AGENTS.md) |
| Wired / retired | [Audit](../docs/AUDIT.md) |

H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.

Pre-slim copies of this README and the old `docs/` essays are in [`../docs/archive/`](../docs/archive/). They are not the manual.
