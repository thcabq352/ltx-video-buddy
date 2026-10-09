# Video Buddy

Local video studio. A plain-language brief becomes a judged clip on your own ComfyUI. No cloud render farm is required.

This repository is the whole product: the Comfy driver, weights policy, workflows, judge, Hermes skill, curriculum, and knowledge base.

Public MiniMax H3 still (8s, golden-hour BMX). Files are in-repo:

<p align="center">
  <a href="docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4">
    <img src="docs/demo/H3-SHOWCASE-BMX-8s-hero.png" alt="MiniMax H3 via Video Buddy — golden-hour BMX berm" width="100%" />
  </a>
</p>

Play [`docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4`](docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4). Write-up: [`docs/demo/`](docs/demo/).

## What's new

Latest on `main` (full history in [`CHANGELOG.md`](CHANGELOG.md)):

- **Workflow ingest.** Hand Buddy a one-off ComfyUI graph (API JSON, UI JSON through a running Comfy, or a Comfy history entry). It learns the tunable fields, reports missing nodes and models, dry-runs without queueing, and can queue the graph or promote it to a local draft. The studio Comfy tab has a drop zone for the same path. Optional `--llm-assist` uses a local model only. [Guide](docs/WORKFLOW_INGEST.md).
- **Managed ComfyUI.** `comfy start` / `stop` / `status` / `restart` through comfy-cli on loopback, with SageAttention requested whenever it is installed. Bring-your-own Comfy stays supported. [Comfy](docs/COMFY.md).
- **Safer decodes.** Tiny preview VAEs are swapped out of, or refused on, tiled decode before anything is queued.
- **Local LLM.** llama.cpp is the preferred backend, then Ollama.

## Status

- **Pre-1.0.** Version numbers in the changelog are proposed. No release has been tagged yet.
- **Local-first.** Video renders on your ComfyUI. The LLM defaults to llama.cpp, then Ollama. A cloud LLM is used only when you have credentials for it and either select it or both local servers are down. Nothing downloads weights without your consent.
- **Tested without a GPU.** The unit suite runs without ComfyUI or a GPU, and CI runs a subset of it on every push. GPU renders are verified on operator hardware, not in CI.
- **In progress.** Agent entry points (MCP, the Hermes facade, A2A on `:8189`) are being consolidated into one gateway. See the [status note](docs/ARCHITECTURE.md#status-agent-gateway).

## Start here

The operator manual is [`docs/INDEX.md`](docs/INDEX.md).

```bash
cd video_buddy
python install.py
python -m master_agent doctor
python -m master_agent workflows
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain"
```

Install, the L0→L5 curriculum, and the first generate: [Getting started](docs/GETTING_STARTED.md).

H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.

Loader order and the rule that doctor does not download weights: [Weights](docs/WEIGHTS.md#loader-policy).

## What the docs cover

| | |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Intake, director, Hands, local LLM |
| [CLI](docs/CLI_REFERENCE.md) | `python -m master_agent` |
| [Comfy](docs/COMFY.md) | Managed server, external mode, attach |
| [Judge](docs/JUDGE.md) | Look vs health, revise loop |
| [Features](docs/FEATURES.md) | Catalog, lipdub, HeartMuLa, music video, Pack C |
| [Workflow ingest](docs/WORKFLOW_INGEST.md) | Learn, dry-run, run, or promote a one-off Comfy graph |
| [Rainey1](docs/RAINEY1.md) | Recipes, opt-in judge rubric, batch top-cut |
| [Audit](docs/AUDIT.md) | Wired, unwired, retired |
| [Agents](docs/AGENTS.md) | Stop-lines |
| [Releasing](docs/RELEASING.md) | Commit, PR, and release-notes conventions |

Package README: [`video_buddy/README.md`](video_buddy/README.md). Shared memory: [`knowledge/`](knowledge/). Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Changes: [`CHANGELOG.md`](CHANGELOG.md).

## Layout

```
docs/                  operator manual, demo stills, archive of the pre-slim manuals
knowledge/             git-synced learnings (Chroma stays local and gitignored)
video_buddy/           master_agent, workflows, tests
```

Hermes: `cd video_buddy && python install_hermes_skill.py`. MCP is not the skill. Profile `ltx`. Buddy does not bind 8642. Details: [CLI](docs/CLI_REFERENCE.md#hermes).

## Acknowledgments

Thanks to Jason Rainey (Rainey1, @jasonr.tv). He has been a collaborator since childhood, helped shape the style of LTX Video Buddy, and has contributed a great deal to the project.

## License

Video Buddy is an orchestration layer. LTX, Flux, Wan, Qwen, MiniMax, and the other models it drives are third-party works under their own licenses. Several restrict commercial use. LTX 2.5 Hub packs are gated. Selected workflow designs credit [Mickmumpitz](https://mickmumpitz.ai). Review each model's license before commercial use. This repository contains no model weights.
