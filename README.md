# Video Buddy

Local video studio. A plain-language brief becomes a judged clip on your own ComfyUI. No cloud render farm is required.

Public MiniMax H3 still (8s, golden-hour BMX). Files are in-repo:

<p align="center">
  <a href="docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4">
    <img src="docs/demo/H3-SHOWCASE-BMX-8s-hero.png" alt="MiniMax H3 via Video Buddy — golden-hour BMX berm" width="100%" />
  </a>
</p>

Play [`docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4`](docs/demo/H3-SHOWCASE-BMX-8s-720p.mp4). Write-up: [`docs/demo/`](docs/demo/).

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
| [Audit](docs/AUDIT.md) | Wired, unwired, retired |
| [Agents](docs/AGENTS.md) | Stop-lines |

Package README: [`video_buddy/README.md`](video_buddy/README.md). Shared memory: [`knowledge/`](knowledge/). Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Layout

```
docs/                  operator manual, demo stills, archive of the pre-slim manuals
knowledge/             git-synced learnings (Chroma stays local and gitignored)
video_buddy/           master_agent, workflows, tests
```

Hermes: `cd video_buddy && python install_hermes_skill.py`. MCP is not the skill. Profile `ltx`. Buddy does not bind 8642. Details: [CLI](docs/CLI_REFERENCE.md#hermes).

## License

Video Buddy is an orchestration layer. LTX, Flux, Wan, Qwen, MiniMax, and the other models it drives are third-party works under their own licenses. Several restrict commercial use. LTX 2.5 Hub packs are gated. Selected workflow designs credit [Mickmumpitz](https://mickmumpitz.ai). Review each model's license before commercial use. This repository contains no model weights.
