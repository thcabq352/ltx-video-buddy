# Video Buddy docs

Operator manual. One fact lives in one file. Other pages link here instead of repeating it.

| Doc | What it is |
|---|---|
| [Getting started](GETTING_STARTED.md) | Install, L0→L5, first generate |
| [Architecture](ARCHITECTURE.md) | Intake, director, Hands, orchestrator, local LLM |
| [CLI reference](CLI_REFERENCE.md) | `python -m master_agent` commands |
| [Weights](WEIGHTS.md) | Inventory, consent, and the only [GGUF Q4 loader policy](WEIGHTS.md#loader-policy) |
| [Comfy](COMFY.md) | Managed server, external mode, attach contract |
| [Judge](JUDGE.md) | Look vs health, quality bar, revise loop |
| [Provenance](PROVENANCE.md) | `buddy.clip.provenance/v1` sidecars |
| [Features](FEATURES.md) | Catalog, lipdub, HeartMuLa, music video, Pack C |
| [Agents](AGENTS.md) | Stop-lines and field lessons |
| [Audit](AUDIT.md) | Wired / unwired / retired |

Package entry: [`video_buddy/README.md`](../video_buddy/README.md). Repo entry: [`README.md`](../README.md). How to change these pages: [`CONTRIBUTING.md`](../CONTRIBUTING.md).

Still next to the code, not folded:

- [`docs/demo/`](demo/) — public MiniMax H3 BMX still, 720p, provenance
- [`docs/showcase/h3-bmx/`](showcase/h3-bmx/) — pointer at `docs/demo/` (no second copy)
- [`video_buddy/workflows/`](../video_buddy/workflows/) — graph READMEs beside the JSON
- [`knowledge/`](../knowledge/) — git-synced memory between machines
- [`docs/archive/`](archive/) — pre-slim copies. Historical. Not the manual.
