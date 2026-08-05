# Video Buddy — Investor One-Pager

**The autonomous AI video studio that runs on your own hardware.**

---

## The problem

AI video models are finally good enough for commercial work — and nearly
impossible to operate. Production-grade results require expert-built node
pipelines, exact weight management, manual consistency fixes, and hours of
GPU trial-and-error per shot. Cloud services (Runway, Pika, Kling) solve
usability but introduce per-minute metering, queues, and a hard privacy
ceiling: **no agency or brand uploads embargoed client material to a third
party.** The result is a capability gap sitting between two broken options.

## The product

Video Buddy is an agent — not another model — that turns a local workstation
into a full production studio:

- **It interviews the client** and writes the creative brief (persona intake).
- **It directs** — an LLM director routes the job; a multi-LLM panel
  storyboards shot-by-shot.
- **It executes safely** — every workflow is validated before GPU time is
  spent; failures auto-recover.
- **It judges its own work** with a three-legged reviewer (heuristics + LLM +
  vision model) and re-renders weak shots autonomously.
- **It remembers everything** in a local knowledge base — every run makes the
  next one better.

Feature surface: short films with consistent characters and **cloned voices**,
beat-synced music videos, talking-head lip-sync, VFX/AI-rendering passes,
procedural fractal visuals, a character-LoRA factory, and production-grade
upscaling. All on one consumer GPU (16GB). No cloud required.

## Why now

- Open video models (LTX 2.3, Flux 2, Wan 2.2) crossed the commercial quality
  bar in the last 12 months — the raw capability is suddenly free.
- The ecosystem ships power, not usability: the operability gap is the
  market opening.
- Privacy regulation and client confidentiality increasingly rule out cloud
  processing for commercial footage.

## Moat

Not the models — the **operational knowledge**: validated pipeline library,
quality-judging ensemble, and a compounding knowledge base of what works.
Every job run deepens it. A competitor can download the same weights; they
cannot download the judge's accumulated judgment.

## Business model [TODO — founder input]

Candidate motions, to be finalized:

- **Studio-in-a-box license** for agencies and in-house brand teams
  (annual seat + support).
- **Managed service**: we operate the machine; clients get the privacy of
  dedicated hardware with none of the ops.
- **Pack marketplace**: premium pipeline packs (editor, mastering-grade
  upscale tiers) on top of the open core.

## Traction / status [TODO — update as it develops]

- Working system today: 29 validated workflows, 5 model families, full
  intake→delivery loop, 104-test suite, documented white paper.
- Reference hardware: single 16GB consumer GPU — i.e., deployable anywhere.
- [TODO: pilot users, demo reel link, first LOIs]

## Market [TODO — founder to size with cited sources]

Anchor segments: independent video agencies, in-house brand/content teams,
music artists and labels, freelance commercial creators. (Insert TAM/SAM/SOM
with citations before distribution.)

## The ask [TODO]

[Round size, instrument, use of proceeds — e.g., 18-month runway to productize
packs, ship the editor, and convert N pilot studios.]

## Honest risks

- **Model licensing**: several underlying models restrict commercial use;
  deployment requires per-model license review (budgeted in the plan).
- **Model dependency**: we orchestrate third-party models — mitigated by a
  multi-family library and a one-command workflow converter that keeps us
  current with new releases.
- **Hardware drift**: consumer GPU tiers shift; the 16GB engineering profile
  is portable across NVIDIA generations.

---

*Contact: [TODO] · White paper: `docs/WHITEPAPER.md` · Demo reel: [TODO]*
