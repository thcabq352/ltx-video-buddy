# Video Buddy: An Autonomous Agent for Local, Production-Grade AI Video

**White paper — v1.0, August 2026**

---

## Abstract

Generative video has crossed the quality threshold for commercial use, but not
the usability threshold. State-of-the-art open models — LTX 2.3, Flux 2 Klein,
Wan 2.2 — are distributed as raw weights and node graphs whose correct
operation demands specialist knowledge, hours of manual iteration, and
hardware most creative teams don't have in-house. The market's answer so far
has been cloud generation services: convenient, but metered, queued,
opaque, and structurally unable to guarantee data privacy.

This paper describes **Video Buddy**, an autonomous agent that converts a
local workstation into a full AI video production studio. The agent
interviews the client, directs the production, operates a curated library of
29 workflows across five model families, grades its own output with a
multi-modal judge, and accumulates institutional knowledge in a local
retrieval store — all on a single consumer GPU, with no mandatory cloud
dependency. We describe the architecture, the quality-control loop, the
self-learning knowledge base, and the engineering constraints of running
production workloads in 16GB of VRAM.

## 1. The problem: the last mile of generative video

The open video-model ecosystem (Lightricks LTX, Black Forest Labs Flux,
Alibaba Wan, and others) now produces footage that clears the bar for
advertising, music video, and short-form narrative work. Between the weights
and a finished deliverable, however, sits a punishing operational gap:

- **Workflow complexity.** Results are produced in ComfyUI, a node-graph
  environment where a single production pipeline may span 200+ nodes, a dozen
  custom extension packs, and exact-match weight files across six model
  categories. Assembling this is days of expert work per pipeline.
- **Consistency.** Characters drift across shots; voices don't exist at all
  without a cloning pipeline; color grades jump between generations.
- **Iteration cost.** Each render is minutes of GPU time. Naive trial-and-error
  burns hours on outputs a professional would have rejected on sight.
- **Privacy and cost.** Cloud alternatives meter by the second, queue behind
  other customers, and require uploading client material to third-party
  infrastructure — a non-starter for embargoed commercial work.

The consequence: a capability that is technically democratized remains
operationally gated behind a small pool of specialists.

## 2. Design goals

1. **Intent in, deliverable out.** The operator supplies a brief in natural
   language; the system returns a finished, reviewed video file.
2. **Local-first.** Every stage — planning, rendering, judging, memory — must
   run on one workstation with a 16GB GPU. Cloud LLMs may assist but are
   never required.
3. **Fail fast.** No GPU time is spent on a workflow that cannot succeed;
   validation is exhaustive and happens before queueing.
4. **Self-correcting quality.** The system must detect its own failures and
   retry with adjusted parameters, the way a human operator would.
5. **Compounding knowledge.** Every run — its brief, plan, scores, and the
   judge's reasoning — must make the next run better.

## 3. System architecture

Video Buddy is organized as five cooperating subsystems around a single
render engine (ComfyUI, driven headlessly over its HTTP API).

### 3.1 Intake: the persona interview

Production failures are usually brief failures. Before any generation, the
agent's persona conducts a structured interview — audience, tone, visual
language, must-haves, deal-breakers — and synthesizes a creative brief that
becomes the pipeline's request object. Personas are declarative markdown
files; the default ("Ara") is calibrated for commercial work, and clients can
author their own. Interviews are persisted as records, so the system learns
each client's preferences across engagements.

### 3.2 Direction: routing and storyboarding

A locally hosted 27B-class LLM acts as director: it reads the brief and
routes it to one of the pipeline variants (text-to-video, lip-sync dub,
music video, fractal, movie builder, image workflows), with hard constraints
(a supplied source video implies lip-sync) always overriding model judgment.
Long-form requests are split by a storyboard stage that plans one shot card
per segment — camera, action, continuity language, per-shot prompt. The
storyboard can be drafted by a *panel* of LLMs (local models, optionally
joined by cloud models), with a judge model selecting or blending the
strongest board. Every panel member is optional and every failure degrades
gracefully to heuristics — the pipeline never hard-fails on LLM
unavailability.

### 3.3 Execution: validate-first orchestration

Generation runs as a state machine:

`SELECT_VARIANT → PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE → DONE/ERROR`

Two properties do the economic work:

- **Pre-flight validation.** Workflows are checked against the live node
  registry and the local model inventory — class types, required inputs,
  widget values, link-type integrity, and exact weight filenames — before
  anything is queued. A misconfigured run costs seconds, not GPU-minutes.
- **Automatic degradation.** A CUDA out-of-memory error walks a documented
  downscale ladder and retries rather than aborting the run.

### 3.4 Judgment: the three-legged quality loop

Every clip is scored by an ensemble that mirrors how a human reviewer judges
footage:

1. **Heuristics** — file integrity, duration vs. request, frame-difference
   motion analysis.
2. **Text LLM verdict** — the judge reasons over the brief and render
   parameters.
3. **Vision evaluation** — a 30B vision-language model reviews extracted
   frames for temporal consistency, subject lock, and artifacts.

Below threshold, the judge doesn't just retry — it rewrites the prompt or
retunes a whitelisted parameter set (steps, guidance, sampler, seed) within
clamped bounds, and regenerates. For multi-shot pieces, a full-video judge
reviews the stitched cut against the whole brief and selectively re-renders
only the weak shots.

### 3.5 Memory: the local knowledge base

All workflows, prose guides, and run records are embedded into a local
ChromaDB store (Ollama embeddings; no data leaves the machine). Before
storyboarding, the pipeline recalls similar past runs — including the judge's
own critique — and injects them into planning. The system thereby converts
its operational history into an asset: prompt strategies that scored well
resurface; failure modes are avoided. Character bibles and LoRA training
outcomes are indexed alongside, so identity assets compound too.

## 4. The production library

The agent operates a curated, validated library of 29 manifest workflows
across five model families, all converted to API format and verified against
a live server (16/16 clean):

- **Movie Builder (LTX 2.3 + Flux 2 Klein).** Shot-by-shot film production:
  shared character references (face / body / blended), voice cloning from a
  five-second sample, two-stage sampler with per-stage seed control,
  automatic color matching across shots, optional 2× spatial upscaling,
  360° equirectangular environment generation enabling matched
  shot-reverse-shot coverage, and a final assembler producing the cut.
- **Consistent Character Creator.** Bible → reference sheet → vision-curated
  dataset → LoRA training → vision-judged validation, with a documented retry
  ladder. Produces reusable identity assets loadable by any workflow.
- **Music video pipeline.** Dependency-free beat detection (spectral-flux
  onsets, comb-filter tempo estimation, phase-aligned beat grid, RMS energy
  sections) drives shot planning and exact beat-window trimming; the source
  track is muxed onto the finished cut.
- **AI-VFX and AI-rendering pipelines** — compositing, depth/clay/line
  control passes, 360° turnarounds, super-resolution.
- **Finishing.** NVIDIA RTX Video Super Resolution for fast upscales and
  SeedVR2 diffusion upscaling for production-grade masters.

A reusable UI→API converter (regression-checked against nine hand-verified
pairs) makes new community workflows a one-command integration, keeping the
library current with the ecosystem's release cadence.

## 5. Engineering within 16GB

The reference hardware is a single RTX 5060 Ti (16GB). Production viability at
this tier is an engineering result, not a given:

- FP8 weight quantization across transformer and text encoders; a GGUF
  low-VRAM profile (Q4_K_S) for constrained runs.
- Training LoRAs in 16GB via quantization, low-VRAM mode, gradient
  checkpointing, and rank-16 adapters; the trainer frees inference VRAM
  before starting.
- Sequential stage loading (stills model, then video model) rather than
  concurrent residency.
- Duration-aware segmentation: requests exceeding the per-clip VRAM cap are
  planned as multi-shot pieces from the start rather than failing late.

## 6. What this is not

Video Buddy is an orchestration and quality layer, not a foundation model.
Its defensibility lies in the operational knowledge encoded in its pipelines,
its judge, and its accumulated run history — the difference between owning a
camera and knowing how to shoot. The models it drives are third-party works
under their own licenses (some non-commercial), and any commercial deployment
requires the appropriate licenses for the models in the chosen pipeline.

## 7. Roadmap

- **Special packs** (in design): a non-linear **editor** stage for
  agent-assisted recuts, and a **production mastering** tier — multi-day,
  maximum-quality upscaling and finishing runs for hero deliverables.
- Scene/style LoRA training beyond characters (scaffold shipped).
- Deepening the self-learning loop: judge-feedback fine-tuning of prompt
  strategies; cross-client preference profiles.
- Multi-GPU and render-queue scheduling for studio deployments.

## 8. Conclusion

The binding constraint on AI video adoption is no longer model quality — it
is operability. Video Buddy demonstrates that the full production loop —
brief, direction, execution, judgment, memory — can be automated on local
hardware with no mandatory cloud dependency, turning a workstation into a
studio that improves with every job it runs.

---

*Video Buddy is built on ComfyUI and open model families from Lightricks,
Black Forest Labs, Alibaba, and others, with selected workflow designs by
Mickmumpitz. All trademarks belong to their owners.*
