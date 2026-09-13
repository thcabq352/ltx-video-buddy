# Video Buddy × Comfy capability audit

**Scope:** repo-only investigation of whether `master_agent` discovers and drives the major ComfyUI packs Scott reports on the tower, or only a subset (LTX / old paths).

**Date:** 2026-09-13  
**Repo:** `thcabq352/ltx-video-buddy` @ this PR’s base (`main`)  
**Comfy in this checkout:** not present (`video_buddy/ComfyUI_windows_portable/` is gitignored). Live tower was **not** queried.

**Evidence used**

| Source | What it is | What it is not |
|---|---|---|
| `video_buddy/master_agent/**/*.py` | How Buddy builds, lints, and queues graphs | Live tower process flags |
| `video_buddy/workflows/manifests.yaml` + API JSON | Curated templates Buddy *can* load | Proof those packs are installed today |
| `video_buddy/state/object_info.json` (2,823 classes) | Cached `/object_info` snapshot committed in-repo | Live Comfy 0.35.1 on `F:\CircusMoved\...` |
| `video_buddy/state/model_inventory.json` (`generated_at` 2026-08-05) | Weights seen on a Scott tree that day | K3NK / FaceID / Realistic Vision added later |

**Hypothesis (do not treat as verified):** the cached `object_info` is from Scott’s Buddy Comfy (same portable layout as the inventory paths). It is **weeks-to-months old** relative to this audit and must not be cited as live 0.35.1 + sage-attention state.

---

## 1. How Buddy builds and queues Comfy graphs

Buddy never synthesizes a graph from `/object_info`. The path is **template → patch → lint → `/prompt`**.

```
brief ──▶ director.choose_variant (5 slugs)
              │
              ▼
     workflow_patcher.load_and_patch_workflow
         • load JSON named by manifests.yaml
         • apply named field map (node_id / class_type)
         • _heuristic_patch (hardcoded LTX/Flux/Wan class lists)
         • optional _ensure_lora_node
         • _sanitize_ltx_nodes (Guider / tiled VAE / 8n+1)
              │
              ▼
     optional power_mode (LLM ops, default OFF)
              │
              ▼
     validator.validate_workflow
         • bypass_optional_accelerators if class missing
         • every remaining class_type must be in object_info
              │
              ▼
     ComfyClient.queue_prompt  →  poll /history
```

### Concrete graph-builder files

| File | Role |
|---|---|
| `video_buddy/master_agent/comfy/workflow_patcher.py` | Load + inject prompt/seed/size/frames/ckpt/lora |
| `video_buddy/master_agent/comfy/graph_ops.py` | Power-mode mutations + **TeaCache bypass** |
| `video_buddy/master_agent/comfy/validator.py` | `/object_info` + inventory lint; LTX 8n+1 snap |
| `video_buddy/master_agent/comfy/client.py` | HTTP: `/prompt`, `/history`, `/object_info`, uploads |
| `video_buddy/master_agent/comfy/cli_run.py` | `comfy run` raw / template / generate |
| `video_buddy/master_agent/comfy/power_mode.py` | LLM `set_widget` / `add_node` (schema of **current** graph only) |
| `video_buddy/master_agent/comfy/diagnose.py` | 9-frame hull; refuses scale until `sec/step` |
| `video_buddy/master_agent/comfy/capabilities.py` | **This PR** — read-only gap probe |
| `video_buddy/master_agent/orchestrator/director.py` | Variant routing allowlist = `WORKFLOW_FILES` |
| `video_buddy/master_agent/orchestrator/machine.py` | PATCH → VALIDATE → SUBMIT; OOM walks `DOWNSCALE_LADDER` |
| `video_buddy/master_agent/fractal/render.py` | CPU Mandelbrot/Julia — **not Comfy** |
| `video_buddy/master_agent/upscale.py` | SeedVR2 (works) / RTX (broken path, gitignored) |
| `video_buddy/workflows/manifests.yaml` | Slug → file + optional `fields` |
| `video_buddy/master_agent/config.py` | Director allowlist, `MODEL_FILES`, `DOWNSCALE_LADDER` |

`object_info` is used for **validation and schema snippets**, not for choosing a pipeline. `object_info_snippets()` only emits classes **already in the patched graph**. Power mode is told “do not invent topology.”

---

## 2. Hardcoded vs dynamically discovered

| Layer | Dynamic? | What is hardcoded |
|---|---|---|
| Director | LLM picks among **5** slugs; rules: lipsync / wan22 / directors / eros / else `base` | `WORKFLOW_FILES`, `_VARIANT_KEYWORDS` |
| Patcher field maps | No | `manifests.yaml` `fields:` for `base`, `eros`, `directors`, `flux`, `lipsync`, `wan22`, `krea2_img` |
| Heuristic patch | No | Class lists: `EmptyLTXVLatentVideo`, `KSampler`, `UNETLoader`, `WanVideoNAG` is **not** specially handled (wan22 uses named node ids) |
| Validator | Yes — live/cache registry | Unknown `class_type` is a **hard fail** except optional accelerators |
| TeaCache | Soft-bypass only | `OPTIONAL_ACCELERATOR_CLASS_TYPES` in `graph_ops.py` |
| `comfy run --template` | Path or slug | **This PR:** slugs also resolve from `manifests.yaml` |
| Fractal | N/A | Numpy + ffmpeg. No Voronoi/Perlin/SAM/Fun nodes |
| Music | Beat map + same director graphs | No MMAudio sampler |
| KB `search_workflows` | Embedding over ingested JSON | Retrieval ≠ queue |

**There is no `ensure_teacache` injector in this repo.** Field notes and tests implement **missing TeaCache → WARNING + rewire MODEL past the node**. That is the correct shape for the graphs Buddy actually queues (see §3 TeaCache).

---

## 3. Tower capability matrix

**Legend — In Buddy?**

- **yes** — director or a first-class CLI path will drive it
- **partial** — a shipped template / post-stage / CPU path exists; agent will not pick it
- **no** — not wired (even if `/object_info` has the class)
- **hyp** — claimed on the tower; **not visible in repo snapshots**

`object_info?` = class present in committed `state/object_info.json` (not live tower).

| Capability | In Buddy? | How | object_info? | Gap / fix |
|---|---|---|---|---|
| LTX short I2V / T2V | **yes** | director `base`/`eros`/`directors`; patcher; 8n+1 + `DOWNSCALE_LADDER` | yes (`EmptyLTXVLatentVideo`, …) | Wired. Default path. |
| LTX lipsync | **yes** | director when `--video` / lipsync keywords | yes | Wired. |
| Wan 2.2 T2V (Mick native high/low UNET) | **yes** | director `wan22` → `260713_VIDEO-BUDDY_WAN-2-2-VID_1-0_api.json` | yes (`WanVideoNAG`) | **Was broken** as `comfy run --template wan22` (stale `MICKMUMPITZ_*` filename). Fixed this PR. Still T2V UNETs, not I2V AIO. |
| K3NK WAN 2.2 AIO I2V HIGH/LOW fp8 | **hyp / no** | — | no class; no inventory name | Need weights + I2V template + `MODEL_FILES` keys. Do not spray onto `wan22` T2V loaders. |
| WanVideoWrapper | **no** | — | **yes** (119 `WanVideo*` classes) | Native `wan22` uses `UNETLoader`+`KSamplerAdvanced`, not `WanVideoSampler`. Wrapper is unused. |
| WAN Fun Inpaint | **no** | — | **yes** `WanFunInpaintToVideo` | No graph, no mask ingest, no patcher fields. Highest-value missing node for recursive inpaint. |
| WAN Fun Control | **no** | manifests `vb_zimage_turbo_cn` only | **yes** `WanFunControlToVideo`, `Wan22FunControlToVideo` | Example file lives under gitignored `AI-RENDERING-EXAMPLE FILES/`. |
| LightX2V LoRAs | **partial** | baked in wan22 Power Lora widgets | n/a (weights) | Inventory has `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32`. Patcher cannot toggle it. |
| Wan TeaCache | **partial (bypass)** | validator drops missing accel | **yes** `WanVideoTeaCache` (out: **CACHEARGS**), `WanVideoTeaCacheKJ` (MODEL, deprecated) | **Do not inject** onto LTX or native wan22. Wrapper TeaCache is args for `WanVideoSampler`, not a MODEL wrapper. |
| VACE | **partial** | templates `vb_aivfx_adv`, `vb_aivfx_adv_13` | **yes** | Not director-routed. No possession/stand-in wiring. |
| Stand-In | **no** | — | **yes** `WanVideoAddStandInLatent` | Wrapper embed node; needs a Wrapper graph. |
| SAM2 masks | **no** | — | **yes** `SAM2Segment` | AI-VFX preprocess uses **SAM3**, and is template-only. |
| LanPaint | **hyp / no** | — | **no** | Not in cache or any workflow JSON. Confirm live class name, then add optional-bypass *if* a graph references it. |
| VideoNoiseWarp | **hyp / no** | — | **no** | Same as LanPaint. |
| FaceID / IPAdapter_plus | **partial (dead UI)** | SDXL ADV **UI** graphs only | easy-use wrappers only; **no** `IPAdapterUnifiedLoader` | UI JSON is not queueable via validator. No API conversion, no director path. |
| ControlNet depth/canny SD1.5 | **partial (dead UI)** | same SDXL ADV UI files | **yes** `ControlNetLoader`, canny/depth preprocessors | Same: UI-only. |
| Realistic Vision SD1.5 | **hyp / no** | — | n/a | Not in `MODEL_FILES` / inventory snapshot. |
| Voronoi / Perlin IMAGE nodes | **no** (Comfy) | — | **yes** `CreateVoronoiMask`, `Image Perlin Power Fractal` | Buddy “fractal” is CPU Mandelbrot (`fractal/render.py`). |
| CPU fractal zoom/inpaint/outpaint | **yes** | `python -m master_agent fractal` | n/a | No recursive Comfy loop (mask → Fun Inpaint → warp → again). |
| RAFT | **hyp / no** | — | **no** `RAFT*` (Recraft* is a different pack) | Confirm live class; not in cache. |
| Qwen / Krea edit | **partial** | manifests + `krea2_img` field map; CCC 4.1 API | **yes** | Not in director allowlist. Character sheet uses Flux, not Krea. |
| MickMumpitz AI-VFX / Movie Builder / CCC | **partial** | large API templates | yes (VACE, CCC_*) | Manual `comfy run --template <slug>` after this PR. Director will not route “possession VFX” here. |
| MMAudio | **no** | music = spectral-flux beats + mux | `OviMMAudioVAELoader` only; no `MMAudioSampler` | Not video→audio generation. |
| SeedVR2 | **yes (post)** | `upscale.py` + `workflows/upscale_seedvr2_api.json` | **yes** | Works as a finisher. RTX method points at gitignored Mickmumpitz filename. |
| Sage-attention | **hyp / partial** | — | **yes** patch nodes | Launch flag not in repo. No graph inserts `*SageAttention*`. Bypass list does not include them (they are not MODEL passthroughs in the TeaCache sense). |
| Movie Builder (LTX 2.3 ADV) | **partial** | `vb_movie_builder` API | mostly yes; `OlmDragCrop` / `PanoramaViewerNode` **missing** from cache | Would fail live validate on those two classes unless packs added or bypassed. |
| Flux t2i (CCC sheets) | **yes (character)** | `character/sheet.py` → variant `flux` | yes | Not a director video variant. |

---

## 4. Recent Buddy work vs this tree

| Claim | In repo? | Where |
|---|---|---|
| Rainey stop-lines / 8n+1 | **yes** | `config.snap_ltx_frames`, `validator`, `AGENTS.md`, `DOWNSCALE_LADDER` first rungs 9/17/25/33 |
| TeaCache **inject** (`ensure_teacache`) | **no** | Only `bypass_optional_accelerators` |
| TeaCache **bypass** | **yes** | `graph_ops.py`, `validator.py`, `tests/test_accelerator_bypass.py` |
| `DOWNSCALE_LADDER` | **yes** | `config.py` + OOM retry in `machine.py` |

Why inject was not (and should not be) bolted onto current graphs:

`WanVideoTeaCache` in the cached registry outputs **`CACHEARGS`**, required widgets `rel_l1_thresh` / `start_step` / `end_step` — it is a WanVideoWrapper sampler argument, not a MODEL inline. `WanVideoTeaCacheKJ` is `KJNodes/deprecated` and wraps MODEL; native `wan22` still would need a measured coeff table (`14B` / `i2v_*`). LTX graphs have no TeaCache node at all. Soft-bypass is the safe pattern.

---

## 5. Gaps that block the two target loops

### Recursive fractal → inpaint

Desired (hypothesis): Comfy fractal/noise or SAM mask → Fun Inpaint / LanPaint → VideoNoiseWarp → repeat.

What exists:

1. CPU fractal inpaint composites Mandelbrot **under a static alpha**. One encode. No Comfy.
2. `WanFunInpaintToVideo` is on the snapshot Comfy and unused.
3. SAM2 is on the snapshot Comfy and unused; preprocess template uses SAM3 and is not chained.
4. LanPaint / VideoNoiseWarp / RAFT are **not** in the snapshot.

**Gap:** no mask producer is wired to any inpaint conditioner, and fractal is a sidecar, not a node.

### Agent-possession VFX

Desired (hypothesis): identity lock (FaceID / Stand-In) + VACE / Mick VFX compositor + control/depth.

What exists:

1. Mick AI-VFX ADV API graphs (`WanVacePhantomSimpleV2`) — template only.
2. `WanVideoAddStandInLatent` unused.
3. FaceID only in non-API SDXL UI graphs.
4. Director cannot say “possession” and leave `base` LTX.

**Gap:** templates exist for VACE compositing; identity + control + agent routing do not.

---

## 6. Recommended wiring plan (impact order)

Do **not** expand `WORKFLOW_FILES` / director allowlist to every manifest slug. A 500-node CCC or VFX graph without a field map will queue the author’s leftover widgets.

1. **Fun Inpaint I2V loop (highest).** New small API template: `LoadImage` + mask (`SAM2Segment` or uploaded mask) → `WanFunInpaintToVideo` → Wan 2.2 I2V (K3NK AIO **or** Wrapper I2V, once weights are named). Patcher fields: prompt, image, mask, length. Optional director keyword `inpaint` / `heal` / `fill`. This is the fractal→inpaint on-ramp.
2. **Confirm live `/object_info` + inventory on the tower.** `python -m master_agent fetch-object-info && python -m master_agent capabilities --offline`. Re-score LanPaint, VideoNoiseWarp, RAFT, K3NK, IPAdapter_plus, Realistic Vision. Until then those rows stay **hyp**.
3. **K3NK AIO as a distinct variant** (`wan22_i2v_aio`), not a silent swap onto `wan22` T2V UNETs. Field map `checkpoint_high` / `checkpoint_low` / `start_image`.
4. **VACE + Stand-In possession preset.** Promote `vb_aivfx_adv` with a tight field map (ref image, control video, prompt) and a director keyword (`vfx`, `possession`, `composite`). Insert `WanVideoAddStandInLatent` only on a Wrapper-based graph.
5. **Comfy fractal plates.** One-shot graph: `CreateVoronoiMask` / `Image Perlin Power Fractal` → IMAGE → Fun Inpaint. Keep CPU fractal for beat-reactive zoom; do not replace it.
6. **VideoNoiseWarp + RAFT** after live class names are known. Optional-accelerator-style **fail-open bypass** if a graph mentions them and the pack is missing — same pattern as TeaCache, but only after I/O types are confirmed (these are not MODEL passthroughs).
7. **FaceID / IPAdapter** — convert one SDXL ADV graph to API **or** drop SD1.5 FaceID in favor of Stand-In / PulID on Wan. UI JSON cannot pass the validator.
8. **MMAudio** — only if a real `MMAudioSampler` (or Ovi path) shows up live; do not confuse with beat-mux.
9. **Movie Builder / CCC** — keep manual (`comfy run --template vb_movie_builder`). Soft-bypass `OlmDragCrop` / `PanoramaViewerNode` if they stay cosmetic.
10. **Power mode** — if ever enabled for VFX, feed `object_info` snippets for *candidate* classes (Fun/VACE/SAM2), not only classes already in the graph. High risk; keep validate-gated.

---

## 7. Easy wins shipped in this PR

1. **`WORKFLOW_FILES["wan22"]`** pointed at a file that does not exist (`260713_MICKMUMPITZ_WAN-2-2-VID_1-0_api.json`). Generate-mode still worked (patcher reads `manifests.yaml`). `comfy run --mode template --template wan22` and `list_templates()` silently omitted the slug. Now points at `260713_VIDEO-BUDDY_WAN-2-2-VID_1-0_api.json`.
2. **`resolve_template` / `list_templates`** accept `manifests.yaml` slugs (`vb_aivfx_adv`, `flux`, `krea2_img`, `vb_movie_builder`, …) without adding them to the director allowlist.
3. **`python -m master_agent capabilities [--offline] [--json]`** prints this matrix from live or cached `object_info`.

Not done (intentionally): TeaCache inject, Fun Inpaint graph, director expansion, RTX upscale path (source JSON is gitignored).

---

## 8. How to re-run the probe on the tower

```powershell
cd video_buddy
.\.venv\Scripts\python.exe -m master_agent health
.\.venv\Scripts\python.exe -m master_agent fetch-object-info
.\.venv\Scripts\python.exe -m master_agent capabilities --json
.\.venv\Scripts\python.exe -m master_agent comfy run --mode template --template vb_aivfx_adv --prepare
```

`--prepare` lints only. A live matrix that still shows Fun Inpaint / TeaCache / VACE as `object_info` hits with `in_buddy=no|partial` is the expected post-0.35.1 result until new templates land.
