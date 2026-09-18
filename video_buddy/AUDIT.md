# Video Buddy × Comfy capability audit

**Scope:** whether `master_agent` discovers and drives the major ComfyUI packs Scott reports on the tower, or only a subset (LTX / old paths).

**Date:** 2026-09-13 (reconciled after merge; 16GB policy in [PR #11](https://github.com/thcabq352/ltx-video-buddy/pull/11))  
**Repo:** `thcabq352/ltx-video-buddy` @ `main`  
**Merged:** [PR #5](https://github.com/thcabq352/ltx-video-buddy/pull/5) (this audit + `capabilities` CLI + wan22 slug fix) then [PR #6](https://github.com/thcabq352/ltx-video-buddy/pull/6) (LTX 2.5 default catalog + scan-first download). [PR #11](https://github.com/thcabq352/ltx-video-buddy/pull/11) adds a shared `vram_policy` so every `manifests.yaml` slug inherits GGUF Q4/Q5 → NVFP4 → int8/fp8; generic VFX routes to `vb_aivfx_adv_13`.  
**Comfy in this checkout:** not present (`video_buddy/ComfyUI_windows_portable/` is gitignored). Live tower was **not** queried from the audit VM.

Post-merge: LTX 2.5 ids are in `WORKFLOW_FILES` and the default catalog. WAN / K3NK wiring from this audit is **unchanged**. LTX TeaCache is inject-when-registered (PR #4). Re-run `python -m master_agent capabilities --offline` for the live matrix.

**Evidence used**

| Source | What it is | What it is not |
|---|---|---|
| `video_buddy/master_agent/**/*.py` | How Buddy builds, lints, and queues graphs | Live tower process flags |
| `video_buddy/workflows/manifests.yaml` + API JSON | Curated templates Buddy *can* load | Proof those packs are installed today |
| `video_buddy/state/object_info.json` (2,823 classes) | Cached `/object_info` snapshot committed in-repo | Live Comfy 0.35.1 on `F:\CircusMoved\...` |
| `video_buddy/state/model_inventory.json` (`generated_at` 2026-08-05) | Weights seen on a Scott tree that day | K3NK / FaceID / Realistic Vision added later |

**Hypothesis (do not treat as verified):** the cached `object_info` is from Scott’s Buddy Comfy (same portable layout as the inventory paths). It is **weeks-to-months old** relative to this audit and must not be cited as live 0.35.1 + sage-attention state.

### Live tower dump (2026-09-13, Comfy Desk)

Operator reported **4114** node classes. The markdown at `/workspace/music-video-lsd-v2/comfy/OBJECT_INFO_CAPABILITIES.md` was **not readable on this VM** (path not mounted). Exact names below are from that follow-up, not from a file we opened.

| Exact live class | In Buddy graphs? | Wiring now |
|---|---|---|
| `WanFunInpaintToVideo` | no | Catalog only. **Not** soft-bypass (payload). |
| `Wan22FunControlToVideo` | no | Catalog only. **Not** soft-bypass. |
| `LanPaint_KSampler` | no | Patcher writes seed/steps/cfg. Soft-bypass if missing. |
| `IPAdapterFaceID` | UI-only SDXL ADV | **Not** soft-bypass. |
| `ControlNetLoader` | `ltx23_lipsync_v08` API + SDXL UI | **Not** soft-bypass. |
| `CreateVoronoiMask` | no | **Not** soft-bypass. CPU fractal is separate. |
| `Image Perlin Power Fractal` | no | Same. |
| `TeaCache` | no (bypass list) | Soft-bypass. Live YES (was missing from old cache). |
| `WanVideoTeaCache` | no (bypass list) | Soft-bypass. Outputs CACHEARGS on old cache schema. |
| `GetWarpedNoiseFromVideo` | no | **Preferred warp name.** Soft-bypass. There is **no** `VideoNoiseWarp`. |
| `MMAudioModelLoader` / `Sampler` / `VoCoder` | no | Soft-bypass. No exact class named `MMAudio`. |

Buddy still does not *queue* Fun Inpaint / LanPaint / warp / FaceID / Voronoi. Soft-bypass only keeps validation from hard-failing if a future graph names those optional enhancers on a machine that lacks the pack.

---

## 1. How Buddy builds and queues Comfy graphs

Buddy never synthesizes a graph from `/object_info`. The path is **template → patch → lint → `/prompt`**.

```
brief ──▶ director.choose_variant (every manifests.yaml slug)
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
| `video_buddy/master_agent/comfy/capabilities.py` | **PR #5** — read-only gap probe (`capabilities` CLI) |
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
| Director | LLM + rules pick among **every** `manifests.yaml` slug (`WORKFLOW_FILES` is derived from the manifest). Hard constraints (`--variant`, `--video` → lipsync) still win. | `load_workflow_files()`, `_VARIANT_KEYWORDS` |
| Patcher field maps | No | `manifests.yaml` `fields:` for core LTX/Wan/Flux/H3 plus aivfx / movie_builder / qwen / ideogram / remaining `ltx25_*`. CCC ADV / preprocess / gitignored renderers stay baked-defaults. |
| Heuristic patch | No | Class lists: `EmptyLTXVLatentVideo`, `KSampler`, `UNETLoader`, `WanVideoNAG` is **not** specially handled (wan22 uses named node ids) |
| Validator | Yes — live/cache registry | Unknown `class_type` is a **hard fail** except optional nodes (TeaCache, `LanPaint_KSampler`, `GetWarpedNoiseFromVideo`, `MMAudio*`) |
| TeaCache | Inject-when-registered + soft-bypass | `ensure_teacache` + `OPTIONAL_ACCELERATOR_CLASS_TYPES` in `graph_ops.py` |
| `comfy run --template` | Path or slug | **PR #5:** slugs also resolve from `manifests.yaml` |
| Fractal | N/A | Numpy + ffmpeg. No Voronoi/Perlin/SAM/Fun nodes |
| Music | Beat map + same director graphs | No MMAudio sampler |
| KB `search_workflows` | Embedding over ingested JSON | Retrieval ≠ queue |

**`ensure_teacache` is the single LTX writer (PR #4).** When `TeaCache` is in `/object_info`, the patcher / `prepare_run` insert welltop-cn TeaCache on LTX MODEL after LoRA. If the class is missing, inject is a no-op and leftover nodes still WARNING + rewire. Packs are never auto-installed.

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
| LTX 2.5 distilled T2V / I2V / FLF / MSR / A2V / T2A | **yes** | director `ltx25_*` (+ research aliases); default catalog; inventory-first loaders | graphs ship `EmptyLTXVLatentVideo` / `LTXVImgToVideo` / `ComfyUILTX25MSR*` | **PR #6.** No env flag. Doctor reports GGUF→NVFP4→int8→bf16. |
| LTX 2.3 short I2V / T2V | **yes** | director `base`/`eros`/`directors`; patcher; 8n+1 + `DOWNSCALE_LADDER` | yes (`EmptyLTXVLatentVideo`, …) | Wired. Still the default when the brief does not name 2.5. |
| LTX lipsync | **yes** | director when `--video` / lipsync keywords | yes | Wired. |
| Wan 2.2 T2V (Mick native high/low UNET) | **yes** | director `wan22` → `260713_VIDEO-BUDDY_WAN-2-2-VID_1-0_api.json` | yes (`WanVideoNAG`) | **Was broken** as `comfy run --template wan22` (stale `MICKMUMPITZ_*` filename). Fixed in PR #5. Still T2V UNETs, not I2V AIO. |
| K3NK WAN 2.2 AIO I2V HIGH/LOW fp8 | **hyp / no** | — | no class; no inventory name | Need weights + I2V template + `MODEL_FILES` keys. Do not spray onto `wan22` T2V loaders. |
| WanVideoWrapper | **no** | — | **yes** (119 `WanVideo*` classes) | Native `wan22` uses `UNETLoader`+`KSamplerAdvanced`, not `WanVideoSampler`. Wrapper is unused. |
| WAN Fun Inpaint | **no** | — | **yes** (cache + **live** `WanFunInpaintToVideo`) | No graph, no mask ingest, no patcher fields. Highest-value missing node for recursive inpaint. |
| WAN Fun Control | **no** | manifests `vb_zimage_turbo_cn` only | **yes** (live `Wan22FunControlToVideo`) | Example file lives under gitignored `AI-RENDERING-EXAMPLE FILES/`. |
| LightX2V LoRAs | **partial** | baked in wan22 Power Lora widgets | n/a (weights) | Inventory has `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32`. Patcher cannot toggle it. |
| Wan TeaCache | **partial (bypass)** | validator drops missing accel | **live YES** `TeaCache` + `WanVideoTeaCache` | `WanVideoTeaCache` is CACHEARGS — do not inject onto native wan22. welltop-cn `TeaCache` on LTX is inject-when-registered (PR #4). |
| VACE | **yes (director)** | director `vb_aivfx_adv` / `_13` (keywords: vfx, possession, vace) | **yes** | Field map writes `IterPromptBuilder.string_1`. No Stand-In wiring. |
| Stand-In | **no** | — | **yes** `WanVideoAddStandInLatent` | Wrapper embed node; needs a Wrapper graph. |
| SAM2 masks | **no** | — | **yes** `SAM2Segment` | AI-VFX preprocess uses **SAM3**, and is director-routed (`vb_aivfx_preprocess`) with baked widgets. |
| LanPaint | **partial (bypass + patcher)** | seed/steps/cfg if a graph uses `LanPaint_KSampler` | **live YES** `LanPaint_KSampler` (not `LanPaint`) | No shipped graph. Soft-bypass if pack missing. |
| Warp / VideoNoiseWarp | **partial (bypass)** | — | **live related** `GetWarpedNoiseFromVideo` | **No exact `VideoNoiseWarp`.** Prefer `GetWarpedNoiseFromVideo`. No graph. |
| FaceID / IPAdapter_plus | **partial (dead UI)** | SDXL ADV **UI** graphs only | **live YES** `IPAdapterFaceID` | UI JSON is not queueable via validator. No API conversion, no director path. |
| ControlNet depth/canny SD1.5 | **partial (dead UI)** | same SDXL ADV UI files | **yes** `ControlNetLoader`, canny/depth preprocessors | Same: UI-only. |
| Realistic Vision SD1.5 | **hyp / no** | — | n/a | Not in `MODEL_FILES` / inventory snapshot. |
| Voronoi / Perlin IMAGE nodes | **no** (Comfy) | — | **live YES** `CreateVoronoiMask`, `Image Perlin Power Fractal` | Buddy “fractal” is CPU Mandelbrot (`fractal/render.py`). |
| CPU fractal zoom/inpaint/outpaint | **yes** | `python -m master_agent fractal` | n/a | No recursive Comfy loop (mask → Fun Inpaint → warp → again). |
| RAFT | **hyp / no** | — | **no** `RAFT*` (Recraft* is a different pack) | Confirm live class; not in cache. |
| Qwen / Krea edit | **yes (director)** | `krea2_img` / `vb_qwen_edit_360` / `vb_aivfx_startimage` field maps; CCC 4.1 API | **yes** | Director-routed. CCC 4.1 still baked-defaults. |
| MickMumpitz AI-VFX / Movie Builder / CCC | **yes (director)** | large API templates + field maps where safe | yes (VACE, CCC_*) | Director routes `vfx` / `movie builder` / `ccc`. CCC ADV + gitignored renderers queue baked widgets — `--template` for precision. |
| MMAudio | **partial (bypass)** | music = spectral-flux beats + mux | **live related** `MMAudioModelLoader` / `Sampler` / `VoCoder` | No exact class named `MMAudio`. Not wired as a generator. |
| SeedVR2 | **yes (post)** | `upscale.py` + `workflows/upscale_seedvr2_api.json` | **yes** | Works as a finisher. RTX method points at gitignored Mickmumpitz filename. |
| Sage-attention | **hyp / partial** | — | **yes** patch nodes | Launch flag not in repo. No graph inserts `*SageAttention*`. Bypass list does not include them (they are not MODEL passthroughs in the TeaCache sense). |
| Movie Builder (LTX 2.3 ADV) | **yes (director)** | `vb_movie_builder` API + first-shot `PrimitiveStringMultiline` field map | mostly yes; `OlmDragCrop` / `PanoramaViewerNode` **missing** from cache | Would fail live validate on those two classes unless packs added or bypassed. |
| Flux t2i (CCC sheets) | **yes** | `character/sheet.py` + director `flux` (character sheet / text-to-image) | yes | Still the stills path, now also director-routable. |

---

## 4. Recent Buddy work vs this tree

| Claim | In repo? | Where |
|---|---|---|
| LTX 2.5 default catalog + scan-first download | **yes (PR #6)** | `workflows/ltx-2.5/`, `catalog.py`, `models/weights.py`, `doctor` / `download-models --ltx25` |
| Rainey stop-lines / 8n+1 | **yes** | `config.snap_ltx_frames`, `validator`, `AGENTS.md`, `DOWNSCALE_LADDER` first rungs 9/17/25/33 |
| TeaCache **inject** (`ensure_teacache`) | **yes (PR #4)** | `graph_ops.ensure_teacache` from patcher + `prepare_run` |
| TeaCache **bypass** | **yes** | `graph_ops.py`, `validator.py`, `tests/test_accelerator_bypass.py` |
| `DOWNSCALE_LADDER` | **yes** | `config.py` + OOM retry in `machine.py` |

Wan vs LTX TeaCache:

`WanVideoTeaCache` in the cached registry outputs **`CACHEARGS`**, required widgets `rel_l1_thresh` / `start_step` / `end_step` — it is a WanVideoWrapper sampler argument, not a MODEL inline. `WanVideoTeaCacheKJ` is `KJNodes/deprecated` and wraps MODEL; native `wan22` still would need a measured coeff table (`14B` / `i2v_*`). Do not inject those Wan aliases. LTX graphs stay template-clean; **PR #4** injects welltop-cn `TeaCache` when the class is registered, and still soft-bypasses leftovers when it is not.

---

## 5. Gaps that block the two target loops

### Recursive fractal → inpaint

Desired (hypothesis): Comfy fractal/noise or SAM mask → Fun Inpaint / `LanPaint_KSampler` → `GetWarpedNoiseFromVideo` → repeat.

What exists:

1. CPU fractal inpaint composites Mandelbrot **under a static alpha**. One encode. No Comfy.
2. Live tower has `WanFunInpaintToVideo`, `CreateVoronoiMask`, `Image Perlin Power Fractal`, `LanPaint_KSampler`, `GetWarpedNoiseFromVideo` — **none** are in a Buddy-queued graph.
3. SAM2 is on the old cache and unused; preprocess template uses SAM3 and is not chained.

**Gap:** no mask producer is wired to any inpaint conditioner, and fractal is a sidecar, not a node.

### Agent-possession VFX

Desired (hypothesis): identity lock (FaceID / Stand-In) + VACE / Mick VFX compositor + control/depth.

What exists:

1. Mick AI-VFX ADV API graphs (`WanVacePhantomSimpleV2`) — director-routed (`vfx` / `possession` / `vace` → **`vb_aivfx_adv_13`** GGUF; explicit “1.0” still picks the heavy e4m3fn graph).
2. `WanVideoAddStandInLatent` unused.
3. Live `IPAdapterFaceID` exists; Buddy only has it in non-API SDXL UI graphs.
4. Director keywords now pick `vb_aivfx_adv` for “possession”; identity lock is still missing.

**Gap:** VACE compositing is director-routable; identity + control loops are not.

---

## 6. Recommended wiring plan (impact order)

**Historical (superseded):** this audit originally said not to expand
`WORKFLOW_FILES` / the director allowlist to every manifest slug, because a
500-node CCC or VFX graph without a field map will queue the author’s leftover
widgets. **Scott override (2026-09-13):** route **all** Video Buddy workflows
through the director. Mitigation: derive the allowlist from `manifests.yaml`,
add safe field maps where widgets are obvious, and document baked-default
graphs so operators prefer `--template` for precision.

1. **Fun Inpaint I2V loop (highest).** New small API template: `LoadImage` + mask (`SAM2Segment` or uploaded mask) → `WanFunInpaintToVideo` → Wan 2.2 I2V (K3NK AIO **or** Wrapper I2V, once weights are named). Patcher fields: prompt, image, mask, length. Optional director keyword `inpaint` / `heal` / `fill`. This is the fractal→inpaint on-ramp.
2. **Live class names are now known** (Desk dump). Remaining **hyp**: K3NK weights, Realistic Vision, RAFT exact class, sage launch flag. Copy the 4114-class `/object_info` into `state/` when someone can run `fetch-object-info` on the tower.
3. **K3NK AIO as a distinct variant** (`wan22_i2v_aio`), not a silent swap onto `wan22` T2V UNETs. Field map `checkpoint_high` / `checkpoint_low` / `start_image`.
4. **VACE + Stand-In possession preset.** Promote `vb_aivfx_adv` with a tight field map (ref image, control video, prompt) and a director keyword (`vfx`, `possession`, `composite`). Insert `WanVideoAddStandInLatent` only on a Wrapper-based graph.
5. **Comfy fractal plates.** One-shot graph: `CreateVoronoiMask` / `Image Perlin Power Fractal` → IMAGE → `WanFunInpaintToVideo`. Keep CPU fractal for beat-reactive zoom; do not replace it.
6. **Warp via `GetWarpedNoiseFromVideo`** (not `VideoNoiseWarp`). Soft-bypass is in; still need a graph that feeds warp into Fun Inpaint / LanPaint.
7. **FaceID** — live class is `IPAdapterFaceID`. Convert one SDXL ADV graph to API **or** drop SD1.5 FaceID in favor of Stand-In / PulID on Wan.
8. **MMAudio** — use `MMAudioModelLoader` / `MMAudioSampler` / `MMAudioVoCoder`. Soft-bypass is in; music pipeline is still beat-mux.
9. **Movie Builder / CCC** — now director-routable. Soft-bypass `OlmDragCrop` / `PanoramaViewerNode` if they stay cosmetic. Prefer `--template` when you need every leftover widget left alone.
10. **Power mode** — if ever enabled for VFX, feed `object_info` snippets for *candidate* classes (Fun/VACE/SAM2), not only classes already in the graph. High risk; keep validate-gated.

---

## 7. Easy wins shipped in PR #5 (merged)

1. **`WORKFLOW_FILES["wan22"]`** pointed at a file that does not exist (`260713_MICKMUMPITZ_WAN-2-2-VID_1-0_api.json`). Generate-mode still worked (patcher reads `manifests.yaml`). `comfy run --mode template --template wan22` and `list_templates()` silently omitted the slug. Now points at `260713_VIDEO-BUDDY_WAN-2-2-VID_1-0_api.json`.
2. **`resolve_template` / `list_templates`** accept `manifests.yaml` slugs (`vb_aivfx_adv`, `flux`, `krea2_img`, `vb_movie_builder`, …). Those slugs are now also director-routable (allowlist derived from the manifest).
3. **`python -m master_agent capabilities [--offline] [--json]`** prints this matrix from live or cached `object_info`.
4. **Live-name follow-up:** catalog + soft-bypass use exact Desk names (`LanPaint_KSampler`, `GetWarpedNoiseFromVideo`, `TeaCache`). Patcher writes seed/steps/cfg on `LanPaint_KSampler`. Fun Inpaint / Fun Control / FaceID stay fail-closed.

Not done (intentionally): Fun Inpaint graph, K3NK AIO I2V, RTX upscale path (source JSON is gitignored). LTX TeaCache inject-when-registered landed in **PR #4**. LTX 2.5 catalog expansion landed separately in **PR #6**.

### Real CLI examples (post-merge)

```bash
cd video_buddy
python -m master_agent capabilities --offline
python -m master_agent capabilities --json
python -m master_agent workflows
python -m master_agent comfy run --mode template --template wan22 --prepare
python -m master_agent comfy run --mode template --template vb_aivfx_adv --prepare
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "neon rain" --prepare
```

`--offline` uses `state/object_info.json`. `--json` includes `director_allowlist` (every `WORKFLOW_FILES` key, including `ltx25_*`) and per-row `in_buddy` / `object_info_hits`.

---

## 8. How to re-run the probe on the tower

```powershell
cd video_buddy
.\.venv\Scripts\python.exe -m master_agent health
.\.venv\Scripts\python.exe -m master_agent fetch-object-info
.\.venv\Scripts\python.exe -m master_agent capabilities --json
.\.venv\Scripts\python.exe -m master_agent comfy run --mode template --template vb_aivfx_adv --prepare
```

`--prepare` lints only. A live matrix that still shows Fun Inpaint / TeaCache / VACE as `object_info` hits with `in_buddy=no|partial` is the expected post-0.35.1 result until new templates land. LTX 2.5 rows should read `in_buddy=yes` after PR #6.
