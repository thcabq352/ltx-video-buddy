# Model selection plan

Short plan for the reasoned route + GGUF decision on branch `cursor/10eros-v15-gguf-a523` (draft PR #70). This file is the checklist. The call-site inventory is `docs/MODEL_SELECTION_AUDIT.md`.

## Rules for this branch

- Commit and push only to `cursor/10eros-v15-gguf-a523`.
- PR #70 stays a draft. Do not merge it and do not mark it ready.
- Do not merge or modify PR #92 or `cursor/gguf-vram-flux-ltx`.
- Do not rebase onto main or force-push unless a normal push is rejected.

## What is wrong today

Diffusion choice is a silent list walk. `_apply_ltx23_gguf_model` keeps `10Eros_v1.5-Q4_K_M.gguf` when that file is on disk, otherwise `resolve_ltx23_gguf` returns the first on-disk name from `LTX23_DISTILLED_GGUF` or `LTX23_DEV_GGUF`, otherwise a folder scan. Nothing explains the pick. If nothing is on disk, the missing name is left in the graph and ComfyUI fails later.

The director ranks workflow stories (rules, then an allowlisted LLM). It does not choose a diffusion file, and it does not prefer a Flux still → LTX I2V route. MiniMax H3 is a keyword route to local Comfy graphs (`h3_t2v` / `h3_i2v` / `h3_flf` / `h3_r2v`), not a scored candidate next to the LTX GGUFs.

## Decision shape

One function, two steps, one text block.

1. **Route**, before any GGUF score.
   - User force / explicit variant wins. Warn if that choice does not fit VRAM.
   - An input image means LTX I2V (or H3 I2V when the user asked for MiniMax), not a new Flux still.
   - Otherwise, when a Flux I2V workflow file and a Flux diffusion file are both on disk and the combined VRAM fits, prefer Flux-makes-the-keyframe then LTX animates it, over plain LTX text-to-video. This applies for whatever LTX generation is installed (2.3, 2.5, and a talking MiniMax H3 graph when that route wins).
   - Plain text-to-video is the route when Flux I2V is absent, does not fit, or the user asked for T2V.
2. **Model**, among on-disk candidates for that route.
   - LTX 2.3: Sulphur `sulphur_dev-Q3_K_S.gguf`, QuantStack distilled `LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf`, distilled Q3 aliases, dev `LTX-2.3-dev-Q4_K_S.gguf`, 10Eros v1.5 `10Eros_v1.5-Q4_K_M.gguf`, plus any other LTX 2.3 GGUF the folder scan finds.
   - LTX 2.5 distilled GGUF when that generation is the route.
   - MiniMax H3 local GGUFs (`minimax_h3_fl2va_pruned-Q4_K.gguf`, `minimax_h3_ref2va_pruned-Q4_K.gguf`) when the route is H3.

The text names the winner and why, and a reason for every rejected alternative. Pairwise lines are always present for the comparisons Scott asked about: Sulphur vs base distilled, Eros preset vs the 10Eros v1.5 file, Q4 vs Q3, dev vs distilled, and MiniMax H3 vs LTX for talking / lip-sync.

## Catalog

Each candidate carries: filename and aliases, family (`ltx23` / `ltx25` / `h3` / `flux`), distilled vs dev vs style, quality tier, quant label, approximate file size, approximate VRAM, intent tags (general, fast, eros, directors, sulphur, talking, lipsync).

`workflows/ltx23_av_presets.yaml` keeps the three presets. Each row gains:

- `intent`: `base` = general/fast distilled, `eros` = 10Eros style, `directors` = dev quality.
- optional `model`: explicit filename. It wins. A VRAM miss is a warning, not a silent swap.
- `prefer`: order used only as a tie-break inside the same intent and tier.

`unet_name` stays the authored graph seed so template loads do not change. The selector rewrites the loader at patch time and records why.

## Scoring

Eligible only if the file is on disk and `vram_gb` is within declared or probed free VRAM. Declared VRAM is `config.VRAM_GB` (env, else `nvidia-smi` total, else 16). Probed free VRAM is `nvidia-smi` `memory.free` when that binary exists; tests can pass the number in. This branch does not have PR #92's `probe_cuda_mem`, and that function is not copied.

Score = intent match, then quality tier, then the preset `prefer` order. A file that does not fit VRAM is rejected with "exceeds VRAM" and a lower tier that fits can win (the Q4 → Q3 downgrade). Not on disk, lower tier, and intent mismatch are the other rejection reasons.

No eligible file: raise before ComfyUI is queued. The message lists what is missing and which catalog rows would have fit the VRAM budget.

## Flux I2V hook

`workflows/flux_ltx_i2v_api.json` and the `flux_ltx_i2v` manifest slug live on draft PR #92 (`cursor/gguf-vram-flux-ltx`). They are not on this branch. The detector looks for that filename (or any manifest slug whose file name contains both `flux` and `i2v`) and for a Flux diffusion weight. Preference order for the still: `flux-2-klein-4b-fp8.safetensors` (main's PR #100, the only Flux stills model in current policy), then another on-disk Klein or Flux GGUF/fp8 already named in this tree (`flux-2-klein-9b-fp8`, `flux1-dev-Q4_K_S`, `flux1-dev-fp8`). If the workflow file is absent, the route reason says so and names the PR #92 dependency. No graph is copied from that branch.

Combined VRAM is the Flux still plus the LTX (or H3) model that would animate it. If that sum exceeds free VRAM, Flux I2V is rejected and T2V (or the user's image route) is why.

## MiniMax H3

H3 in this repo is MiniMax H3: local Comfy graphs under `workflows/minimax-h3/`, GGUF DiT, Comfy text encoder and VAEs. The manifest comment "API pack" means Comfy API JSON, not a paid HTTP API. There is no MiniMax/Hailuo HTTP client here.

- Local weights plus the workflow file: H3 may win when the brief needs talking or lip-sync (or the user named MiniMax / Hailuo / H3). The reason says local, no per-call charge, and why it beat or lost to LTX and Flux.
- Env `MINIMAX_API_KEY`, `HAILUO_API_KEY`, `H3_API_KEY`, or `H3_API_BASE` with no local weights is treated as paid hosted. It is never auto-selected. `H3_OPT_IN=1` or an explicit user ask is required, and the reason states the hosted/cost factor. This repo still cannot place that hosted call; the decision says that instead of pretending to.
- Missing local weights and no opt-in: skip, with the reason.

## Where the text is shown

- Director log line and the run state's `model_decision` (job JSON via `RunState.to_dict`).
- `load_and_patch_workflow` meta (`model_decision`), which the orchestrator already stores on the run.
- `doctor` row `ltx23-weights` (and the route sentence on `vram-policy` / the selector report).
- `python -m master_agent` model selector output when version 2.3 is assessed.
- Logger `master_agent.models.model_decision`.

## What stays out of the scorer

Documented in the audit, not folded into the LTX catalog:

- Ingest `classify.py` only names which widget is the checkpoint role.
- `loader_names.py` only fixes Comfy combo spelling.
- `resolve_sulphur_lora` looks up a filename the graph already chose.
- LLM GGUF (`LLAMACPP_MODEL`, `KB_EMBED_MODEL`) is the director's language model, not the video unet.
- Wan, VACE, Krea, Qwen, and Flux character-sheet loaders keep `preference_order`. The new decision owns the Flux still only when the route is Flux I2V.
- Text-encoder heretic swap and the safetensors `CHECKPOINT_FALLBACKS` scan stay for non-GGUF slots. LTX 2.3 `base` / `eros` / `directors` diffusion no longer uses that scan.

## Tests

New tests, no GPU:

- Intent match: base → distilled Q4, eros → 10Eros, directors → dev Q4, when all are on disk.
- VRAM downgrade: Q4 rejected, Q3 chosen, reason says exceeds VRAM.
- `model:` override wins, including a warning when it does not fit.
- Nothing on disk raises and the message lists missing names and what would fit.
- Pairwise sentences for Sulphur vs base, Eros vs 10Eros, Q4 vs Q3, dev vs distilled, H3 vs LTX.
- Flux I2V preferred only when the workflow file and a Flux weight are both present and VRAM fits; absent file keeps T2V and cites PR #92.
- Hosted H3 key is not auto-picked; opt-in and explicit user choice are stated.
- User image selects LTX I2V, not Flux.
- Existing suite stays green. `test_gguf_first.py` assertions that require every preset to stay on 10Eros are updated to the reasoned picks. Template-seed assertions (`unet_name` in the preset YAML and `load_workflow_template`) stay.

## Checklist

- [x] Write this plan and push it on `cursor/10eros-v15-gguf-a523`.
- [x] Write `docs/MODEL_SELECTION_AUDIT.md` (file, line, current logic, silent or not, planned change, Flux and H3 notes) and push it on its own.
- [x] Add the candidate catalog and `select_model` / `decide_generation` with structured reasons.
- [x] Declare intent, optional `model`, and `prefer` on the three presets.
- [x] Route patcher, `resolve_ltx23_gguf`, director, doctor, and the selector report through the decision. Leave the audit's exemptions alone.
- [x] Fail the job before ComfyUI when no GGUF fits. Override still wins, with a VRAM warning. Dry-run records the error and still builds the graph.
- [x] Surface the same text in the director log, run JSON, patch meta, and doctor.
- [x] Tests listed above. Full `video_buddy` pytest suite green aside from the Flux fp8 assertion that already fails on `origin/main`.
- [x] Update `docs/WEIGHTS.md` and the README weights pointer. Update draft PR #70's body. Confirm the PR is still draft and unmerged.
