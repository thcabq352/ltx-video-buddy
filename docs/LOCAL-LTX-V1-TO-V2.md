# Local LTX V1 → V2 audit

Video Buddy renders on local Comfy (`:8188`) with local checkpoints and a local
text encoder. This pass did **not** touch the LTX Cloud API, `/v1` or `/v2`
HTTP generation, upload, or async job polling. No `api.ltx.io` caller exists
in this tree.

Deadline context: LTX Cloud API V1 ends 2026-10-26 11:59 PM UTC. The local
stack is already on LTX 2.3 and LTX 2.5. There is no LTX-Video 0.9 / 13B / 2B
checkpoint and no T5 text encoder wired as an LTX loader.

## Remapped

No checkpoint or text-encoder filename was renamed.

The only V1-era LTX weights still loaded are two LTX-2 **19B** IC-LoRAs on the
2.3 lipsync graph. They stay, on purpose (see “Left alone”).

What did change is **routing**, so new work prefers local 2.5 and plate
retake/extend stay on local 2.3 Pro:

| Intent | Local graph | Why |
|---|---|---|
| New scene | `ltx25_t2v_i2v` | LTX 2.5 T2V/I2V graph exists |
| Synced dialogue | `ltx25_a2v` | LTX 2.5 audio-to-video graph exists |
| Multi-cut | `directors` | No 2.5 multi-cut graph. `directors` loads the 2.3 **dev** GGUF (`LTX-2.3-dev-Q4_K_S.gguf`), not distilled `base` |
| Retake / temporal extend, no source video | `directors` | 2.5 has no retake/extend. Dev GGUF is the local Pro slot |
| Retake / temporal extend, source video attached | `lipsync` | Loads `ltx-2.3-22b-dev-fp8.safetensors` |
| Canvas extend (“extend the frame / canvas”) | `ltx23_inoutpaint` | Unchanged. Same dev-fp8 checkpoint |

Lock lives in `video_buddy/master_agent/orchestrator/ltx_routing.py` and is
applied from `director.py` ahead of the LLM. A retake brief cannot fall
through to an `ltx25_*` variant. Hands can still reject a 2.5 pin when that
pack is missing; retake never uses that fallback.

## Left alone (and why)

| Surface | Path | Why it stayed |
|---|---|---|
| LTX 2.3 dev / distilled / EROS checkpoints, Gemma 3 text encoder | `base`, `eros`, `directors`, `lipsync`, `ltx23_inoutpaint`, movie builder | Already local 2.3. Gemma 3 is the 2.3 text encoder. Do not point these graphs at Gemma 4 |
| LTX 2.5 distilled transformer + Gemma 4 | `video_buddy/workflows/ltx-2.5/` | Already local 2.5 |
| `ltx/ltx-2-19b-ic-lora-union-control-ref0.5.safetensors` | lipsync UI + API JSON, `state/download_models.py` | 22B replacement exists (`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`). The lipsync graph note says the older 19B adapter works better than the 22B one on this 2.3 plate |
| `ltx-2-19b-ic-lora-detailer.safetensors` | same lipsync graph | Lightricks `MODELS-LTX-2.3.md` still lists the 19B detailer. No 2.3 file to swap in |
| TeaCache `model_type=ltxv` / `LTX-Video` | `comfy/graph_ops.py` | Comfy node enum for the accelerator, not a checkpoint |
| Flux `t5xxl_fp8_e4m3fn.safetensors` | `flux_t2i.json` | Flux text encoder, not LTX |
| Wan `umt5_xxl_*` | Wan graphs | Not LTX |
| Comfy client `127.0.0.1:8188` | `comfy/client.py` | Local Comfy, not LTX Cloud |
| `state/model_inventory.json` 19B rows | snapshot of files already on disk | Not a loader target. Refreshing it would pretend the 22B union-control file is installed |

## Drift

- Installed-file snapshot `video_buddy/state/model_inventory.json` still lists the 19B union-control and detailer names. That matches the loaders, which still ask for those files.
- `directors.json` on disk names the EROS all-in-one checkpoint. The patcher replaces the MODEL slot with the 2.3 dev GGUF when that file is present (`weights.resolve_ltx23_gguf`). Pro routing depends on that patch, not on a second copy of the graph.
- Default unsure briefs still use `base` (2.3 distilled). The 2.5 preference is for new-scene and synced-dialogue phrasing, not for every prompt. Diagnose hull stays `--variant base`.
- No local retake or temporal-extend graph exists. Those intents reuse `directors` / `lipsync` (2.3 Pro checkpoints) instead of inventing a cloud-style retake node.

## Touched files

- `video_buddy/master_agent/orchestrator/ltx_routing.py` (new)
- `video_buddy/master_agent/orchestrator/director.py`
- `video_buddy/master_agent/orchestrator/prompts/director.md`
- `video_buddy/state/download_models.py` (comment only; URLs unchanged)
- `video_buddy/models/README.md`
- `video_buddy/skills/video-buddy/SKILL.md`
- `video_buddy/AGENTS.md`
- `video_buddy/tests/test_ltx_routing.py`
- `.github/workflows/pytest.yml`
- `docs/LOCAL-LTX-V1-TO-V2.md`
