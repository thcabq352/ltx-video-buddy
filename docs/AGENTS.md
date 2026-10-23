# Agent notes

Stop-lines and field lessons for anyone driving `master_agent`. Install, Hermes, and L0→L5 are in [Getting started](GETTING_STARTED.md). The loader order is in [Weights](WEIGHTS.md#loader-policy). The matrix is in [Audit](AUDIT.md).

Authoritative framing: **Briefing for LTX Video Buddy (`master_agent`) — fleet curriculum & field notes** (2026-09-10). The stop-lines below are implemented.

## Starting ComfyUI

- **Only `python -m master_agent agent comfy_start` or `python -m master_agent comfy start`.** Both run ComfyUI detached, log to `video_buddy/state/logs/comfyui_8188.log`, and return when `:8188` answers or the launch fails (180 s default). `comfy_status` / `comfy status` checks; `comfy_restart` restarts.
- **NEVER run ComfyUI `main.py` (`python_embeded/python.exe -s ComfyUI/main.py ...`) or `run_*.bat` in a foreground terminal.** ComfyUI never exits, so the call blocks until the tool time limit (Hermes: 420 s) kills it, and ComfyUI dies with it. Field note: the LTX bot did this five times in a row before Hermes halted it for repeating the same call.
- If a raw launch is truly unavoidable, run it with `background=true`, then poll `agent comfy_status` or `GET http://127.0.0.1:8188/system_stats`.
- **After a failed start, do not retry the identical command.** The result carries `error` and `launch.log_tail` (exit code, command, cwd, python). Fix that cause or report it. `starting` means still loading: poll `comfy_status`, do not start again.

## Field lessons

- **`length=8` is junk.** LTX wants `8n+1` (minimum 9). Length 8 collapses to a 1-frame file. Never queue 8. Snap to 9.
- **Know `sec/step` before scale.** Ada field note: 285 s/step dropped to 66.5 s/step after a low-VRAM hull-sized fire. Do not climb `DOWNSCALE_LADDER` until diagnose recorded `sec/step`.
- **Audio field is `frames_number`** on `LTXVEmptyLatentAudio`, paired with video `length`.
- **TeaCache** injects when `/object_info` has `TeaCache` (welltop-cn, LTX `model_type=ltxv`, `rel_l1_thresh=0.06`, `start_percent=0`, `end_percent=1`, `cache_device=cuda`, `max_skip_steps=3` only if the node exposes it). Templates stay clean. If the class is missing, do not insert it. Leftover nodes still bypass. Do not auto-install packs. `WanVideoTeaCache` is not LTX TeaCache. Pack: https://github.com/welltop-cn/ComfyUI-TeaCache
- **LightX2V** (Wan distill LoRA `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32`) is on by default for `wan22` when it is registered: listed in `/object_info` LoRA choices, or on disk when there is no list. Default strength 0.6 (the authored Power Lora slot). Toggle with `create_video` `lightx2v: false` / `lightx2v_strength`, or `--lightx2v on|off` / `--lightx2v-strength` on `run` and `comfy run`. If the file is not registered, it is an INFO no-op and the graph stays as authored. Off switches the Power Lora slot off or removes the core loader and rewires MODEL/CLIP. It does **not** retune the sampler: the graph is a distill schedule (CFG 1, 8 authored steps split 0-3 / 3-end; `DEFAULT_STEPS` overwrites the count). Native Wan 2.2 without it wants about 20 steps at CFG 3.5, so pass those yourself. ClipProvenance records `params.lightx2v` (enabled, requested, strength, lora, action, reason).
- **H3 voice.** H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v. Default photo + voice stays `ltx25_a2v`. CFG stays 1.0. H3 frames use the 17k+5 grid, not `8n+1`.
- **Pack C is hard local-only.** Local-only is a hard requirement: no cloud APIs, no cloud services, no hosted inference. Generate only on `http://127.0.0.1:8188` with `ltx25_t2v_i2v`, `ltx25_flf2v`, or `ltx25_msr`. Partner stubs are field-shape records and are not queued. See [Features](FEATURES.md#pack-c-seedance).
- **Seat rule and Klein 4B.** A ComfyUI submit unloads a GPU llama-server first. Under 20 GB, install sets `LLAMACPP_NGL=0`. 12 GB also sets `VRAM_GB=12` and `FORCE_LOADER=gguf`. FLUX.2 [klein] 4B is the only Flux. `flux_ltx_i2v` stills use it, then `ltx25_t2v_i2v`. First launch: `python -m master_agent first-gate`.
- **Brain ranks stories. Hands answers fit.** Contract `buddy.capability.contract/v1` has no VRAM, slot, or weight path. See [Architecture](ARCHITECTURE.md#brain-and-hands).
- **Missing optional nodes bypass** (`TeaCache`, `WanVideoTeaCache`, `LanPaint_KSampler`, `GetWarpedNoiseFromVideo`, `MMAudioSampler`). Do not bypass structural nodes (`WanFunInpaintToVideo`, `Wan22FunControlToVideo`). FaceID, ControlNet, Voronoi, Perlin, Stand-In, LanPaint, MMAudio, WanVideoWrapper, and K3NK AIO are retired, not deferred. There is no `VideoNoiseWarp` class.
- **Tracker `DONE` can lie.** Confirm with history, `ffprobe`, size, and frames. Junk under 100KB or under 3 frames is FAIL.
- **Port in use is not a reason to kill the cook.** If `:8188` is bound, attach or wait.
- **Budget HOLD** is `input-required`, not failed. `budget reset-shift` archives `previous_used` / `previous_shift_id` and does not wipe history.
- **LoRA A/B locks the encoder.** Do not swap `gemma_3_12B_it_fp8_scaled` for Heretic mid-comparison. Windows folder-prefixed weight names stay backslash style.

## Stop-lines

- **LTX frame law.** `snap_ltx_frames()` never returns 8. Validator warns and auto-corrects unless `--strict`. `DEFAULT_FRAMES`, diagnose, and draft start at 9. `DOWNSCALE_LADDER` first rungs are 9/17/25/33, not 121.
- **Diagnose before scale.** Scale is refused until `state/control/diagnose_hull.json` has `sec_per_step`.
- **TeaCache has one writer:** `graph_ops.ensure_teacache`, called from `load_and_patch_workflow` and `prepare_run`. Do not bake it into JSON templates.
- **LightX2V has one writer:** `graph_ops.apply_lightx2v`, called from `load_and_patch_workflow` and `prepare_run`. `_ensure_lora_node` skips LightX2V names so a second copy is not stacked on the high UNET.
- **Judge look vs health.** Retry uses look only. Low brief adherence plus high look is `human_veto`. See [Judge](JUDGE.md).
- **Shift reset** archives the previous used total, then sets `used=0` and `paused=False`. Diagnose and dry-run do not increment `used`.
- **Tiny preview VAE is not valid with tiled decode.** `taeltx*` / `tae*` preview decoders (including `taeltx2_3`) may feed plain `VAEDecode` only. Tiled decode uses `LTX23_video_vae_bf16.safetensors`.

## Tests

From `video_buddy/`, with the project venv. Prefer one file at a time:

```bash
python -m pytest tests/test_ltx_frames.py tests/test_curriculum.py -q
python -m pytest tests/test_brain_hands.py tests/test_vram_policy.py tests/test_capabilities.py -q
python -m pytest tests/test_judge_split.py tests/test_self_improvement_loop.py -q
```

Do not commit model weights, `state/runs`, or Comfy portable.
