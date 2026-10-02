> **Archived 2026-10-02.** Superseded by [`docs/INDEX.md`](../../INDEX.md). If this copy disagrees with `docs/`, `docs/` wins.

# Video Buddy capability matrix

One page. Re-run the live rows with `python -m master_agent capabilities --offline`.
Director allowlist is every on-disk `workflows/manifests.yaml` slug (`WORKFLOW_FILES`).
`object_info` validates graphs. It does not synthesize them.

**Retired scope, not deferred work:** the old recursive fractal→inpaint gap, the agent-possession VFX identity-lock gap, and wiring-plan items 1 (Fun Inpaint I2V loop), 3 (K3NK AIO variant), 4 (VACE + Stand-In), 5 (Comfy Voronoi/Perlin plates), and 7 (FaceID API) are closed. Do not treat them as coming soon.

| Capability | Status | How |
|---|---|---|
| LTX 2.5 T2V / I2V / FLF / MSR / A2V / T2A / in-outpaint | **wired** | director `ltx25_*`; inventory-first loaders |
| LTX 2.3 base / eros / directors | **wired** | director; 8n+1 + `DOWNSCALE_LADDER` |
| LTX lipsync / LipDub | **wired** | director when `--video` or lipsync keywords; `ltx25_a2v` for photo + voice |
| MiniMax H3 T2V / I2V / FLF / R2V | **wired** | director `h3_*`; coarse sample-voice mouth, not tight lip-sync |
| Wan 2.2 T2V (native high/low UNET) | **wired** | director `wan22` |
| WAN Fun Inpaint | **wired** | director `wan_fun_inpaint` (start image + `LoadImageMask`) |
| LTX 2.3 in/outpaint | **wired** | director `ltx23_inoutpaint` |
| Qwen / Krea edit | **wired** | director `krea2_img` / `vb_qwen_edit_360` / start-image |
| Mick AI-VFX / Movie Builder / CCC | **wired** | director when the brief names them; heavy on 16GB |
| SAM3 / DepthCrafter preprocess | **wired** | director `vb_aivfx_preprocess` |
| Flux stills | **wired** | director `flux` and the character sheet path |
| CPU fractal zoom / inpaint / outpaint | **wired** | `python -m master_agent fractal` (no Comfy) |
| LTX TeaCache | **wired** | inject-when-registered; missing class soft-bypasses |
| SeedVR2 upscale | **wired** | post-stage only |
| HeartMuLa | **wired** | `heartmula` CLI via heartlib, not Comfy nodes |
| LightX2V LoRAs | **unwired** | baked into wan22 widgets when the file is present; patcher does not toggle them |
| Wan TeaCache | **unwired** | bypass only; do not inject onto native wan22 |
| WAN Fun Control | **unwired** | example graph is gitignored |
| Warp (`GetWarpedNoiseFromVideo`) | **unwired** | soft-bypass if a graph names it; no Buddy graph |
| SAM2 masks | **unwired** | preprocess uses SAM3 |
| Realistic Vision / RAFT / sage-attention | **unwired** | not in Buddy graphs |
| Seedance 2.5 draft pointers | **unwired** | field-shape records; local Comfy only; never queued |
| RTX upscale | **unwired** | gitignored source JSON |
| K3NK WAN AIO I2V | **retired** | no attested pack; not a variant |
| WanVideoWrapper | **retired** | native wan22 does not use `WanVideoSampler` |
| Stand-In | **retired** | no template |
| LanPaint | **retired** | no graph; stray `LanPaint_KSampler` still soft-bypasses |
| FaceID / IPAdapter | **retired** | UI JSON is not a director path |
| ControlNet depth/canny | **retired** | director will not pick it |
| Voronoi / Perlin IMAGE nodes | **retired** | CPU fractal stays; Comfy plates are not a path |
| MMAudio | **retired** | music stays beat-detect + mux |

`doctor` never fetches weights. `download-models` lists confirmed-missing files and downloads only after `--yes`.
