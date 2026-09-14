# LTX 2.5 API workflows

Queueable ComfyUI API-format graphs (`POST /prompt`). Ported from
[ltx2.5-research-agent](https://github.com/thcabq352/ltx2.5-research-agent)
and merged to Video Buddy `main` in PR #6. Video Buddy lists **all of these
in the default catalog** — no env flag.

Stub checkpoint names in the JSON (`ltx-2.5-22b-distilled.safetensors` on
`CheckpointLoaderSimple`) are remapped at patch time. The default loader
follows the 16GB-class pick — **GGUF Q4 → NVFP4 (`VRAM_GB` ≥ 14) →
int8-convrot → official bf16** — and rewrites `CheckpointLoaderSimple` to
`UNETLoader` / `UnetLoaderGGUF` + `LTXAVTextEncoderLoader`. Official bf16
Gemma is not required when a heretic / int8 TE is present.

Baked API defaults are the 16GB hull: **768×512**, **25 frames** (17 on
two-stage), 8 steps, CFG 1.0. The old research plate (1280×720 / 145f) is
the high-VRAM option — pass `--width` / a longer `--duration` only after
`diagnose` has recorded `sec/step`. Two-stage is labeled **offload**.

See [`../../REQUIRED-FILES.md`](../../REQUIRED-FILES.md) and
[`../../docs/QUICKSTART.md`](../../docs/QUICKSTART.md).

| Buddy id | Research id | File | Notes |
|---|---|---|---|
| `ltx25_t2v_i2v` | `t2v_i2v` | `LTX-2.5_T2V_I2V_Single_Stage_Distilled_api.json` | default 2.5 T2V/I2V |
| `ltx25_t2v_i2v_two_stage` | `t2v_i2v_two_stage` | `LTX-2.5_T2V_I2V_Two_Stage_Distilled_api.json` | latent upscale pass |
| `ltx25_t2a` | `t2a` | `LTX-2.5_T2A_Single_Stage_Distilled_api.json` | audio only |
| `ltx25_a2v` | `a2v` | `LTX-2.5_A2V_Two_Stage_Distilled_api.json` | needs LoadAudio |
| `ltx25_v2v_ic_lora` | `v2v_ic_lora` | `LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled_api.json` | IC-LoRA |
| `ltx25_msr` | `msr` | `LTX-2.5_MSR_Multi_Reference_api.json` | pic1–pic4 + background |
| `ltx25_flf2v` | `flf2v` | `LTX-2.5_FLF2V_api.json` | first + last frame |

```bash
python -m master_agent workflows
python -m master_agent doctor
python -m master_agent download-models --ltx25
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "BRIEF"
python -m master_agent comfy run --mode generate --variant t2v_i2v --prompt "BRIEF" --prepare
python -m master_agent run "BRIEF" --variant ltx25_flf2v --no-interview
python -m master_agent run "BRIEF" --variant flf2v --image start.png --no-interview
```

Research aliases (`t2v_i2v`, `flf2v`, …) resolve without an env flag. Frame
counts snap to 8n+1 (min 9). Sibling `ltx_director/workflows/ltx-2.5/` is a
filename fallback if a JSON is missing from this folder.

Do not reconvert UI JSON with `/workflow/convert` without re-checking subgraph prefixes.
