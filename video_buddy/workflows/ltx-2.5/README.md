# LTX 2.5 API workflows

Queueable ComfyUI API-format graphs (`POST /prompt`). Ported from
[ltx2.5-research-agent](https://github.com/thcabq352/ltx2.5-research-agent).
Video Buddy lists **all of these in the default catalog** — no env flag.

Stub checkpoint names in the JSON (`ltx-2.5-22b-distilled.safetensors`) are
remapped at patch time to the official `Lightricks/LTX-2.5` Comfy split pack.
See [`../../REQUIRED-FILES.md`](../../REQUIRED-FILES.md).

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
python -m master_agent comfy run --mode generate --variant ltx25_t2v_i2v --prompt "BRIEF"
python -m master_agent run "BRIEF" --variant ltx25_flf2v --no-interview
```

Do not reconvert UI JSON with `/workflow/convert` without re-checking subgraph prefixes.
