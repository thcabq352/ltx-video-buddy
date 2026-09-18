# LTX 2.5 API workflows

Queueable ComfyUI API-format graphs (`POST /prompt`). Converted from the
official Lightricks UI examples at
[Lightricks/ComfyUI-LTXVideo `example_workflows/2.5`](https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.5)
(`class_type` + `inputs` dicts). Video Buddy lists **all of these in the
default catalog** — no env flag.

Each video graph is loader → text encode / conditioning → sample →
decode (`VAEDecodeTiled` / `LTXVTiledVAEDecode`) → `CreateVideo` +
`SaveVideo`. `ltx25_t2v_i2v` uses official `LTXVImgToVideoInplace`
(bypass when no still; first-frame latent when `image_name` is set).
`ltx25_flf2v` is the same single-stage graph plus a last-frame insert
(`LTXVImgToVideoInplaceKJ`). `ltx25_msr` is the official Ingredients
multi-reference sheet graph saved under the Buddy MSR filename.

The patcher still rewrites `CheckpointLoaderSimple` → `UNETLoader` /
`UnetLoaderGGUF` and remaps local GGUF / heretic TE names onto the
official loaders. Official bf16 Gemma is not required when a heretic /
int8 TE is present. TeaCache stays inject-when-registered.

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
