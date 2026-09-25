# MiniMax H3 API workflows

**Public demo (photoreal):** golden-hour BMX berm → backflip → dusty stick,
~8s, MiniMax H3 via Video Buddy / Comfy. Poster + provenance live at
[`docs/demo/`](../../../docs/demo/) (in-repo, no
Tailscale).

Queueable ComfyUI API-format graphs (`POST /prompt`). Native ComfyUI ≥0.30
nodes (`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`) plus
`UnetLoaderGGUF` (ComfyUI-GGUF). Video Buddy lists **all of these in the
default catalog** — no experimental flag.

16GB-class default (`VRAM_GB=16`): **GGUF Q4_K DiT** + Comfy Qwen3-VL TE
(NVFP4 AWQ on Blackwell, else int8 / int4 convrot) + official video/audio
VAEs. CFG stays **1.0** (distilled). Sweet spot **0.6–0.8 MP**, **≤12 s**,
**4 steps** (LightX2V turbo LoRA when present).

See [`../../REQUIRED-FILES.md`](../../REQUIRED-FILES.md) and
[`../../docs/QUICKSTART.md`](../../docs/QUICKSTART.md).

| Buddy id | Alias | File | Mode |
|---|---|---|---|
| `h3_t2v` | `fl2va`, `h3_fl2va` | `MiniMax-H3_T2V_FL2VA_api.json` | fl2va text-to-AV |
| `h3_i2v` | — | `MiniMax-H3_I2V_FL2VA_api.json` | fl2va image-to-AV |
| `h3_flf` | — | `MiniMax-H3_FLF_FL2VA_api.json` | fl2va first + last frame |
| `h3_r2v` | `ref2va`, `h3_ref2va` | `MiniMax-H3_R2V_REF2VA_api.json` | ref2va voice reference (not lip-sync) |

```bash
python -m master_agent workflows
python -m master_agent doctor
python -m master_agent download-models --h3
python -m master_agent comfy run --mode generate --variant h3_t2v --prompt "BRIEF"
python -m master_agent comfy run --mode generate --variant fl2va --prompt "BRIEF" --prepare
python -m master_agent run "BRIEF" --variant h3_i2v --image start.png --no-interview
python -m master_agent run "BRIEF" --variant h3_r2v --image ref.png --no-interview
python -m master_agent run "she says the line" --variant h3_r2v --image face.png --audio line.wav --no-interview
```

`h3_r2v` accepts a reference still plus reference audio on dotted autogrow
slots (`ref_images.ref_image_0`, `ref_audios.ref_audio_0`).
h3_r2v uses your audio as a voice reference; it doesn't lip-sync to it. Use ltx25_a2v for a supplied voice.
The route still goes to `h3_r2v` when the brief names MiniMax, Hailuo, H3,
or ref2va, and the CLI / studio / MCP response warn with that sentence.
fl2va graphs do not take a voice file. One H3 clip, capped at 12s.
Default photo + voice stays `ltx25_a2v`.

Frame counts snap to the H3 **17k+5** grid (124 ≈ 5 s at 24 fps). Do not
use LTX `8n+1` lengths on these graphs. LTX 2.5 / WAN paths are unchanged.
