# Rainey1 director preset (Phase 0)

Phase 0 is a **prompt and recipe pack**. It does not train a LoRA, download weights, or run Topaz. LoRA / IC-LoRA stays out of scope until a captioned dataset exists.

Craft target, not a face: locked character, density that escalates by adding world detail, emissive light from more than one practical, anti-slop finish, and a music-video hold. Steal that craft. Do **not** install Jason Rainey's face as an identity asset or default hero. Lock Scott / Blaze / KK / the operator plate in the brief.

## How to call it

A brief that contains `rainey1` pins catalog variant **`base`**. On a 16GB tower, `base` is the distilled 10Eros bake (`DEFAULT_ALL_IN_ONE_CKPT`). The word photoreal in the same brief does not route to Wan.

```powershell
python -m master_agent workflows
python -m master_agent run "rainey1 lock open: photoreal locked character at desk, open PC tower, keyboard, high-key window, identity stable" --no-interview --dry-run
```

`--dry-run` patches and lints only. It does not queue Comfy. `--variant`, `--width`, `--height`, and `--duration` still win when you pass them.

| Recipe | Frames | Size | Use |
|---|---|---|---|
| `rainey1_lock_open` | 9 | 768×512 | Identity desk plate |
| `rainey1_breach` | 17 | 640×384 | Same character, desert / craft breach |
| `rainey1_density` | 25 | 512×384 | Umbilicus / look-up |
| `rainey1_myth_16x9` | 25 | 768×512 | 16:9 myth set-piece |
| `rainey1_story_9x16` | 17 | 640×384 | Vertical story, generated landscape then cropped |

Cue words: `lock open`, `breach`, `density` / `umbilicus` / `look up`, `myth` / `set-piece` / `16:9`, `story` / `9:16` / `reels` / `vertical`. A bare `rainey1` brief uses the lock-open plate.

## Frame law

Latent length is `8n+1`, minimum **9**. `snap_ltx_frames` never returns 8. These recipes use 9, 17, and 25. The 9-frame plate is 0.375s at 24fps. Hands may still report a 1.0s segment floor for a story that short; the patched latent `length` stays 9.

`DOWNSCALE_LADDER` is `(768,512,9) → (640,384,17) → (512,384,25) → (512,320,33)`. Lock, breach, density, and the story plate sit on that ladder. The myth plate starts at 768×512×25, which is a legal length but a heavier pair than the 25-frame rung. If that OOMs, the existing ladder walks; the next legal length after 25 is 33.

## 9:16

`clamp_resolution` caps height at 512 (balanced) or 384 (draft), so a 512×768 portrait latent is not a legal start size. `rainey1_story_9x16` generates the 640×384×17 rung and expects a center crop:

```bash
ffmpeg -i plate.mp4 -vf crop=216:384 story.mp4
```

216×384 is 9:16 from a 384-tall frame (`height * 9/16`).

## Prompts

The negative is the anti-slop block (identity drift, face morph, glow soup, soft beauty filter, watermark, text burned in, and the rest of that list). It is stored on the recipe pack and written into the graph negative.

Positive additives appended to the brief: emissive multi-light, identity lock, zero slop, music-video hold, no text overlay. Each recipe also carries its shot grammar (desk lock, breach, density climax, myth plate, story crop).

Pack file: `master_agent/orchestrator/presets/rainey1.json`.
