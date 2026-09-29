# vertical concept remake — ltx25_msr

- **status:** recorded
- **machine:** both
- **shot type:** character-locked vertical remake (stadium concert, Clearwater pier)
- **buddy id:** `ltx25_msr`
- **graph:** video_buddy/workflows/ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json
- **why it worked:** The shipped graph is LTX 2.5 Ingredients IC-LoRA: `LTXICLoRALoaderModelOnly` plus `LTXAddVideoICLoRAGuide`. Cached `video_buddy/state/object_info.json` registers those classes and does not register `ComfyUILTX25MSRMultiReferenceGuide`. CoS rejected the rough verticals for morphing, sticker-flat Blaze, and jumbotron garbage text. The remake regenerates from locked stills on that guide path. No tower render is claimed here.
- **watch-outs:** See failure modes and the VRAM default below. Do not queue length 8. Do not pass the rough mp4 as a source video. Do not reuse this recipe for the landscape 30s intro.

## Operator files (not in git)

Keep these on the tower. Do not commit the mp4s or the stills.

| Role | Local path |
|---|---|
| Scott lock | `inputs/blaze/scott.jpg` |
| Blaze lock | `inputs/blaze/blaze.jpg` |
| Blaze alternate still | `inputs/blaze/blaze-alt-1.jpg` (pass as `--blaze` if you want that angle instead) |
| Concert reference, review only | `inputs/blaze/concert-src.mp4` |
| Pier reference, review only | `inputs/blaze/clearwater-pier-src.mp4` |

The rough files are 720×1280, 24 fps, 241 frames (`8n+1`, about 10.04s). They are not graph inputs. `ltx25_v2v_ic_lora` would start from the morphing picture.

## Safe default vs source size

`ltx25_msr` is catalog class `tight`, expected about 13.2 GB, before this canvas. That figure is the existing VRAM table, not a new measurement.

| Canvas | Pixels | Frames | Use |
|---|---|---|---|
| 448×800×97 | 358400 (under 768×512) | 97 = 8×12+1, ~4.04s, 9:16 | **safe default** for the first tower smoke |
| 720×1280×241 | 921600 | 241 = 8×30+1 | rough source geometry. Opt-in only. 720 snaps to **704×1280** on the 32-pixel grid. Unmeasured on 16GB. |

Loader pick on 16GB stays GGUF Q4, then NVFP4, then int8, then bf16. Guide strength stays 1.

`diagnose` still applies the landscape 768×512 clamp, so it is the wrong tool for this vertical. Smoke with `comfy run --prepare`, then queue the same command without `--prepare` only after lint is clean.

Frame cap on this recipe is 241. A 30s landscape intro is a different job and this command will refuse it.

## Tower smoke

From `video_buddy/`:

```powershell
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --recipe blaze-concert --scott inputs/blaze/scott.jpg --blaze inputs/blaze/blaze.jpg --prepare
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --recipe blaze-pier --scott inputs/blaze/scott.jpg --blaze inputs/blaze/blaze.jpg --prepare
```

Queue (same lines, drop `--prepare`) only after prepare lint passes. Filename prefixes are `blaze-concert` and `blaze-pier`.

Prompt override and the unmeasured source size:

```powershell
.\.venv\Scripts\python.exe -m master_agent comfy run --mode generate --recipe blaze-concert --scott inputs/blaze/scott.jpg --blaze inputs/blaze/blaze.jpg --prompt "YOUR LINE" --width 720 --height 1280 --frames 241 --prepare
```

`--prompt` replaces the Positive section for that run. `--negative` replaces the Negative section. Aliases: `concert`, `pier`, `clearwater-pier`.

Provenance is `buddy.clip.provenance/v1`. `params.refs` is Scott then Blaze, bare filenames. `hash` stays null until a real file exists. Prepare writes the sidecar next to `--out` when `--out` is set.

## Failure modes

1. **1-frame collapse.** Length 8 is illegal. `snap_ltx_frames` never returns 8. Do not leave the guide `RepeatImageBatch` amount as a still. See `knowledge/failures/2026-09-29-ltx23-outpaint-1-frame-collapse.md`.
2. **Hallucinated on-screen text.** The concert jumbotron morphed into "MHETLIICCA". Positives ban letters on screens. Negatives ban jumbotron text, captions, subtitles, and that misspelling. Do not ask for burned-in "Clearwater rocks".
3. **Sticker-flat mascots.** Blaze is a second IC-LoRA guide, strength 1, not a 2D paste. Negatives ban sticker, paper cutout, and pasted overlay.
4. **Unlocked Scott face.** Both stills must feed `LTXAddVideoICLoRAGuide`. A plain `ltx25_t2v_i2v` run with no stills does not lock him. Do not drop guide strength below 1 for the lock pass.

Briefs: `knowledge/prompts/2026-09-29-blaze-concert-remake.md` and `knowledge/prompts/2026-09-29-blaze-pier-remake.md`.
