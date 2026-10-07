# Features

Catalog behavior lives here. Loader order and consent live in [Weights](WEIGHTS.md#loader-policy). Wired versus retired lives in [Audit](AUDIT.md).

## Default catalog

`python -m master_agent workflows` lists these with no env flag. Research aliases resolve the same way. `--prepare` does not require weights or a GPU queue.

| Buddy id | Alias | What it does |
|---|---|---|
| `ltx25_t2v_i2v` | `t2v_i2v` | Single-stage distilled T2V / I2V |
| `ltx25_t2v_i2v_two_stage` | `t2v_i2v_two_stage` | Latent spatial upscale |
| `ltx25_flf2v` | `flf2v` | First + last frame |
| `ltx25_msr` | `msr` | pic1–pic4 + background |
| `ltx25_v2v_ic_lora` | `v2v_ic_lora` | Video-to-video IC-LoRA |
| `ltx25_a2v` | `a2v` | Audio-to-video. Keeps the supplied voice |
| `ltx25_t2a` | `t2a` | Text-to-audio |
| `ltx25_inoutpaint` | — | Inpaint / outpaint |
| `h3_t2v` | `fl2va` | MiniMax H3 text-to-AV, native stereo |
| `h3_i2v` | — | H3 image-to-AV |
| `h3_flf` | — | H3 first + last frame |
| `h3_r2v` | `ref2va` | H3 reference: a written line in a 2–12s sample voice |

Graph files: [`video_buddy/workflows/ltx-2.5/`](../video_buddy/workflows/ltx-2.5/README.md), [`video_buddy/workflows/minimax-h3/`](../video_buddy/workflows/minimax-h3/README.md).

H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.

A brief that names MiniMax, Hailuo, H3, or ref2va still routes to `h3_r2v` and must warn with that sentence. fl2va does not take a voice file. One H3 clip, capped at 12s. Default photo + voice stays `ltx25_a2v`. Pass the exact words with `--line`. Samples longer than 12s are trimmed to the loudest 12s.

LTX 2.3 (`base`, `eros`, `directors`, `lipsync`), Wan 2.2 (`wan22`), Fun Inpaint (`wan_fun_inpaint`), and the Mick graphs stay on the director allowlist. Large graphs that queue with baked leftover widgets are safer as `comfy run --template <slug>`.

Public H3 still: [`docs/demo/`](demo/).

## Lipdub

`ltx25_a2v` with audio at or under **6.5s** (`--lipdub-max-s`, `LIPDUB_SEGMENT_MAX_S`) is one pass. Longer audio is split, rendered, and stitched. The original wav is muxed back on. There is no hard 7-second attention mask. Locally the distilled latent stops tracking audio inside the window (tower motion died near 7.4s / 177 frames). The default slice stays at or under 169 frames (7.04s) after the overlap snap.

| Flag | Default | Meaning |
|---|---|---|
| `--silence-min-s` | 0.25 | Pauses at least this long are closed-mouth |
| `--silence-mode` | `idle` | `idle` renders a breathing piece. `hold` is a still plate and frozen tail. `bridge` is a 9-frame mouth close |
| `--anchor` | `previous` | Last-frame chain. `hybrid` and `source` are experimental. `pause-reset` is opt-in |
| `--reframe` | `off` | `on` scales each piece back to the source still. Experimental |
| `--lipdub-overlap` | 8 | Frames trimmed on speech-to-speech seams |
| `--max-piece-seconds` | 3.0 | After the clip is already segmented, split a long speech run at the quietest legal frame. Does not change the 6.5s one-pass threshold |
| `--pause-reset-strength` | 0.65 | `LTXVAddGuide` strength. 1.0 turned the head in about 0.2s |
| `--pause-reset-min-s` | 0.5 | Only pauses at least this long get the source keyframe |
| `--words` | none | `{w,s,e}` timestamps. Splits prefer pauses and never cut a word. Wins over HeartMuLa transcribe |
| `--heartmula-transcribe` | off | When `--words` is omitted, transcribe `--audio` into that same shape. Dry-run plans only |
| `--tripod` | off | Locked-off talking head. Stage-1 strength 0.7 → 0.85. Stage 2 stays 1.0 |

`--dry-run` prints the segment plan and validates each piece. Nothing is queued. Defaults recorded on `params.lipdub` when the flags are omitted: `silence_mode=idle`, `anchor=previous`, `reframe=false`. The pre-slim flag essay is archived at [`docs/archive/video_buddy/docs/LIPDUB.md`](archive/video_buddy/docs/LIPDUB.md).

## HeartMuLa

Optional local music via [heartlib](https://github.com/HeartMuLa/heartlib) (Apache-2.0). Buddy does not clone the Comfy node and does not download weights as part of install. Importing `master_agent` does not import heartlib.

1. **Generate** lyrics + tags → wav (`HeartMuLaGenPipeline`). HeartCodec sits inside that pipeline.
2. **Transcribe** audio → `{words:[{w,s,e}]}` for `--words`.

`--audio` wins. Lyrics and tags run only when `--audio` is omitted. `--dry-run` prints the plan, returns 0, and does not import heartlib or write a wav.

```bash
python -m master_agent heartmula generate \
  --lyrics "[Verse]\nlocal night" --tags piano,happy --duration 30 --out out/track.wav --dry-run
python -m master_agent heartmula transcribe --audio vocals.wav --out words.json --dry-run
```

Tags are lowercase and comma-separated without spaces. Default duration is 30s (`HEARTMULA_DURATION_S`). `--seed` is `torch.manual_seed` before the call. The pipeline `__call__` has no seed argument. Defaults: topk 50, temperature 1.0, cfg_scale 1.5. `HEARTMULA_DTYPE` defaults to bf16. Codec dtype stays fp32. Lazy-load is on when `VRAM_GB` ≤ 16.

Tower Comfy classes, measured 2026-09-30, and the only names to use: `HeartMuLa_Generate`, `HeartMuLa_Transcribe` (`benjiyaya/HeartMuLa_ComfyUI` @ `fdb53c4`). Do not invent others. Buddy's own commands use heartlib, not those nodes.

Files, the 401 codec id, and the sequential VRAM note: [Weights](WEIGHTS.md#heartmula-files).

Provenance adds optional `params.heartmula` (lyrics, tags, repo ids, version, dtypes, lazy load, seed, wav, transcribe source, words path, `max_audio_length_ms`, dry-run). Empty blocks are omitted.

## Music video

Two paths:

- `music` — spectral-flux beat grid, shots trimmed to beat windows, track muxed on top. `--visual fractal` is CPU-only. `run` auto-routes music intent, or audio longer than one segment, unless `--variant` is set.
- `mv` — one unique Comfy/LTX clip per beat window, then Remotion at 30 fps, 1080p, full track. Not Grok Imagine. Schema `buddy.mv.beat_plan/v1`. Duplicate SHA-256 hashes and 0/1-frame still-holds refuse the stitch.

```bash
python -m master_agent music "dreamy synthwave MV" --audio track.mp3
python -m master_agent mv plan --audio track.mp3 --out out/beat_plan.json
python -m master_agent mv render "I'm in love with a bot" --audio track.mp3 --out out/MV-FIXED.mp4
python -m master_agent mv render --audio track.mp3 --dry-run
```

`mv plan` requires `--audio`. `mv render --dry-run` without a file writes a silent wav so planning can run. That wav is not model audio. TeaCache injects when the class is registered and is not installed by this command.

## Fractal

CPU Mandelbrot/Julia in `master_agent/fractal/`. No Comfy. Targets: `seahorse`, `elephant`, `minibrot`, `spiral`. Palettes: `fire`, `ocean`, `monochrome`, `neon`, `sunset`.

```bash
python -m master_agent fractal "title" --duration 20 --target seahorse --palette fire
python -m master_agent fractal --mode inpaint --image plate.png
python -m master_agent fractal --audio track.mp3
```

Comfy Voronoi and Perlin plates are retired. This CPU path is the one that remains.

## Movie Builder, CCC, upscale

Movie Builder is shot-by-shot LTX 2.3 (Flux stills, per-shot video and audio, voice sample, assembler). The shot guide stays next to the graphs: [`video_buddy/workflows/260507_VIDEO-BUDDY_MOVIE-BUILDER_GUIDE.md`](../video_buddy/workflows/260507_VIDEO-BUDDY_MOVIE-BUILDER_GUIDE.md). It is a heavy graph. See the pack table in [Weights](WEIGHTS.md#pack-table).

CCC: `character create` writes a bible, a Flux sheet, and a captioned dataset. `--train` chains Flux LoRA via a separate ai-toolkit venv (`lora setup`). Rank 16, 1500 steps, resolutions 512 and 768 on a 16GB card. Expect hours. `download-flux` is the one-time Flux fp8 pull (~17GB) and is a different consent from the LTX ask.

`--upscale seedvr2` is the shipped post-stage (`workflows/upscale_seedvr2_api.json`). `--upscale rtx` needs a gitignored template and raises `FileNotFoundError` on a clean clone. The audit row for RTX stays unwired.

## Pack C Seedance

**Local-only is a hard requirement.** This route uses no cloud APIs, no cloud services, and no hosted inference.

Generate only on `http://127.0.0.1:8188` with the on-disk catalog: text or one-take and first-frame → `ltx25_t2v_i2v`, first + last → `ltx25_flf2v`, references → `ltx25_msr`. Wan and H3 stay available when you pass that catalog id. Optional Grok may rank a story. That client is not video inference.

`python -m master_agent run "Seedance 2.5 draft one-take"` fail-closes onto `ltx25_t2v_i2v`. An explicit local `--variant` is honored. A Partner stub id is rewritten to the local pack and is not loaded.

`master_agent/comfy/partner_pointers.py` records the field shape. `partnerGraphsExecutable` is false. Those stubs are not in `WORKFLOW_FILES`, `GET /api/variants`, or the studio picker. No `workflows/**/api_seedance*.json`. No comfy.org, BytePlus, ModelArk, or KIE client on this route.

| Buddy stub | Recorded template id | Recorded scout class |
|---|---|---|
| `seedance25_draft_t2v` | `api_seedance2_5_draft_t2v` | `ByteDance2TextToVideoNode` |
| `seedance25_draft_i2v` | `api_seedance2_5_draft_i2v` | `ByteDance2FirstLastFrameNode` |
| `seedance25_draft_r2v` | `api_seedance2_5_draft_r2v` | `ByteDance2ReferenceNodeV2` |

Recorded promote class: `ByteDance2DraftToFinalVideoNode`. Policy object: `PACK_C_LOCAL_ONLY` in `master_agent/config.py`.

## Ingested workflows

One-off Comfy API graphs are learned from the CLI into gitignored `state/ingested/<slug>/`. They are not catalog variants and are not promoted. Commands, the field map, and the dry-run default: [Workflow ingest](WORKFLOW_INGEST.md).
