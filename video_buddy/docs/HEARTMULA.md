# HeartMuLa (local lyrics → track, and word timestamps)

HeartMuLa is the open music model from [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) (Apache-2.0). Buddy drives it through that Python package. The Comfy custom node is a tower-side install; this repo does not clone it and does not download weights.

Two jobs:

1. **Generate** a wav from lyrics + tags (`HeartMuLaGenPipeline`). HeartCodec sits inside that pipeline. There is no separate codec command.
2. **Transcribe** audio to word timestamps (`HeartTranscriptorPipeline`) in the same `{"words":[{w,s,e}]}` shape that `run --words` already accepts.

A file passed as `--audio` stays the track. Generated audio is optional and only runs when `--audio` is omitted.

## CLI

From `video_buddy/` (or `PYTHONPATH=video_buddy`):

```bash
python -m master_agent heartmula generate \
  --lyrics "[Verse]\nlocal night, local light" \
  --tags piano,happy \
  --duration 30 \
  --seed 7 \
  --out out/track.wav \
  --dry-run

python -m master_agent heartmula transcribe \
  --audio vocals.wav \
  --out words.json \
  --dry-run
```

`--dry-run` prints the plan and returns 0. It does not import `heartlib`, does not use a GPU, and does not write the wav or the words file.

Live generate calls `HeartMuLaGenPipeline.from_pretrained`, shrinks `backbone.max_seq_len` before `setup_caches` (see the KV window below), then:

```text
pipe({"lyrics": ..., "tags": ...}, max_audio_length_ms=..., save_path=...,
     topk=50, temperature=1.0, cfg_scale=1.5)
```

`__call__` has no seed argument and no seq-len argument. `--seed` is `torch.manual_seed` (and CUDA when present) before that call. The window is applied on the loaded backbone, not passed into `pipe`. Reconstruction (`run_music_reconstruction.py`) is not this path.

Tags are lowercase and comma-separated **without spaces** (`piano,happy,wedding`). Lyrics may use `[Intro]` / `[Verse]` / `[Chorus]`. Default duration is **30s** (`HEARTMULA_DURATION_S`). The heartlib example uses 240s; pass `--duration 240` when you want that. Four minutes plus an LTX music video does not fit a 16GB card.

Live transcribe uses the official whisper kwargs plus `return_timestamps="word"` (the class subclasses transformers `AutomaticSpeechRecognitionPipeline`; the official example does not request timestamps). Chunks `{text, timestamp:(s,e)}` become `{w,s,e}`. HeartTranscriptor is trained on separated vocals. Demucs (or similar) is recommended first. Buddy does not run separation.

## Music video and `music`

`--audio` wins. If you pass both a file and lyrics, the file is the track and HeartMuLa is not called.

```bash
python -m master_agent mv render "I'm in love with a bot" \
  --heartmula-lyrics "[Chorus]\nhold the local line" \
  --heartmula-tags piano,happy \
  --heartmula-duration 30 \
  --out out/MV-FIXED.mp4

python -m master_agent music "dreamy synthwave" \
  --heartmula-lyrics "[Verse]\nchrome grids" \
  --heartmula-tags synth,night
```

`mv render --dry-run` without `--audio` writes a **silent** wav under the work dir so beat planning can run. That file is not model audio (`placeholder: silent-wav`). `heartmula generate --dry-run` does not write a wav.

`mv plan` still requires `--audio`.

See [MUSIC_VIDEO.md](MUSIC_VIDEO.md).

## Lipdub words

`--words` stays the faster-whisper (or any) JSON file. When it is set, `--heartmula-transcribe` is skipped.

When `--words` is omitted and `--heartmula-transcribe` (or `HEARTMULA_TRANSCRIBE=1`) is set, Buddy runs HeartTranscriptor and feeds the same words path. Dry-run and `--self-improve-dry` only print the plan. HeartTranscriptor is an alternative, not a replacement.

See [LIPDUB.md](LIPDUB.md).

## VRAM (16GB, sequential with LTX)

`workflows --vram` row `heartmula`: expected about **12GB**, class **tight**, range about **8–16GB** depending on dtype. HeartMuLa 3B and LTX cannot share 16GB comfortably. Run them one after the other.

heartlib defaults Buddy uses:

| Knob | Default | Notes |
|---|---|---|
| `HEARTMULA_DTYPE` | `bf16` | Mula. Also `fp16` / `fp32`. |
| `HEARTMULA_CODEC_DTYPE` | `fp32` | Official codec dtype. bf16 codec hurts quality. |
| `HEARTMULA_LAZY_LOAD` | on when `VRAM_GB` ≤ 16 | Unload HeartMuLa before HeartCodec. |
| `HEARTMULA_DEVICE` | `cuda` | |
| `HEARTMULA_CODEC_DEVICE` | same as device | A second GPU turns lazy load off only if you set it and set `HEARTMULA_LAZY_LOAD=0`. |

heartlib has **no nf4 / fp4**. Setting `HEARTMULA_DTYPE=nf4` is an error. The Comfy node's 4-bit path is tower-owner only and is not this Python pipeline.

`HEARTMULA_VERSION` must stay `3B`. heartlib opens the folder `HeartMuLa-oss-3B`. 7B is not released.

### KV window

heartlib builds the 3B backbone with `max_seq_len=8192`. torchtune 0.4 (the version heartlib pins) allocates the KV cache at the **query** head count, after the GQA expand (24 heads, not the 8 KV heads). At the default cfg batch of 2 that cache is about 5.3GB. On a 16GB card, next to bf16 weights, `setup_caches` OOMs. A 512-token window passed a 5 second tower smoke (bf16, lazy load, wav out).

Buddy does not leave every run at 8192, and it does not pin every card to 512.

- Audio frames are `duration_ms // 80` (heartlib).
- Plus a lyric/tag token estimate. The tokenizer is not loaded, so a dry-run stays offline. The estimate is one token per character, which is generous for English and close for CJK.
- Rounded up to a multiple of 128.
- Never above 8192. RoPE was built at 8192; Buddy will not grow it.

The card caps that window. About 14GB is reserved for bf16 weights and activations, and the rest is the GQA-expanded cache. A 16GB card then holds about **3072** tokens at cfg 1.5 (batch 2). A 30 second song still fits. A 240 second song does not: the plan fails before heartlib is imported and names the cap. Once the card is about 20GB the same reserve fits the full 8192 window, so a long song on a larger GPU is unchanged. `cfg_scale=1` uses batch 1 and a higher cap.

| Knob | Role |
|---|---|
| `HEARTMULA_MAX_SEQ_LEN` / `--max-seq-len` | Exact window. Skips the cap. `512` matches the 5s smoke. A value shorter than the song can still run out of cache. |
| `HEARTMULA_LOW_VRAM=1` / `--low-vram` | Plan as if the card is 16GB, even when detection says more. |

5 second proof when the weights are already under `models/heartmula` (this does not download them). `--max-seq-len 512` is the window that passed on the tower. Drop it and Buddy sizes a 5 second clip smaller than that. Codec dtype stays **fp32** unless you set it; the tower smoke used bf16, which is optional and lowers quality.

```bash
HEARTMULA_DEVICE=cuda HEARTMULA_CODEC_DEVICE=cuda \
HEARTMULA_LAZY_LOAD=1 \
HEARTMULA_DTYPE=bf16 HEARTMULA_CODEC_DTYPE=bf16 \
python -m master_agent heartmula generate \
  --lyrics "[Verse]\nsmoke line" \
  --tags piano,happy \
  --duration 5 \
  --max-seq-len 512 \
  --low-vram \
  --out heartmula_smoke_test.wav
```

`--dry-run` prints `max_seq_len` and does not import heartlib.

### Saving the wav

heartlib calls `torchaudio.save`. TorchAudio 2.9+ saves through torchcodec. Comfy's embedded Python often has a missing or broken torchcodec DLL, and that save raises after the song is already decoded.

Buddy wraps `torchaudio.save` for the generate call. If it raises, the same channels-first waveform is written with `soundfile` (time-major, the sample rate heartlib passed, 48 kHz). The original `torchaudio.save` is restored afterward. Nothing in the Comfy environment is installed or changed. `soundfile` is a heartlib dependency. If it is also missing, the error names both failures.

## Weights (consent, attested names only)

`doctor` reports `heartlib`, `heartmula-weights`, and `heartmula-comfy`. Those rows print NEED and a fix line. They do **not** fail doctor by themselves, and they do not fetch.

```bash
python -m master_agent download-models --heartmula
# lists confirmed-missing; prints "Nothing downloaded."
python -m master_agent download-models --heartmula --yes   # only after you agree
```

`--scan-only` and `--use-existing` never fetch. Zero-byte files count as missing. `--heartmula` returns before the LTX scan, so it does not also pull LTX.

**Check free space before `--yes`.** Tower disk check 2026-09-30: **F: had about 28GB free**. The default pull is multi-GB (Hub `usedStorage`, 2026-09-30):

| Repo (attested) | On-disk folder (heartlib) | Size |
|---|---|---|
| `HeartMuLa/HeartMuLaGen` | `tokenizer.json`, `gen_config.json` at the root | small |
| `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` (default) | `HeartMuLa-oss-3B/` | ~15.8GB |
| `HeartMuLa/HeartCodec-oss-20260123` (default) | `HeartCodec-oss/` | ~6.6GB |
| `HeartMuLa/HeartTranscriptor-oss` | `HeartTranscriptor-oss/` | ~3.1GB |

Default generate + transcribe is about **25GB**. That does not fit comfortably next to other packs when F: is near 28GB free.

`HeartMuLa/HeartCodec-oss` (the unsuffixed base id) returned **401** on the tower and in this pack's Hub lookup. Do not use it. Prefer **`HeartMuLa/HeartCodec-oss-20260123`**, downloaded into the folder heartlib opens (`HeartCodec-oss`). If you switch the checkpoint to `HeartMuLa/HeartMuLa-RL-oss-3B-20260123`, pair it with that same codec. `HEARTMULA_CODEC_REPO=HeartMuLa/HeartCodec-oss` is refused: this pack will not guess that repo's filenames.

Other attested mula repos (same four-shard names: `config.json`, `model.safetensors.index.json`, `model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors`):

- `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` — default, placed in `HeartMuLa-oss-3B/`
- `HeartMuLa/HeartMuLa-oss-3B`
- `HeartMuLa/HeartMuLa-RL-oss-3B-20260123`

Codec shards (20260123 only): `config.json`, `model.safetensors.index.json`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`.

heartlib layout:

```text
{HEARTMULA_MODELS_DIR}/          # default: MODELS_DIR/heartmula
  tokenizer.json
  gen_config.json
  HeartMuLa-oss-3B/
  HeartCodec-oss/                # contents of HeartCodec-oss-20260123
  HeartTranscriptor-oss/
```

The Comfy node README uses different folders (`./HeartMuLa/HeartMuLa-oss-3B`, `./HeartMuLa/HeartCodec-oss-20260123` kept under that name, RL in its own folder). Buddy's download path fills the **heartlib** layout only. It does not fill the Comfy tree.

Install heartlib yourself when you want a live run: clone the repo and `pip install -e .`. Importing `master_agent` does not import heartlib.

## Tower Comfy node (measured 2026-09-30, not installed by this repo)

Comfy Desk on the tower already has the pack. This checkout does not clone it, does not pip-install it, and does not download its weights.

| Fact | Value |
|---|---|
| Clone | `benjiyaya/HeartMuLa_ComfyUI` @ `fdb53c4` |
| Location | tower Comfy portable `custom_nodes` |
| Load | no import errors; node pack loaded in about 0s after restart |
| Pip | Comfy embedded Python; `torchao` pinned to **0.9.0** for torchtune |
| Weights | **not** downloaded (empty model dirs only) |

`/object_info` classes (do not invent others):

| `class_type` | Display name |
|---|---|
| `HeartMuLa_Generate` | HeartMuLa Music Generator |
| `HeartMuLa_Transcribe` | HeartMuLa Lyrics Transcriber |

The cached `video_buddy/state/object_info.json` in this repo predates that install and does not list those classes. `doctor`'s `heartmula-comfy` row says so and points at `fetch-object-info` on the tower. It does not fetch.

These two classes are **payload nodes**, not optional accelerators. They are not in the TeaCache / bypass set. The capability catalog lists them with `surfaces=()`: Buddy's generate and transcribe commands use heartlib, not a shipped API template, and attach does not patch their widgets (widget names were not taken from `/object_info`).

An attach recipe that names any other class containing `HeartMuLa` raises `AttachError`. The two names above are allowed if a recipe lists them; they are not wired into LTX graphs.

See [COMFY_ATTACH.md](COMFY_ATTACH.md).

## Provenance

Same schema id: `buddy.clip.provenance/v1`. Optional `params.heartmula` when lyrics, tags, a wav, a transcribe source, or a words path is present. Empty blocks are omitted so older sidecars stay clean. Fields: lyrics, tags, model repo ids, version, dtypes, lazy load, seed, wav path, transcribe source, words path, `max_audio_length_ms`, `max_seq_len`, dry-run. See [CLIP_PROVENANCE.md](CLIP_PROVENANCE.md).

## Env

| Variable | Role |
|---|---|
| `HEARTMULA_MODELS_DIR` | Checkpoint root (default `MODELS_DIR/heartmula`). Must stay inside `MODELS_DIR`; pack pulls will not write an attached Comfy tree. |
| `HEARTMULA_MULA_REPO` | One of the three attested 3B ids |
| `HEARTMULA_CODEC_REPO` | `HeartMuLa/HeartCodec-oss-20260123` only |
| `HEARTMULA_TRANSCRIPTOR_REPO` | `HeartMuLa/HeartTranscriptor-oss` |
| `HEARTMULA_GEN_REPO` | `HeartMuLa/HeartMuLaGen` |
| `HEARTMULA_VERSION` | `3B` |
| `HEARTMULA_DURATION_S` | Default 30 |
| `HEARTMULA_LYRICS` / `HEARTMULA_TAGS` | Used when the CLI flags are omitted |
| `HEARTMULA_TRANSCRIBE` | `1` acts like `--heartmula-transcribe` |
| `HEARTMULA_DTYPE` / `HEARTMULA_CODEC_DTYPE` | bf16 / fp16 / fp32 |
| `HEARTMULA_DEVICE` / `HEARTMULA_CODEC_DEVICE` | torch device strings |
| `HEARTMULA_LAZY_LOAD` | `1` / `0` |
| `HEARTMULA_MAX_SEQ_LEN` | Exact backbone KV window (1–8192). Unset: sized to the clip and the card |
| `HEARTMULA_LOW_VRAM` | `1` plans that window as if the card is 16GB |
