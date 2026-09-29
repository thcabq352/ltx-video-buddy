# Rainey1 batch top-cut

Phase 0.4 loop: generate N seeds for one Rainey1 recipe, drop junk, judge, copy the top-K into `--out` with a `buddy.clip.provenance/v1` sidecar (`.buddy.json`).

The director preset (`orchestrator/presets/rainey1.json`) and the judge rubric (`judge/prompts/rainey1.md`, `judge/rubric.py`) already ship on main. This command only adds the seed batch and the top-cut. It loads recipe geometry from the preset pack and asks the judge for rubric `rainey1`.

GPU renders belong on the operator tower **after this draft is merged and approved**. CI and the agent VM run `--dry-run` only. `--dry-run` queues zero Comfy jobs and writes no clips.

## Dry-run (no GPU)

From `video_buddy/`:

```bash
python -m master_agent rainey1-batch \
  --recipe lock_open \
  --seeds 42,43,44,45 \
  --top-k 2 \
  --out outputs/rainey1/topcut/lock_open \
  --dry-run \
  --no-interview
```

Recipes: `lock_open`, `breach`, `density`, `myth_16x9`, `story_9x16` (or the `rainey1_` prefixed ids). Frames stay `8n+1`. Junk (`<100KB` or `<3` frames) is dropped before the judge. Look below `0.55` drops. Low `brief_adherence` with a high look is a human veto, not an auto-keep. Exit status is non-zero when a real run keeps nothing, unless `--allow-empty`.

`story_9x16` follows the preset: generate the ladder plate, then ffmpeg-crop to 9:16 (portrait 512×768 is not a legal start size under `clamp_resolution`). OOM on a live run walks `DOWNSCALE_LADDER` downward, one retry at a time, and writes the rung's frame count.

## Phase 1 tower commands

Run these on the tower (RTX 5060 Ti 16GB), from `video_buddy/` with the venv active and Comfy on `:8188` (`--lowvram` or `--normalvram`). That GPU batch waits until this draft is merged and approved. LoRA train is Phase 2.

```bat
python -m master_agent health
python -m master_agent run "rainey1 lock open: photoreal locked character at desk, open PC tower, keyboard, high-key window, identity stable" --variant base --no-interview --dry-run
python -m master_agent run "rainey1 lock open: photoreal locked character at desk, open PC tower, keyboard, high-key window, identity stable, emissive monitor glow" --variant base --no-interview

python -m master_agent rainey1-batch --recipe lock_open --seeds 42,43,44,45,46,47,48,49 --top-k 3 --out outputs/rainey1/phase1/lock_open --no-interview
python -m master_agent rainey1-batch --recipe breach --seeds 42,43,44,45,46,47 --top-k 3 --out outputs/rainey1/phase1/breach --no-interview
python -m master_agent rainey1-batch --recipe density --seeds 42,43,44,45,46,47 --top-k 3 --out outputs/rainey1/phase1/density --no-interview
```

The first `run` is a dry plan. The second is the 9-frame proof. The three `rainey1-batch` lines are the captioned practice-clip batch. Human top-cut still decides what is copied into the dataset.

Keepers: `outputs/.../seed-<n>.mp4` and `seed-<n>.buddy.json` (`schema` `buddy.clip.provenance/v1`, plus a `rainey1` block with recipe id, seed, look score, brief adherence, and fail reasons).

## Phase 2

Phase 2 is a later train job: `training/configs/train_lora_ltx23_rainey1_caliber_16gb.yaml` forked from `train_lora_ltx23_16gb.yaml`, dataset under `training/datasets/styles/rainey1_caliber/`. It waits until Phase 1 has at least 40 captioned stills and 12 short clips. Lot Desk launches that train. This CLI only writes keeper clips under `outputs/rainey1/`.

Tests (no GPU): `python -m pytest tests/test_rainey1_batch.py tests/test_rainey1_preset.py tests/test_rainey1_rubric.py tests/test_ltx_frames.py tests/test_clip_provenance.py -q`
