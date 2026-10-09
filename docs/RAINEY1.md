# Rainey1

The Rainey1 caliber pack: a director preset, a selectable judge rubric, and a seed batch with a top-K cut. Named for Jason Rainey (Rainey1), who shaped the style. The judge still does not album-lock. Human eyes beat it on taste. See [Judge](JUDGE.md).

Code: `master_agent/orchestrator/presets/rainey1.json` (recipes), `master_agent/judge/rubric.py` and `master_agent/judge/prompts/rainey1.md` (rubric), `master_agent/rainey1/` (batch).

## Recipes

`python -m master_agent workflows` lists them after the catalog. All run on LTX 2.3 `base`, and frames stay `8n+1`.

| Recipe | Plate |
|---|---|
| `lock_open` | Identity desk plate, 768×512, 9 frames |
| `breach` | Same character in a desert / craft breach, 640×384, 17 frames |
| `density` | Climax look-up, 512×384, 25 frames |
| `myth_16x9` | 16:9 music-video set-piece, 768×512, 25 frames |
| `story_9x16` | Vertical plate, 640×384, 17 frames, then an ffmpeg crop to 9:16 |

The `rainey1_` prefixed ids are accepted too. `story_9x16` renders the ladder plate and crops it, because portrait 512×768 is not a legal start size under `clamp_resolution`.

## Rubric

Any one of these loads `judge/prompts/rainey1.md` (vision calls load `rainey1_vision.md`):

- `PANEL_JUDGE_RUBRIC=rainey1`
- The brief, shot card, or prompt contains the keyword `rainey1`
- A recipe flag: `judge_rubric`, `rubric`, or `recipe` set to `rainey1`, or `"rainey1": true` on the shot, context, or `attach_recipe`

Unset, empty, or `PANEL_JUDGE_RUBRIC=default` keeps `prompts/judge.md`. An explicit `default` / `off` / `none` also ignores the keyword. `judge_rubric_debug()` returns `rubric`, `prompt_file`, `loaded`, and a short hash of the instructions.

The JSON stays compatible with the default judge (`pass`, `score`, `look_score`, `brief_adherence`, `issues`, `prompt_rewrite`, `param_hints`, `reason`). Rainey1 adds look dimensions, each 0.0–1.0:

| Field | What it scores |
| --- | --- |
| `identity_lock` | Same face, wardrobe, silhouette. No morph or twin. |
| `density_escalation` | Intentional flora / craft / cables / cast, not a sparse melt. |
| `emissive_lighting` | Several lights key the subject; deep blacks stay. |
| `anti_slop` | No glow-soup, rainbow noise, plastic skin, watermark, or UI text. |
| `pacing_hold` | Music-video hold and readable motion. |
| `brief_adherence` | Separate from look. |

`look_score` is the mean of the five look dimensions when the model returns them. `brief_adherence` is not part of that mean.

Gates:

- **Retry** when `look_score` < 0.55, or on a hard-fail.
- **Pass candidate** when `look_score` ≥ 0.75 and `hard_fails` is empty.
- **Human veto** (no further automatic retry) when `brief_adherence` < 0.5 and `look_score` ≥ 0.7. `album_lock` stays false.

| Hard-fail code | Rule |
| --- | --- |
| `identity_morph` | Face morph, twin faces, melted face, identity drift. "No morph" in a negative prompt is not a hit. |
| `filesize_junk` | Probe `tiny_file` (under 100KB). |
| `too_few_frames` | Probe under 3 frames. |

Short duration and other ffprobe noise are not hard-fails unless the frames themselves are junk. The default judge, with the rubric unset, does not apply these gates.

## Batch top-cut

`rainey1-batch` generates one seed at a time for one recipe, drops junk (`<100KB` or `<3` frames) before the judge, and copies the top-K keepers into `--out` with a `buddy.clip.provenance/v1` sidecar (`.buddy.json`).

```bash
python -m master_agent rainey1-batch --recipe lock_open --seeds 42,43,44,45 --top-k 2 \
  --out outputs/rainey1/topcut/lock_open --dry-run
```

`--dry-run` resolves the recipe and prints the table. It queues zero Comfy jobs and writes no clips. Drop the flag for a live run on a GPU machine with Comfy on `:8188`. Look below `0.55` drops. Low `brief_adherence` with a high look is a human veto, not an auto-keep. A live run that keeps nothing exits non-zero unless `--allow-empty`. OOM walks `DOWNSCALE_LADDER` downward one retry at a time. `--no-llm-judge` uses the heuristic look score only.

Keepers: `seed-<n>.mp4` plus `seed-<n>.buddy.json`, with a `rainey1` block (recipe id, seed, look score, brief adherence, fail reasons). A human top-cut still decides what goes into a dataset. LoRA training on keepers is not part of this command and has no shipped config.

Tests (no GPU): `python -m pytest tests/test_rainey1_batch.py tests/test_rainey1_preset.py tests/test_rainey1_rubric.py -q` from `video_buddy/`.
