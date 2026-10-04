# Rainey1 judge rubric

Selectable caliber check for the panel judge. Preset recipes, the batch CLI, and LoRA training are out of scope here. The judge still does **not** album-lock: Admiral / human eyes beat it on taste.

## Select it

Any one of these loads `master_agent/judge/prompts/rainey1.md` (vision calls load `rainey1_vision.md`):

- Env `PANEL_JUDGE_RUBRIC=rainey1`
- The brief, shot card, or prompt contains the keyword `rainey1`
- A recipe flag: `judge_rubric`, `rubric`, or `recipe` set to `rainey1`, or `"rainey1": true` on the shot, context, or `attach_recipe`

Unset, empty, or `PANEL_JUDGE_RUBRIC=default` keeps `prompts/judge.md`. An explicit `default` / `off` / `none` also ignores the keyword. Debug dump: `judge_rubric_debug()` returns `rubric`, `prompt_file`, `loaded`, and a short hash of the instructions.

## Scores

JSON stays compatible with the default judge (`pass`, `score`, `look_score`, `brief_adherence`, `issues`, `prompt_rewrite`, `param_hints`, `reason`). Rainey1 adds look dimensions, each 0.0–1.0:

| Field | What it scores |
| --- | --- |
| `identity_lock` | Same face, wardrobe, silhouette. No morph or twin. |
| `density_escalation` | Intentional flora / craft / cables / cast, not a sparse melt. |
| `emissive_lighting` | Several lights key the subject; deep blacks stay. |
| `anti_slop` | No glow-soup, rainbow noise, plastic skin, watermark, or UI text. |
| `pacing_hold` | Music-video hold and readable motion. |
| `brief_adherence` | Separate from look. |

`look_score` is the mean of the five look dimensions when the model returns them. `brief_adherence` is not part of that mean.

## Gates

- **Retry** when `look_score` < 0.55, or on a hard-fail.
- **Pass candidate** when `look_score` ≥ 0.75 and `hard_fails` is empty.
- **Human veto** (no further automatic retry) when `brief_adherence` < 0.5 and `look_score` ≥ 0.7. `album_lock` stays false.

Hard-fail codes, also returned on the judge payload when this rubric is active:

| Code | Rule |
| --- | --- |
| `identity_morph` | Face morph, twin faces, melted face, identity drift. "No morph" in a negative prompt is not a hit. |
| `filesize_junk` | Probe `tiny_file` (under 100KB). |
| `too_few_frames` | Probe under 3 frames. |

Short duration and other ffprobe noise are not hard-fails unless the frames themselves are junk. Junk size and under-3-frames still retry under this rubric even when look was stripped of health penalties. The default judge, with the rubric unset, does not apply these gates.
