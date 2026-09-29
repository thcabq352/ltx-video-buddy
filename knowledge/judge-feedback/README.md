# Judge feedback

Judge verdicts and the change that fixed them.

One file per notable run, or per recurring failure pattern once the
same verdict shows up more than once. Schema:
[`../AGENTS.md`](../AGENTS.md).

Quality-bar codes the live judge already uses (see
[`video_buddy/docs/SELF_IMPROVEMENT_LOOP.md`](../../video_buddy/docs/SELF_IMPROVEMENT_LOOP.md)):

| id | code | Meaning |
|---|---|---|
| `a` | `missing_music_bed` | Music-video brief with no bed |
| `c` | `thin_still_i2v` | Still→I2V with no shot plan |
| `d` | `unused_control_pack` | Control pack present, unused |

Rule `b` (`face_similarity`) is not scored. Do not invent a face score.

Verdict labels match the loop: `pass`, `fail`, `human_veto`,
`exhausted`.

## EXAMPLE entry — not a recorded verdict

```markdown
# EXAMPLE — thin still→I2V on a product hero

- **status:** example
- **verdict:** fail
- **fail reasons:** `c` / `thin_still_i2v` — still→I2V with no shot plan
- **what fixed it:** Placeholder. A real note names the revise (camera
  move, action, continuity) that the next judge pass accepted.
- **applies when:** `--image` or an i2v variant and the brief has no
  camera / action / continuity line.
```

Write `status: recorded` only after a real judge pass on that machine.
