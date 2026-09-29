# Decisions

Short decision log: what was tried, what was chosen, why.

One file per choice. Schema: [`../AGENTS.md`](../AGENTS.md).
Keep it shorter than a design doc. Link the workflow JSON or the
failure note instead of restating them.

Supersede by appending a new file that points at the old one. Leave
the old file so the other machine can see what changed.

## EXAMPLE entry — not a recorded decision

```markdown
# EXAMPLE — hero product shot stays on H3 t2v

- **status:** example
- **date:** 2026-09-29
- **tried:** `h3_t2v` versus `ltx25_t2v_i2v` for a logo-free product pour
- **chosen:** `h3_t2v` — placeholder, not a comparison that was run
- **why:** A real note says the concrete reason (audio, length, grade,
  or a judge score) in one to three lines.
- **revisit when:** the shot needs a still as the first frame (`h3_i2v`)
  or tight lip-sync (`ltx25_a2v`)
- **graph:** video_buddy/workflows/minimax-h3/MiniMax-H3_T2V_FL2VA_api.json
```
