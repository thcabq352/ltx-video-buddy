# Failures

Failure cases and the fix that resolved them.

**This is the highest-value folder.** Both machines should read it
before retrying a known-bad approach. A repeated bad queue wastes a
shift budget on a lesson the other site already paid for.

One file per failure. Prefer a new file over rewriting history.
Schema: [`../AGENTS.md`](../AGENTS.md). Every real entry sets
`priority: check-before-retry`.

If a folder of daily files gets hard to scan, add a weekly summary
(`YYYY-Www-<slug>.md`) that lists the open "do not retry" lines and
links the dailies. Keep the daily files.

## EXAMPLE entry — not a recorded failure

Field lessons already live in [`video_buddy/AGENTS.md`](../../video_buddy/AGENTS.md)
(for example LTX length 8 collapsing to one frame). Do not copy them
here unless a machine re-hit them and you are logging that incident.
The block below only shows the shape.

```markdown
# EXAMPLE — length 8 queued on an LTX graph

- **status:** example
- **priority:** check-before-retry
- **symptom:** Placeholder. A real note says what the file did
  (frame count, size, tracker state).
- **bad approach:** Queuing `length=8` on LTX. Valid lengths are
  `8n+1`, minimum 9.
- **fix:** Placeholder. A real note says the snap or patch that
  produced a clip with at least 9 frames.
- **applies when:** LTX 2.3 / 2.5 graphs (`base`, `eros`, `ltx25_*`).
  Not the H3 17k+5 grid.
```
