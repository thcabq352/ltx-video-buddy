<!--
Title: one Conventional Commit line written as a user-facing outcome, e.g.
  feat(ingest): queue a dropped ComfyUI graph through the same frame law
The title becomes the squash-merge commit on main. See docs/RELEASING.md.
-->

## Outcome

What an operator can do, or what now works correctly, after this merges. One or two sentences.

## Scope

- Subsystems touched:
- Not included / follow-ups:
- Local-only: no new cloud calls, downloads, or hosted nodes (or explain).

## Tests

- [ ] Full suite from `video_buddy/`: `PYTHONPATH=. python -m pytest tests -q` (paste the pass count)
- [ ] New or changed behavior has a test, or the reason it can't is stated
- [ ] GPU / live ComfyUI check, if relevant (hardware and result)

## Docs and changelog

- [ ] Docs updated in the one place each fact lives (see `docs/INDEX.md`)
- [ ] One theme line under `## [Unreleased]` in `CHANGELOG.md`, or "none needed" (refactor/test only)

Do not list individual fixes here or in the changelog; summarize them in the outcome.
