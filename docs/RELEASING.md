# Releasing

How commits, merges, and release notes are written. History already on `main` stays as it is; this applies going forward. Changes land in [`CHANGELOG.md`](../CHANGELOG.md).

## Commit messages

One subject line in [Conventional Commits](https://www.conventionalcommits.org/) form:

```
<type>(<optional scope>): <what changes for the user, imperative, no period>
```

| Type | Use for |
|------|---------|
| `feat` | New capability an operator can use |
| `fix` | Wrong behavior corrected |
| `perf` | Faster or lighter, same behavior |
| `refactor` | Internal change, no behavior change |
| `docs` | Docs, templates, changelog |
| `test` | Tests only |
| `build` / `ci` | Dependencies, install, CI |
| `chore` | Anything else that ships nothing |

Rules:

- Keep the subject at or under 72 characters. Put detail in the body, not the subject.
- Scope is a subsystem: `ingest`, `comfy`, `llm`, `judge`, `music`, `ui`, `models`, `workflows`, `rainey1`, `docs`.
- Breaking changes add `!` after the type (`feat(comfy)!: ...`) and a `BREAKING CHANGE:` line in the body.
- No author names, machine names, PR numbers, or "update" / "wip" / "fix typo" subjects on `main`.

## Merging

Squash-merge every pull request. The squash title is the line people will read in history, so write it as a release-note outcome:

- Good: `feat(ingest): queue a dropped ComfyUI graph through the same frame law and VAE guard`
- Avoid: `Fix review comments`, `Rainey1 update`, `PR #NN follow-ups`

Small fixes that belong to one theme are squashed into one commit with one theme line. A pull request that only fixes its own earlier commits never gets a separate changelog entry.

## Changelog

- Add user-facing changes under `## [Unreleased]` in [`CHANGELOG.md`](../CHANGELOG.md), in the theme section they belong to (Workflow Ingest, Managed ComfyUI, Local LLM, Music, Rendering Quality, Models and Weights, Reliability, Documentation, and so on).
- One line per outcome. Fold related fixes into the line for the feature they repair.
- Describe what shipped and is tested. Do not write "production ready", and do not list a feature that is still on an open branch.
- Internal refactors and test-only changes do not need an entry.

## Cutting a release

1. Run the full suite from `video_buddy/`:

   ```bash
   PYTHONPATH=. python -m pytest tests -q
   ```

   CI ([`.github/workflows/pytest.yml`](../.github/workflows/pytest.yml)) runs a subset; run everything before a release.
2. Move the `Unreleased` entries into a new `## [x.y.z] - YYYY-MM-DD` section and add its link at the bottom of the file.
3. Pick the version: minor (`0.x.0`) for new capability, patch (`0.x.y`) for fixes only. Before 1.0, a breaking change also bumps the minor version.
4. Tag the merge commit `vX.Y.Z` and use the template below for the release notes.

## Release notes template

```markdown
## Video Buddy x.y.z

One or two sentences on what this release lets an operator do.

### Highlights
- <theme>: <outcome>

### Upgrade notes
- <new env var, changed default, or a step to run; "None" if nothing>

### Known limits
- <what is not covered yet, for example "GPU renders verified on operator hardware only">

Full list: CHANGELOG.md
```
