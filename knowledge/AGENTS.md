# Agents reading `knowledge/`

This folder is the git-synced memory between Ocala and Albuquerque.
Recorded markdown is copied into each machine's local Chroma collection
`knowledge` on Buddy startup and by `python -m master_agent kb ingest`.
Schema docs and `EXAMPLE` files are not indexed. Read these files
yourself before you write; git stays the source of truth.

## Before you write

1. **Reread the current tree.** Pull, then read the README in the
   target folder and any entries that match the shot, workflow, or
   failure you are about to record. Do not append a duplicate of a
   lesson that is already here.
2. **Check [`failures/`](failures/) before you retry** a known-bad
   approach. That folder is the highest-value one.
3. **Prefer appending a new file** over editing an old one. Old entries
   are a log. Correct a factual mistake in place only when the old text
   is wrong (a bad path, a leaked secret, a typo that changes the fix).
   Add a dated note at the bottom if you must supersede an entry, and
   leave the original paragraphs intact.
4. **Use the schemas below.** Copy the field list. Fill real values
   only from a run you actually observed. Otherwise mark the file
   `EXAMPLE` and say it is not a recorded learning.
5. **Roll daily notes into weekly summaries** as a folder grows. Keep
   the daily file. Add `YYYY-Www-<slug>.md` that points at the dailies
   and states the pattern in a few lines. Do not delete the dailies.

## Naming

| Kind | Filename |
|---|---|
| Single run or day | `YYYY-MM-DD-<slug>.md` |
| Weekly rollup | `YYYY-Www-<slug>.md` |
| Placeholder | `EXAMPLE-<slug>.md` |

Slug is lowercase, hyphenated, and specific (`h3-cold-brew-helmet-morph`,
not `notes`).

## Path and secret hygiene

Same rule as provenance sidecars
([`docs/PROVENANCE.md`](../docs/PROVENANCE.md),
[`docs/demo/H3-SHOWCASE-BMX-8s.provenance.json`](../docs/demo/H3-SHOWCASE-BMX-8s.provenance.json)):

- Repo-relative paths and bare filenames only.
- Comfy role-relative weight names (`gguf/…`, `loras/…`).
- No API keys, tokens, or `.env` values.
- No absolute paths.

## Schemas

Every real entry starts with `status: recorded` and a machine
(`ocala`, `albuquerque`, or `both`). Examples use `status: example`.

### `prompts/`

```markdown
# <slug>

- **status:** recorded | example
- **goal:** one line
- **model:** family + buddy id (e.g. MiniMax H3, `h3_t2v`)
- **key parameters:** steps, cfg, size, frames or duration, seed if it mattered
- **workflow:** repo-relative JSON path
- **prompt:** the text that was queued
- **result note:** one line — what was good enough to reuse
```

### `workflows/`

```markdown
# <shot type> — <buddy id>

- **status:** recorded | example
- **shot type:** t2v, i2v, flf, dialogue, inpaint, …
- **buddy id:** catalog id
- **graph:** repo-relative path under `video_buddy/workflows/`
- **why it worked:** one to three lines
- **watch-outs:** frame law, CFG, duration cap, or "none"
```

### `judge-feedback/`

```markdown
# <pattern or run>

- **status:** recorded | example
- **verdict:** pass | fail | human_veto | exhausted
- **fail reasons:** quality-bar id/code, or judge wording
- **what fixed it:** the revise that moved the verdict
- **applies when:** shot type or brief shape
```

### `failures/`

```markdown
# <symptom>

- **status:** recorded | example
- **priority:** check-before-retry
- **symptom:** what the clip or queue actually did
- **bad approach:** what not to do again
- **fix:** what resolved it
- **applies when:** variant, frame count, or brief shape
```

### `decisions/`

```markdown
# <choice>

- **status:** recorded | example
- **date:** YYYY-MM-DD
- **tried:** options actually compared
- **chosen:** the one kept
- **why:** one to three lines
- **revisit when:** condition that would reopen it, or "not pending"
```
