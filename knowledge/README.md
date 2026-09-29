# Shared knowledge

Durable, human-readable learnings that sync via git between machines.

Ocala (Scott) and Albuquerque (Jason) each run a separate local ChromaDB
plus Ollama. That store lives at `video_buddy/state/chroma/` and is
gitignored, so a lesson learned on one machine never reaches the other.
This folder is the slice both machines can pull and commit.

It does not replace the local knowledge base. Chroma still holds workflow
digests and run records on that machine. Markdown here is what git carries.

## Rules

1. **Pull before a run.** `git pull` on this repo before you generate, so
   you are not retrying a failure the other machine already fixed.
2. **Commit after.** When a run teaches something worth keeping, add a
   short entry and commit it before you walk away.
3. **Keep entries short and structured.** One file per prompt, workflow
   note, judge pattern, failure, or decision. Follow the schema in
   [`AGENTS.md`](AGENTS.md) and the README in each subfolder.
4. **No secrets.** Never commit API keys, tokens, `.env` values, or
   Hugging Face credentials.
5. **No local absolute paths.** Strip them the way provenance sidecars do.
   The public H3 sidecar
   ([`docs/demo/H3-SHOWCASE-BMX-8s.provenance.json`](../docs/demo/H3-SHOWCASE-BMX-8s.provenance.json))
   keeps a filename (`h3_showcase_bmx_8s_00001_.mp4`) and a role-relative
   weight name (`gguf/minimax_h3_fl2va_pruned-Q4_K.gguf`). New clip
   sidecars use repo-relative outputs such as
   `outputs/<run_id>/shot-1.mp4`
   ([`video_buddy/docs/CLIP_PROVENANCE.md`](../video_buddy/docs/CLIP_PROVENANCE.md)).
   Write those. Drop drive letters and home directories
   (`C:\Users\...`, `/Users/...`, `/home/...`).

## Layout

| Folder | What goes here |
|---|---|
| [`prompts/`](prompts/) | Working prompt templates that produced good results. |
| [`workflows/`](workflows/) | Which Comfy graphs worked for which shot types. JSON lives under [`video_buddy/workflows/`](../video_buddy/workflows/). |
| [`judge-feedback/`](judge-feedback/) | Judge verdicts and what fixed them. |
| [`failures/`](failures/) | Failure cases and the fix. Check this folder before retrying a known-bad approach. |
| [`decisions/`](decisions/) | What was tried, what was chosen, why. |

Files under those folders that start with `EXAMPLE` are placeholders.
They are not recorded runs.

## TODO

**Next step, separate PR — do not do it in the scaffold PR.** Wire each
local ChromaDB to ingest `knowledge/` on startup.

Today `python -m master_agent kb ingest` loads workflow digests and run
records through `video_buddy/master_agent/kb/ingest.py`
(`ingest_workflows`, `ingest_all_runs`). It does not read this folder.
The follow-up should add that ingest (startup, alongside the existing
bulk load) without changing how workflows or run records are stored.
Leave runtime code alone until that PR.
