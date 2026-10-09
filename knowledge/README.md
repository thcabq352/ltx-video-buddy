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
   ([`docs/PROVENANCE.md`](../docs/PROVENANCE.md)).
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

## Ingest into local Chroma

Recorded entries are embedded into each machine's local Chroma collection
`knowledge` (same store as workflows and runs: `video_buddy/state/chroma/`,
same Ollama / llama.cpp embedding path). Workflow digests and run records
are unchanged.

The pass runs inside `python -m master_agent kb ingest` (agent tool
`kb_ingest`) and before storyboard / power-mode recall, so a new process
picks up a pull without a separate command. After you edit this folder, the next recall
picks it up, or run `kb ingest` again.

**What is indexed.** `knowledge/**/*.md` with a real learning. One Chroma
document per file, split into ~4000-character chunks only when an entry is
longer than that. Document id is `knowledge:<relative-path>:<content-hash>`
(plus `:chunk` when split). A second run upserts that id. A changed file
deletes the previous id. A removed file is dropped. Re-running does not
pile up copies.

**What is skipped.**

| Skip | Why |
|---|---|
| `README.md`, `AGENTS.md` | Schema and operator docs, not learnings. |
| Filename starts with `EXAMPLE` | Placeholder seeds. |
| First `status:` line is `example` | Same rule when the name was not prefixed. |
| Secret-like tokens (`api_key=`, `sk-…`, `hf_…`, `ltxv_…`, bearer tokens) | Refused whole. Nothing from that file is embedded. |

Absolute local paths (`C:\…`, `/Users/…`, `/home/…`) are replaced with
`[local-path]` before embed. Repo-relative paths stay.

`failures/` is stored with metadata `priority=high` and recalled ahead of
other shared notes. Other metadata: `source=knowledge`,
`category=prompts|workflows|judge-feedback|failures|decisions`, `path`.

The scaffold ships only READMEs and one EXAMPLE prompt, so a fresh
`kb stats` shows `knowledge=0` until somebody commits a `status: recorded`
entry. That is the skip policy, not a failed ingest.

### On the tower (Ocala or Albuquerque)

From the repo root, after someone has pushed a recorded entry:

```powershell
git pull
cd video_buddy
.\.venv\Scripts\python.exe -m master_agent kb ingest
.\.venv\Scripts\python.exe -m master_agent kb stats
.\.venv\Scripts\python.exe -m master_agent kb search "the symptom you care about" --knowledge
```

`kb ingest` prints `N knowledge doc(s)` and a skipped-file count. Run it
twice: `knowledge=` in `kb stats` stays the same. `agent health` includes
`kb.knowledge`; `agent search_knowledge --args '{"query": "..."}'` searches it.
