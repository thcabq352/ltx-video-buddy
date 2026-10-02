# Documentation Plan — LTX Video Buddy

**Status:** Draft for review
**Owner:** Scott
**Reference repo:** https://github.com/thcabq352/ltx-video-buddy
**Date:** 2026-10-02
**Branch:** `docs/restructure-slim`

---

## 1. Executive Summary

The engineering in this repo is ahead of its packaging. The core loop — brief, route, patch, validate, render, judge, revise — is solid, tested (775 passing unit tests), and documented in depth. The problem is not missing information; it is **information density and discoverability**.

Today the documentation surface is:

| Document | Size | Role |
|---|---|---|
| Root `README.md` | ~17 KB | Marketing + quickstart + architecture + status |
| `video_buddy/README.md` | ~44 KB | Full operator manual (CLI, doctor, judge, MV, fractal, power mode) |
| `video_buddy/docs/QUICKSTART.md` | ~14 KB | Catalog, doctor, download-models, 16GB policy |
| `video_buddy/docs/MANAGED_COMFY.md` | ~12 KB | Managed vs external Comfy mode |
| `video_buddy/docs/REQUIRED-FILES.md` | ~12 KB | Weight inventory, accepted names, Hub catalog |
| `video_buddy/docs/BRAIN_HANDS.md` | ~3 KB | Capability contract + Hands fit |
| `video_buddy/docs/SELF_IMPROVEMENT_LOOP.md` | ~7 KB | Judge → revise → re-run map |
| `video_buddy/docs/CLIP_PROVENANCE.md` | ~7 KB | Sidecar schema + lifecycle |
| `video_buddy/docs/MUSIC_VIDEO.md` | ~7 KB | Beat plan → Remotion MTV path |
| `video_buddy/AGENTS.md` | ~10 KB | Stop-lines + field lessons |
| `knowledge/README.md` | ~5 KB | Shared-knowledge sync rules |
| `video_buddy/AUDIT.md` | ~3 KB | Wired / unwired / retired matrix |

**Total: roughly 140 KB of prose across 12 files**, with heavy overlap (the 16GB loader policy appears in at least four places; the LTX 2.5 catalog table appears in three; the "doctor never fetches" rule appears in five).

**Goal:** cut the total surface by ~50%, eliminate duplication, and give every reader exactly one authoritative answer per question.

---

## 2. Design Principles

1. **One source of truth per topic.** If a fact lives in two files, one is canonical and the other links to it.
2. **Audience-first structure.** A new contributor, an operator, and an agent (LLM) have different needs. Serve each with a different entry point, not one mega-file.
3. **Progressive disclosure.** Root README answers "what is this and how do I start" in under two minutes. Everything else is one click away.
4. **Docs mirror code.** File paths, CLI commands, and schema IDs in docs must match the repo. Stale docs are worse than no docs.
5. **Field lessons are first-class.** The `knowledge/failures/` pattern (e.g., the 1-frame collapse entry) is the single most valuable documentation habit in this repo. Expand it; never bury it.
6. **No secrets, no absolute paths.** Same rules as provenance sidecars and `knowledge/` entries.
7. **Agent-readable.** `AGENTS.md` and schema docs must be scannable by an LLM in one pass. Short bullets, explicit "do not" lists, no narrative filler.

---

## 3. Target Structure

```
ltx-video-buddy/
├── README.md                      # Landing page (≤ 3 KB). What, why, demo, one-command start.
├── docs/
│   ├── INDEX.md                   # Master table of contents by audience
│   ├── GETTING_STARTED.md         # Install → doctor → first render (replaces QUICKSTART + install sections)
│   ├── ARCHITECTURE.md            # Brain/Hands, state machine, judge loop, data flow
│   ├── CLI_REFERENCE.md           # Full command/flag reference (extracted from operator README)
│   ├── WEIGHTS.md                 # Inventory, doctor, download-models, 16GB policy, accepted names
│   ├── COMFY.md                   # Managed/external mode, attach contract, model paths YAML
│   ├── JUDGE.md                   # Judge legs, quality bar, revise loop, loop_status
│   ├── PROVENANCE.md              # ClipProvenance schema + lifecycle
│   ├── FEATURES.md                # Music video, fractal, LoRA, upscale, HeartMuLa (one section each)
│   ├── AGENTS.md                  # Stop-lines + field lessons (moved from video_buddy/AGENTS.md)
│   └── AUDIT.md                   # Wired / unwired / retired matrix (moved from video_buddy/)
├── knowledge/                     # Unchanged — shared learnings between machines
│   ├── README.md
│   ├── AGENTS.md
│   ├── prompts/
│   ├── workflows/
│   ├── judge-feedback/
│   ├── failures/
│   └── decisions/
└── video_buddy/
    ├── README.md                  # Slimmed to ~10 KB: pointer + install + top-10 commands
    └── docs/                      # Feature-specific deep dives only (LIPDUB, HEARTMULA,
                                  # SEEDANCE_2_5_DRAFT, COMFY_ATTACH*, etc.)
```

### What moves where

| Current location | New location | Action |
|---|---|---|
| Root README (17 KB) | `README.md` (≤ 3 KB) + `docs/GETTING_STARTED.md` | Split: landing vs. tutorial |
| `video_buddy/README.md` (44 KB) | `docs/CLI_REFERENCE.md` + slim `video_buddy/README.md` | Extract reference; keep install + pointers |
| `video_buddy/docs/QUICKSTART.md` | `docs/GETTING_STARTED.md` + `docs/WEIGHTS.md` | Split by topic |
| `video_buddy/docs/MANAGED_COMFY.md` | `docs/COMFY.md` | Rename + merge attach notes |
| `video_buddy/docs/REQUIRED-FILES.md` | `docs/WEIGHTS.md` | Merge with 16GB policy |
| `video_buddy/docs/BRAIN_HANDS.md` | `docs/ARCHITECTURE.md` (section) | Fold in; keep contract example |
| `video_buddy/docs/SELF_IMPROVEMENT_LOOP.md` | `docs/JUDGE.md` | Merge |
| `video_buddy/docs/CLIP_PROVENANCE.md` | `docs/PROVENANCE.md` | Rename |
| `video_buddy/docs/MUSIC_VIDEO.md` | `docs/FEATURES.md` (section) + keep deep dive | Both: summary + link |
| `video_buddy/AUDIT.md` | `docs/AUDIT.md` | Move |
| `video_buddy/AGENTS.md` | `docs/AGENTS.md` | Move; keep as the agent entry point |
| `knowledge/` | unchanged | — |

---

## 4. Per-Document Specs

### 4.1 `README.md` (landing page)

**Audience:** first-time visitor, GitHub browser.
**Length:** ≤ 3 KB. Hard cap.
**Must contain:**
- One-sentence pitch
- Hero demo (existing H3 BMX still + link)
- Three bullets: what it does, what it doesn't (no cloud, no auto-download, consent-first)
- One copy-paste block: install → doctor → first render
- Link to `docs/INDEX.md` for everything else
- Status line (actively developed, test count, last major merge)

**Must NOT contain:** CLI flag tables, weight filename lists, architecture diagrams, changelog history. Those live elsewhere.

### 4.2 `docs/INDEX.md` (master TOC)

**Audience:** anyone who landed past the README.
**Format:** one table, grouped by audience:

| Audience | Start here |
|---|---|
| New user | GETTING_STARTED |
| Operator (daily) | CLI_REFERENCE, WEIGHTS |
| Developer | ARCHITECTURE, AGENTS |
| Agent / LLM | AGENTS (stop-lines), schema docs |
| Contributor | CONTRIBUTING (new), tests/ |

Each row: one link, one five-word description. No prose.

### 4.3 `docs/GETTING_STARTED.md`

**Audience:** someone who cloned and wants a render today.
**Length:** ≤ 4 KB.
**Structure:**
1. Prerequisites (Python 3.10+, one GPU 12GB+, ffmpeg, Ollama)
2. Install (one command per OS)
3. `doctor` — what it checks, what it never does
4. First render (one `comfy run` command, one `run` command)
5. "If something's missing" → point to WEIGHTS.md
6. "If Comfy is elsewhere" → point to COMFY.md

No flag encyclopedias. No 16GB policy tables (link out).

### 4.4 `docs/ARCHITECTURE.md`

**Audience:** developers and agents who need to modify behavior.
**Length:** ≤ 5 KB.
**Structure:**
1. The loop (one diagram: intake → director → Hands → orchestrator → Comfy → judge → revise)
2. Brain / Hands split (contract fields, what never goes on the contract)
3. State machine (SELECT_VARIANT → … → DONE, with OOM ladder and judge loop as sub-flows)
4. Data on disk (`state/runs/`, sidecars, Chroma, knowledge/)
5. "Do not" list (no VRAM on contract, no auto-download, no auto-install packs, no Partner graph queueing)

Fold BRAIN_HANDS.md and SELF_IMPROVEMENT_LOOP.md here. Keep the JSON contract example verbatim.

### 4.5 `docs/CLI_REFERENCE.md`

**Audience:** operators.
**Source:** extracted from `video_buddy/README.md`.
**Format:** grouped command tables (health/doctor, comfy run modes, run/pipeline, music/mv, fractal, lora, kb, hermes). One row per command: invocation, purpose, key flags. Deep flag semantics link out to feature docs.

### 4.6 `docs/WEIGHTS.md`

**Audience:** operators setting up a machine.
**Source:** merged QUICKSTART §2–5 + REQUIRED-FILES.md.
**Structure:**
1. Inventory-first rule (one paragraph)
2. Search roots (ordered list)
3. `doctor` rows (table)
4. `download-models` usage (commands only)
5. 16GB loader preference (numbered list — the canonical version)
6. Accepted-names table (the big one from REQUIRED-FILES)
7. Official Hub catalog (reference only)
8. VRAM bands (NVIDIA/AMD/ROCm sentences from MANAGED_COMFY)

This is the one file where the 16GB policy is authoritative. Everywhere else links here.

### 4.7 `docs/COMFY.md`

**Audience:** operators with an existing Comfy install.
**Source:** MANAGED_COMFY.md + COMFY_ATTACH*.md notes.
**Structure:** managed vs external mode table, model-paths YAML, attach contract summary, stale-pack policy, hardware scan sentences.

### 4.8 `docs/JUDGE.md`

**Audience:** developers tuning quality behavior.
**Source:** SELF_IMPROVEMENT_LOOP.md.
**Structure:** three judge legs, look vs health split, quality-bar rule ids (a/c/d; b skipped), revise plan shape, loop_status values, `--self-improve-dry` for CI, remaining holes.

### 4.9 `docs/PROVENANCE.md`

**Audience:** developers and agents reading run records.
**Source:** CLIP_PROVENANCE.md.
**Structure:** sidecar location, schema id, required fields (keep the JSON example), write/read lifecycle table, code map, tests.

### 4.10 `docs/FEATURES.md`

**Audience:** operators using advanced modes.
**Source:** MUSIC_VIDEO.md + fractal/LoRA/upscale/HeartMuLa sections.
**Format:** one section per feature: 3–5 sentences + one command block + link to deep dive. Deep dives (LIPDUB.md, HEARTMULA.md, SEEDANCE_2_5_DRAFT.md) stay in `video_buddy/docs/`.

### 4.11 `docs/AGENTS.md`

**Audience:** LLMs and human contributors modifying code.
**Source:** `video_buddy/AGENTS.md` (moved, not rewritten).
**Invariant:** this file is the single agent entry point. It contains only stop-lines and field lessons. No install instructions, no CLI reference. If an agent needs those, it follows a link.

### 4.12 `docs/AUDIT.md`

**Audience:** developers and auditors.
**Source:** `video_buddy/AUDIT.md` (moved).
**Unchanged content; updated paths.**

---

## 5. Deduplication Map

| Fact | Canonical home | Other mentions become links |
|---|---|---|
| 16GB loader preference (GGUF → NVFP4 → …) | WEIGHTS.md §5 | doctor output examples, AGENTS.md, QUICKSTART remnants |
| LTX 2.5 catalog table (7 ids) | GETTING_STARTED.md (short) + WEIGHTS.md (full) | CLI_REFERENCE, FEATURES |
| "doctor never fetches" | WEIGHTS.md §1 | GETTING_STARTED, AGENTS, every command example |
| LTX 8n+1 frame law | AGENTS.md (stop-line) | ARCHITECTURE, JUDGE, WEIGHTS |
| Brain/Hands contract | ARCHITECTURE.md | AGENTS.md |
| Quality-bar rule ids | JUDGE.md | AGENTS.md, SELF_IMPROVEMENT remnants |
| ClipProvenance schema | PROVENANCE.md | AGENTS.md, MUSIC_VIDEO remnants |
| TeaCache inject-when-registered | AGENTS.md | COMFY.md, FEATURES |
| Hardware routing sentences | WEIGHTS.md §8 | COMFY.md |

**Rule:** after the migration, `grep -r "GGUF Q4" docs/` should return exactly one authoritative block plus links.

---

## 6. Migration Plan

### Phase 0 — Prep (1 session)
- [ ] Create `docs/` at repo root if absent; add `docs/INDEX.md` skeleton
- [ ] Inventory every internal link in current docs (script: `grep -rhoE '\[[^\]]+\]\([^)]+\)'`)
- [ ] Snapshot current docs to a `docs-archive/` branch or tag so nothing is lost

### Phase 1 — Extract (2–3 sessions)
- [ ] Write new `README.md` landing page (≤ 3 KB) from existing hero/demo/pitch content
- [ ] Write `docs/GETTING_STARTED.md` from QUICKSTART install + first-render sections
- [ ] Extract `docs/CLI_REFERENCE.md` from `video_buddy/README.md` CLI section
- [ ] Merge QUICKSTART §2–5 + REQUIRED-FILES.md → `docs/WEIGHTS.md`
- [ ] Move MANAGED_COMFY.md → `docs/COMFY.md` (content unchanged, paths updated)
- [ ] Move AUDIT.md → `docs/AUDIT.md`
- [ ] Move `video_buddy/AGENTS.md` → `docs/AGENTS.md`

### Phase 2 — Fold (2 sessions)
- [ ] Merge BRAIN_HANDS.md + SELF_IMPROVEMENT_LOOP.md → `docs/ARCHITECTURE.md` + `docs/JUDGE.md`
- [ ] Rename CLIP_PROVENANCE.md → `docs/PROVENANCE.md`
- [ ] Write `docs/FEATURES.md` as summary sections; keep deep dives in `video_buddy/docs/`
- [ ] Slim `video_buddy/README.md` to install + top-10 commands + pointers

### Phase 3 — Verify (1 session)
- [ ] Every link resolves (CI check or script)
- [ ] Every CLI command in docs runs (smoke: `--help` or `--dry-run` where applicable)
- [ ] Grep dedup check (Section 5 rule)
- [ ] 775 tests still pass
- [ ] Agent smoke test: feed `docs/AGENTS.md` + `docs/ARCHITECTURE.md` to an LLM and confirm it can answer "how do I add a new variant?" correctly

### Phase 4 — Retire
- [ ] Delete or redirect old paths (QUICKSTART.md, REQUIRED-FILES.md, MANAGED_COMFY.md, BRAIN_HANDS.md, SELF_IMPROVEMENT_LOOP.md, CLIP_PROVENANCE.md, MUSIC_VIDEO.md, video_buddy/AUDIT.md, video_buddy/AGENTS.md)
- [ ] Add redirects (GitHub supports `_redirects` or simple stub files with "moved to…") so external links don't 404
- [ ] Update root README's Documentation section to point at `docs/INDEX.md` only

---

## 7. Success Metrics

| Metric | Current | Target |
|---|---|---|
| Total doc prose | ~140 KB | ≤ 70 KB |
| Files in docs surface | 12 | ≤ 14 (more files, less each) |
| Overlapping facts | 5+ copies of 16GB policy | 1 canonical + links |
| Time-to-first-render (new user) | unknown | ≤ 15 min guided by GETTING_STARTED |
| Broken links | unknown | 0 (CI-enforced) |
| Agent can route a task from docs alone | partial | yes (AGENTS.md + ARCHITECTURE.md sufficient) |

---

## 8. Open Questions

1. **Keep `video_buddy/README.md` as a separate operator entry, or make root README the only entry?** Recommendation: keep a slim one; operators `cd video_buddy` and expect a README there. **Decision: yes, slim.**
2. **Should `docs/` live at repo root or inside `video_buddy/`?** Recommendation: repo root, because `knowledge/`, `docs/demo/`, and `docs/showcase/` already live at root — one docs home.
3. **CONTRIBUTING.md?** Not in current repo. Recommend adding in Phase 3: branch naming, test requirements, knowledge/ commit rules.
4. **White paper / investor materials** are "available on request" per current README. Keep that; don't publish.
5. **Per-workflow READMEs** (`workflows/ltx-2.5/README.md`, `workflows/minimax-h3/README.md`) — keep as-is; they're discoverable from the workflow folders and short.

---

## 9. Effort Estimate

| Phase | Effort |
|---|---|
| Phase 0 | 2–3 hours |
| Phase 1 | 1–2 days |
| Phase 2 | 1 day |
| Phase 3 | 4–6 hours |
| Phase 4 | 2–3 hours |
| **Total** | **~4–5 working days** |

Can be split across sessions; each phase produces a reviewable diff.

---

*This plan treats the current documentation as source material, not as the target. The target is a docs surface that matches the quality of the code it describes.*
