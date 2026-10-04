> **Archived 2026-10-02.** Superseded by [`docs/INDEX.md`](../../../INDEX.md). If this copy disagrees with `docs/`, `docs/` wins.

# Pack C — Seedance 2.5 one-take (local only)

**Local-only is a hard requirement.** This route uses no cloud APIs, no cloud services, and no hosted inference. Everything it runs is local.

Generate only on `http://127.0.0.1:8188` with the on-disk catalog:

| Brief | Local catalog id |
|---|---|
| Text / one-take | `ltx25_t2v_i2v` |
| First frame | `ltx25_t2v_i2v` |
| First + last | `ltx25_flf2v` |
| References | `ltx25_msr` |

Wan (`wan22`) and MiniMax H3 (`h3_t2v`, `h3_i2v`, `h3_flf`, `h3_r2v`) stay available when you pass that catalog id. Pack C does not replace them and does not send their burns to a cloud host.

Optional Grok may rank a story. That client is not video inference. A Pack C burn is still local Comfy.

Policy: `PACK_C_LOCAL_ONLY` in `master_agent/config.py`.

## Route the agent runs

1. Write the one-take prompt locally.
2. Scout and move on `http://127.0.0.1:8188` with the catalog ids above.
3. Reuse an explicit seed when repeating the same shot.

```text
16:9, cinematic, single continuous take. Refs: @Image1 character, @Image2 location, @Clay Render 1 blocking. 0–10s: … 10–20s: … 20–30s: …. No subtitles, No BGM. Keep identity/wardrobe locked.
```

One continuous take. A multi-body fight stays out of one 30s pass.

`python -m master_agent run "Seedance 2.5 draft one-take"` fail-closes onto `ltx25_t2v_i2v`. An explicit local id (`--variant wan22`, `--variant h3_t2v`, `--variant ltx25_flf2v`) is honored. A Partner stub id is rewritten to the local pack for that mode and is not loaded.

## Field-shape record (not an execution route)

`master_agent/comfy/partner_pointers.py` records the Partner node shape. Those class types call hosted inference. Buddy does not load, enable, or queue them.

| Buddy stub | Recorded template id | Recorded scout class |
|---|---|---|
| `seedance25_draft_t2v` | `api_seedance2_5_draft_t2v` | `ByteDance2TextToVideoNode` |
| `seedance25_draft_i2v` | `api_seedance2_5_draft_i2v` | `ByteDance2FirstLastFrameNode` |
| `seedance25_draft_r2v` | `api_seedance2_5_draft_r2v` | `ByteDance2ReferenceNodeV2` |

Recorded promote class: `ByteDance2DraftToFinalVideoNode`. Recorded model option: `Seedance 2.5 Draft`. Recorded model id: `dreamina-seedance-2-5-260628`. Recorded scout resolution: 480p with `draft_task_id`.

`partnerGraphsExecutable` is false. No `workflows/**/api_seedance2_5_draft_*.json`. No comfy.org, BytePlus, ModelArk, or KIE client on this route.

`python -m master_agent workflows` lists the stubs under field-shape records. They are not in `WORKFLOW_FILES`, `GET /api/variants`, or the studio picker.
