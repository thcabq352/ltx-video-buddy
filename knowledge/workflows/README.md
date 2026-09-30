# Workflows

Which Comfy workflow graphs worked for which shot types.

Graphs live under [`video_buddy/workflows/`](../../video_buddy/workflows/)
(LTX 2.5 in `ltx-2.5/`, MiniMax H3 in `minimax-h3/`, plus the curated
library at the workflows root). Link the JSON. Do not paste the graph.

Schema: [`../AGENTS.md`](../AGENTS.md). One file per shot-type note,
named `YYYY-MM-DD-<shot>-<buddy-id>.md`.

## EXAMPLE entry — not a recorded learning

The block below is the shape of a real note. It is not a claim that
this graph won a cold-brew test.

```markdown
# EXAMPLE — product pour, text-to-video — h3_t2v

- **status:** example
- **shot type:** text-to-video product pour, no identity lock
- **buddy id:** `h3_t2v`
- **graph:** video_buddy/workflows/minimax-h3/MiniMax-H3_T2V_FL2VA_api.json
- **why it worked:** Placeholder. A real note says what the graph held
  (pour continuity, lighting, length) in one to three lines.
- **watch-outs:** H3 frame counts use the 17k+5 grid. CFG stays 1.0.
  Do not borrow LTX `8n+1` lengths for this graph.
```

Catalog ids and invoke lines:
[`video_buddy/workflows/minimax-h3/README.md`](../../video_buddy/workflows/minimax-h3/README.md),
[`video_buddy/workflows/ltx-2.5/README.md`](../../video_buddy/workflows/ltx-2.5/README.md).

Seedance 2.5 Draft → Final is not a graph in this tree. Stub ids and the
Comfy Partner template names are in
[`video_buddy/docs/SEEDANCE_2_5_DRAFT.md`](../../video_buddy/docs/SEEDANCE_2_5_DRAFT.md).
