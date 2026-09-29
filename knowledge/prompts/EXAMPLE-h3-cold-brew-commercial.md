# EXAMPLE — H3 cold-brew commercial

> **EXAMPLE — not a recorded learning.** Invented so the prompt schema
> is obvious. The root README uses "a premium cold-brew commercial" as
> a product sentence; nobody queued the text below. Do not cite this
> file as evidence that a setting works.

- **status:** example
- **goal:** 8-second premium cold-brew spot, one hero pour, no logos.
- **model:** MiniMax H3, buddy id `h3_t2v` (alias `fl2va`)
- **key parameters:** 4 steps, CFG 1.0, 24 fps, duration on the H3
  frame grid (not LTX `8n+1`). Seed and pixel size omitted on purpose —
  fill them from the sidecar when this becomes a real entry.
- **workflow:** [`video_buddy/workflows/minimax-h3/MiniMax-H3_T2V_FL2VA_api.json`](../../video_buddy/workflows/minimax-h3/MiniMax-H3_T2V_FL2VA_api.json)
- **prompt:**

  ```text
  Premium cold-brew commercial, one continuous 8-second shot. A glass
  bottle of dark coffee pours over clear ice in a short tumbler on a
  walnut bar. Slow side-on camera, warm practical light, condensation
  on the glass, no logos, no text, no hands entering frame.
  ```

- **result note:** Example only — replace this line with what was
  actually good enough to reuse (subject lock, pour continuity, grade).
