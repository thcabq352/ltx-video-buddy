# Seedance 2.5 Draft → Final (Comfy Partner pointer)

Video Buddy does not queue this path. The default catalog is local Comfy
graphs (LTX 2.5 / 2.3, MiniMax H3, Wan, Flux). There is no ByteDance client
and no Partner workflow JSON in `workflows/`.

20–30s cloud one-takes stay on **Comfy Partner** templates. Open them in a
ComfyUI that is signed in to Partner nodes (template index: ComfyUI
**≥ 0.37.3**). Buddy will refuse a `run` / `comfy run` that names these ids
or asks for Seedance, and print the template to open.

## Stub ids (not queueable)

Listed by `python -m master_agent workflows`. They are not in
`WORKFLOW_FILES`, `GET /api/variants`, or the studio variant picker.

| Buddy stub | Comfy template | Scout node in the official template |
|---|---|---|
| `seedance25_draft_t2v` | `api_seedance2_5_draft_t2v` | `ByteDance2TextToVideoNode` |
| `seedance25_draft_i2v` | `api_seedance2_5_draft_i2v` | `ByteDance2FirstLastFrameNode` (one start frame) |
| `seedance25_draft_r2v` | `api_seedance2_5_draft_r2v` | `ByteDance2ReferenceNodeV2` |

Source of truth in code: `master_agent/comfy/partner_pointers.py`.

The image-to-video template is the Draft option on the first-last-frame
node, with a single `LoadImage`. It is not a separate I2V class.

## Scout, then promote

Both stages are already in each official template. Stage 2 starts bypassed.

1. **Scout.** On the scout node, model **`Seedance 2.5 Draft`**. That option
   is the Comfy surface for API `draft: true` on model
   `dreamina-seedance-2-5-260628`: resolution locks to **480p** and the node
   outputs **`draft_task_id`**. `SaveText` writes that id to a `.txt` in the
   output folder. Same prompt and refs as the take you mean to keep.
2. **Fix the seed** before you promote. The shipped templates set
   `control_after_generate` to `randomize`. A second Stage 1 run with a new
   seed mints a new draft instead of the take you reviewed. Set the seed
   control to fixed, or bypass Stage 1 (`Ctrl+B` / `Cmd+B`) once the preview
   is the one you want.
3. **Promote.** Paste `draft_task_id` into **`ByteDance2DraftToFinalVideoNode`**
   (display name **ByteDance Seedance 2.5 Draft to Final Video**). That node
   renders native **1080p** and reuses the draft's prompt, refs, duration,
   aspect ratio, and audio setting. Enable Stage 2, leave Stage 1 bypassed.
4. Draft ids last about **7 days**. Promote inside that window.
5. Set `generate_audio` off when post owns the soundtrack. The text-to-video
   template ships that widget on (true).
6. There is no Seedance API 4K on this path. Promote is 1080p.

Prompt shape for the Partner node (unchanged pack text, not a Buddy variant):

```text
16:9, cinematic, single continuous take. Refs: @Image1 character, @Image2 location, @Clay Render 1 blocking. 0–10s: … 10–20s: … 20–30s: …. No subtitles, No BGM. Keep identity/wardrobe locked.
```

Avoid multi-body fights in one 30s pass.

To render on this machine instead of Partner, pass an on-disk catalog id,
for example `--variant ltx25_t2v_i2v`. Buddy will not silently substitute
`base` for a Seedance brief.

## What this change deliberately does not do

- No vendored `api_seedance2_5_draft_*.json`. Any `workflows/**/*.json` shows
  up in the Comfy template picker, and an API-format graph would enter the
  default catalog.
- No director keyword that selects a local LTX graph and calls it Seedance.
- No ModelArk or `comfy.org` HTTP client, no API key, no live Partner call.
- Kling Partner remap, LTX 2.5 / H3 / Wan catalogs, and the judge loop are
  unchanged. The self-improvement loop still does not score Partner renders.

## Sources checked for this pointer

- Comfy template index names `api_seedance2_5_draft_{t2v,i2v,r2v}` (2026-09-25)
- Official graphs: scout nodes above, promote `ByteDance2DraftToFinalVideoNode`
- [ComfyUI PR #16529](https://github.com/Comfy-Org/ComfyUI/pull/16529) — Draft model option, 480p, `draft_task_id`, Draft-to-Final → 1080p
- [Seedance 2.5 Draft tutorial](https://docs.comfy.org/tutorials/partner-nodes/bytedance/seedance-2-5-draft)
