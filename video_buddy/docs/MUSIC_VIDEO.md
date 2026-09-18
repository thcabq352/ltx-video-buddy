# Music-video mode (Comfy/LTX → Remotion)

MTV-style cut: beat-synced windows, one **unique motion clip** per window
burned on **local ComfyUI / LTX**, then assembled with **Remotion** plus the
full track. This is the product shape of Scott’s Grok+Remotion “I’m in love
with a bot” cut — **without** Grok Imagine or any cloud video API.

Engine = the existing LTX graph path (`prepare_run` → `ensure_teacache`
inject-when-registered, soft-bypass if the TeaCache pack is missing).
Do not install Comfy packs from this command. Do not download weights.

## Flow

```
audio ──▶ buddy.mv.beat_plan/v1
              │
              ▼
        Comfy/LTX burn   one clip / window
        I2V if --image   else T2V
        still-hold fail
              │
              ▼
        uniqueness gate  (SHA-256; refuse stitch on dups)
              │
              ▼
        Remotion         Sequence @ 30fps 1080p + full audio
                         out/MV-FIXED.mp4
```

```bash
cd video_buddy
python -m master_agent mv plan --audio track.mp3 --out out/beat_plan.json
python -m master_agent mv render "I'm in love with a bot" \
  --audio track.mp3 --out out/MV-FIXED.mp4
python -m master_agent mv render --audio track.mp3 --image still.png   # I2V lock
python -m master_agent mv render --audio track.mp3 --dry-run           # no GPU
```

`--dry-run` proves plan + unique mock burns + Remotion wiring. It does not
queue Comfy and does not need Node.

## Beat-plan schema (`buddy.mv.beat_plan/v1`)

```json
{
  "schema": "buddy.mv.beat_plan/v1",
  "fps": 30,
  "duration_s": 16.0,
  "duration_frames": 480,
  "bpm": 120.0,
  "audio_path": "track.mp3",
  "windows": [
    {
      "index": 0,
      "start_s": 0.0,
      "end_s": 4.0,
      "start_frame": 0,
      "end_frame": 120,
      "energy": 0.25,
      "label": "verse"
    }
  ]
}
```

Library: `master_agent.music.plan.build_beat_plan` /
`build_beat_plan_from_audio`. Seconds come from the existing spectral-flux
beat grid; frames are the 30 fps snap, then stitched so they tile
`[0, duration_frames)`. A window that collapses to 0/1 frame is a
**still-hold** and is refused.

## Comfy requirements

- ComfyUI on `:8188` for a live burn (`--dry-run` skips the queue).
- LTX variant default: `ltx25_t2v_i2v` (same graph for T2V and I2V).
- `--image` / character lock → I2V (`image_name` on `prepare_run`).
  Without a still → T2V.
- TeaCache: `graph_ops.ensure_teacache` inside `prepare_run`. Missing pack
  soft-bypasses. **No auto-install.**
- Each window writes `window-NNN.mp4` plus `window-NNN.buddy.json`
  (`buddy.clip.provenance/v1`): prompts, engine, params, `window_id`, hash.
- Hard-fail if a segment would be a still-hold (1-frame plan, still image
  used as the clip, dead motion / &lt;3 frames on a live probe).

## Uniqueness gate

Before Remotion, every window clip is SHA-256 hashed. If any two share
bytes, stitch is refused with a clear error listing the duplicate indices:

```
FAIL  uniqueness gate: refuse stitch: duplicate window clips: windows [0, 3] share hash …
```

## Remotion render

In-repo project: `video_buddy/remotion-mv/`.

```bash
cd video_buddy/remotion-mv
npm install
npx remotion render src/index.ts MusicVideo /abs/path/out/MV-FIXED.mp4 \
  --props=/abs/path/to/remotion-props.json
```

`mv render` writes `remotion-props.json` + `remotion-command.txt` next to
the clips. Live mode runs that command. Output is **1080p** (1920×1080 @
30 fps) as `out/MV-FIXED.mp4`.

**720p note:** add `--scale=0.666667` (1280×720) to the same command, or
`npm run render:720p` inside `remotion-mv/`.

## vs single-shot Buddy / `music`

| | `run` / `diagnose` / `comfy run` | `music` | `mv render` |
|---|---|---|---|
| Job | One (or storyboard) LTX shot | Beat-snapped shots, **ffmpeg** concat + mux | Beat plan → unique LTX burns → **Remotion** |
| Audio | Optional | Muxed after concat | Full track on the Remotion timeline |
| Still-holds | Judge / quality bar | Trim + mux can keep a weak clip | **Hard-fail** |
| Duplicate clips | Not gated | Not gated | **Refuse stitch** |
| Cloud / Grok | Optional LLM only | Optional LLM only | **Never** a render engine |

`music` stays as the ffmpeg-mux path. `mv` is the MTV product path.
Diagnose / draft / generate are unchanged — MV reuses `prepare_run`, it
does not fork a second orchestrator.
