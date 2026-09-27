# Long-audio lipdub (`ltx25_a2v`)

A still plus a voice wav. Audio at or under the threshold (default **6.5s**) is one pass, the same render as before. Longer audio is split, rendered, and stitched. The original wav is muxed back on. The sidecar stays `buddy.clip.provenance/v1` with an optional `params.lipdub` block (same pattern as `params.voice_sample`).

## Why a 12s pass loses the mouth

There is no hard 7-second attention mask. LTX 2.5 a2v is bidirectional cross-attention with temporal RoPE, and the cloud API accepts audio well past 12s.

Locally, `frames_for_duration` caps every LTX 2.5 latent at 193 frames (~8.04s) while the audio-trim widget still gets the full wav. The latent ends and the mouth holds. On the tower, motion died near **7.4s**, which is the legal count **177 frames = 7.375s**, *before* that 193-frame cap. The distilled latent stops tracking audio inside the window. The default slice stays at or under **169 frames (7.04s)** after the 8-frame overlap snap.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--lipdub-max-s` | `6.5` (`LIPDUB_SEGMENT_MAX_S`) | Split when audio is longer than this. At or under it, one pass. |
| `--silence-min-s` | `0.25` (`LIPDUB_SILENCE_MIN_S`) | Pauses at least this long are closed-mouth. |
| `--silence-mode` | `idle` | `idle` renders a closed-mouth breathing piece for each pause (default). `hold` is the #34 still plate and frozen tail. `bridge` is a 9-frame mouth close; the tail past 9 frames is held. A pause shorter than 9 frames stays a trimmed bridge in `idle`. |
| `--anchor` | `hybrid` | `hybrid` conditions on the source still and crossfades the previous frame over the overlap. `source` is the still only, with the same crossfade. `previous` is the #34 last-frame chain. |
| `--reframe` | on when segmented, off for a short pass | After each piece, measure zoom and shift against the source still and scale the frames back. Recorded as `scale_drift`. |
| `--lipdub-overlap` | `8` (`LIPDUB_OVERLAP_FRAMES`) | Frames of overlap trimmed at a seam. `hybrid` / `source` crossfade those frames instead of dropping them unseen. |
| `--words` | none | JSON word timestamps (`[{w,s,e}]` or `{"words":[...]}`). Splits prefer pauses and never cut a word. A word longer than the cap is kept whole. |
| `--tripod` | off | Locked-off talking head. Prompt, negative, and stage-1 `LTXVImgToVideoInplace` strength `0.7 → 0.85`. Stage 2 stays `1.0`. The graph has no camera-motion input. |
| `--dry-run` | | Prints idle pieces, anchor, reframe, and the segment plan, then validates each Comfy piece. Nothing is queued. |

H3 (`h3_r2v`) is unchanged: one clip, capped at 12s.

## Continuity and silence

The a2v graph (`LTX-2.5_A2V_Two_Stage_Distilled_api.json`) has one `LoadImage` into both `LTXVImgToVideoInplace` stages. There is no second keyframe and no `LTXVAddGuide` node. A guide node is not added.

Default `--anchor hybrid` puts the **original source still** on that image slot (identity and framing). The previous piece's last frame is only a continuity guide: it is blended into the conditioning image at `hybrid_prev_weight` 0.25 after its zoom is undone, and the overlap frames are crossfaded into the previous piece's tail. Stage 2 strength stays `1.0`, so frame 0 of a render copies the guide; those frames sit in the overlap and are blended away, which is what keeps the head from snapping back to the still. `--anchor source` uses the still with the same crossfade and no previous-frame blend. `--anchor previous` is the #34 chain (each piece starts from the previous last frame, overlap trimmed, no crossfade).

`--reframe` (default on only when the clip is segmented) samples frames, estimates zoom and shift against the source still (OpenCV ECC when `cv2` is already installed, otherwise a normalized cross-correlation search), smooths the track, and scales the piece back. Borders uncovered by the shrink come from the source still. The measured zoom is `params.lipdub.scale_drift` (`scale` > 1 is a push-in). No new weights are downloaded.

Silence (`--silence-mode`, default `idle`):

- A pause at least `--silence-min-s` and at least 9 frames is an a2v idle: closed mouth, a breath, a blink, micro head motion, and a negative for talking, an open mouth, and camera motion. The graph is fed the real pause slice of the wav (silence or room tone). The original wav is still muxed at the end.
- A shorter pause stays a trimmed mouth bridge. Idle mode does not hold a frozen frame for more than 3 frames.
- `hold` is the #34 behaviour: leading still plate, 9-frame bridge, frozen tail.
- `bridge` makes every pause that 9-frame close, including the lead-in. The tail past 9 frames is held.
- Frame grid is `round(duration * fps)`. 12.0s at 24 fps is 288 frames. Every render length is `8n+1`. Pieces tile `[0, timeline)`.

`--tripod` is unchanged and still opt-in. With the flag off, a short clip's prompt and strength are unchanged. Passing `--reframe on` on a short clip is the one short-pass post-step; idle and anchor apply only when the audio is split.

## Provenance

`outputs/<run>/shot-1.buddy.json` is still `buddy.clip.provenance/v1`. `params.lipdub` keeps the #34 keys and adds optional fields (no second schema): `silence_mode`, `anchor`, `reframe`, `join`, `scale_drift`, `hybrid_prev_weight`. Each segment may also carry `crossfade_frames`, `scale`, `dx`, `dy`, `audio_feed`, and `identity_frame`. All of these are optional. The Rust `ClipProvenance` in `your-video-buddy` should mirror them as optional fields on the existing lipdub object. If that struct denies unknown fields, add the new `Option`s the same way it allows `voice_sample`. This repo does not fork the schema id.

## Tower: rerender the 12s ringmaster test

From `video_buddy/` (or with `PYTHONPATH=video_buddy`). No `LTXV_API_KEY`. `LLM_PROVIDER=ollama`.

```bash
python -m master_agent run \
  "The ringmaster clown speaks directly to camera in this exact voice recording, saying word for word: 'I would like to make this a sound sample of what my voice is, so we could clone my voice, so we could clone whatever we want. Hey, thank you.' Lips tightly synced to the voice with clear articulation; his lips press fully closed on every m, b and p sound, and his mouth rests closed during pauses. His face paint, red ringmaster coat and axe-cane stay locked and unchanged; minimal head motion. Static locked-off camera, no push-in, no zoom. Smoky circus tent behind him. Portrait framing." \
  --variant ltx25_a2v \
  --image ringmaster_notitle.png \
  --audio scott_voice_sample_12s.wav \
  --words words.json \
  --line "I would like to make this a sound sample of what my voice is, so we could clone my voice, so we could clone whatever we want. Hey, thank you." \
  --width 352 --height 480 --seed 42 \
  --silence-mode idle --anchor hybrid --reframe on \
  --tripod --no-interview --storyboard off --no-judge \
  --llm-panel local --panel-judge ollama \
  --dry-run
```

Drop `--dry-run` to render. `--words` should be the faster-whisper timestamps for that wav (pauses near 0–0.9s, 5.7–6.25s, and 10.6–11.15s). Without `--words`, Buddy still splits on energy pauses or, if those cannot be read, on the max length.
