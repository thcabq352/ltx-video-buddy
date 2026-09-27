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
| `--silence-mode` | `idle` (`LIPDUB_SILENCE_MODE`) | `idle` renders a closed-mouth breathing piece for each pause (default). `hold` is the #34 still plate and frozen tail. `bridge` is a 9-frame mouth close; the tail past 9 frames is held. A pause shorter than 9 frames stays a trimmed bridge in `idle`. |
| `--anchor` | `previous` (`LIPDUB_ANCHOR`) | `previous` is the last-frame chain (default). `hybrid` (experimental) conditions on the source still and crossfades the previous frame over the overlap. `source` (experimental) is the still only, with the same crossfade. `pause-reset` (opt-in) chains speech from the previous frame and pulls each silence back to the source still. |
| `--reframe` | `off` (`LIPDUB_REFRAME`) | Default off. `on` is experimental: measure zoom and shift against the source still and scale the frames back. Recorded as `scale_drift`. |
| `--lipdub-overlap` | `8` (`LIPDUB_OVERLAP_FRAMES`) | Frames of overlap. Default `previous` and `pause-reset` trim them only on a speech-to-speech seam. Experimental `hybrid` / `source` crossfade those frames instead of dropping them unseen. |
| `--max-piece-seconds` | `3.0` (`LIPDUB_MAX_PIECE_S`) | After the clip is already segmented, split a speech run longer than this at the quietest audio frame. Word edges are snapped onto the frame grid first (faster-whisper is 0.02s; 24fps is 1/24s). The cut is that frame, so the join stays locked to the wav. A window with no snapped edge keeps the run whole. This does not change the 6.5s one-pass threshold. |
| `--pause-reset-strength` | `0.65` (`LIPDUB_PAUSE_RESET_STRENGTH`) | `LTXVAddGuide` strength when `--anchor pause-reset` pins the source still. `1.0` turned the head in about 0.2s. |
| `--pause-reset-min-s` | `0.5` (`LIPDUB_PAUSE_RESET_MIN_S`) | Only pauses at least this long get the source keyframe. Shorter breaths stay plain idle. |
| `--words` | none | JSON word timestamps (`[{w,s,e}]` or `{"words":[...]}`). Splits prefer pauses and never cut a word. A word longer than the cap is kept whole. |
| `--tripod` | off | Locked-off talking head. Prompt, negative, and stage-1 `LTXVImgToVideoInplace` strength `0.7 → 0.85`. Stage 2 stays `1.0`. The graph has no camera-motion input. |
| `--dry-run` | | Prints idle pieces, anchor, reframe, and the segment plan, then validates each Comfy piece. Nothing is queued. |

H3 (`h3_r2v`) is unchanged: one clip, capped at 12s.

## Continuity and silence

The a2v graph (`LTX-2.5_A2V_Two_Stage_Distilled_api.json`) has one `LoadImage` into both `LTXVImgToVideoInplace` stages. `hybrid`, `source`, and `previous` do not add a second keyframe.

`pause-reset` is opt-in. Speech pieces still chain from the previous piece's last frame (steady framing, trimmed speech-to-speech overlap, no crossfade across speech). Each idle or mouth-bridge at least `--pause-reset-min-s` (default 0.5s) is frame 0 = that previous frame and a source-still keyframe a few frames before the last kept frame (`LTXVAddGuide`, default strength 0.65, lead 4 frames). Shorter breaths stay plain idle, with no keyframe. The keyframe is wired on the video latent after `LTXVImgToVideoInplace` and before `LTXVConcatAVLatent`, then `LTXVCropGuides` after `LTXVSeparateAVLatent`. Both nodes are core ComfyUI-LTXVideo / Comfy `nodes_lt` classes. Nothing new is downloaded. `doctor` and a lipdub dry-run say so when either class is missing, or when the graph cannot take the keyframe. The fallback re-anchors that pause on the source still and crossfades the first 8 frames (or fewer, if the pause is shorter) inside the silence only.

Default `--anchor previous` is the #34 last-frame chain. Each piece after the first generated piece starts from the previous piece's last frame. Overlap is trimmed only on a speech-to-speech seam, and there is no pixel crossfade. When idle pauses sit between speech pieces, those seams have no overlap drop (`join: none`). Tower seed 42 scored this best on framing and seams: steady framing, smooth joins, natural idle pauses.

`--anchor hybrid` and `--anchor source` are experimental opt-ins. Hybrid puts the **original source still** on the image slot and blends the previous last frame in at `hybrid_prev_weight` 0.25 after its zoom is undone, then crossfades the overlap into the previous piece's tail. Source uses the still with that crossfade and no previous-frame blend. On the same seed-42 tower those joins ghosted and speech pieces zoomed in and then snapped back (4/10 with reframe on). Stage 2 strength stays `1.0`, so frame 0 of a render copies the guide.

`--reframe` defaults **off** for a short pass and for a segmented lipdub. `--reframe on` is experimental: it samples frames, estimates zoom and shift against the source still (OpenCV ECC when `cv2` is already installed, otherwise a normalized cross-correlation search), smooths the track, and scales the piece back. Borders uncovered by the shrink come from the source still. The measured zoom is `params.lipdub.scale_drift` (`scale` > 1 is a push-in). No new weights are downloaded. Leave it off unless you are comparing that snap-back on purpose.

Silence (`--silence-mode`, default `idle`):

- A pause at least `--silence-min-s` and at least 9 frames is an a2v idle: closed mouth, a breath, a blink, micro head motion, and a negative for talking, an open mouth, and camera motion. The graph is fed the real pause slice of the wav (silence or room tone). The original wav is still muxed at the end.
- A shorter pause stays a trimmed mouth bridge. Idle mode does not hold a frozen frame for more than 3 frames.
- `hold` is the #34 behaviour: leading still plate, 9-frame bridge, frozen tail.
- `bridge` makes every pause that 9-frame close, including the lead-in. The tail past 9 frames is held.
- Frame grid is `round(duration * fps)`. 12.0s at 24 fps is 288 frames. Every render length is `8n+1`. Pieces tile `[0, timeline)`.

`--tripod` is unchanged and still opt-in. With the flag off, a short clip's prompt and strength are unchanged. Passing `--reframe on` on a short clip is the one short-pass post-step; idle and anchor apply only when the audio is split. Omitting `--anchor` and `--reframe` records `previous` and `reframe: false` on `params.lipdub`.

## Provenance

`outputs/<run>/shot-1.buddy.json` is still `buddy.clip.provenance/v1`. `params.lipdub` keeps the #34 keys and adds optional fields (no second schema): `silence_mode`, `anchor`, `reframe`, `join`, `scale_drift`, `hybrid_prev_weight`, `max_piece_s`, `guide_node`, `guide_available`, `pause_reset`, `pause_reset_strength`, `pause_reset_min_s`, `pause_reset_guide_lead`. Each segment may also carry `crossfade_frames`, `scale`, `dx`, `dy`, `audio_feed`, `identity_frame`, `end_keyframe`, `silence_crossfade_frames`, `guide_frame_idx`, and `guide_strength`. All of these are optional. The Rust `ClipProvenance` in `your-video-buddy` should mirror them as optional fields on the existing lipdub object. If that struct denies unknown fields, add the new `Option`s the same way it allows `voice_sample`. This repo does not fork the schema id.

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
  --silence-mode idle --anchor previous --reframe off \
  --tripod --no-interview --storyboard off --no-judge \
  --llm-panel local --panel-judge ollama \
  --dry-run
```

Drop `--dry-run` to render. Those three flags are the defaults; writing them out makes the plan explicit. `--words` should be the faster-whisper timestamps for that wav (pauses near 0–0.9s, 5.7–6.25s, and 10.6–11.15s). Those edges are on a 0.02s grid. Buddy snaps each start and end to the nearest 24fps frame before it looks for a legal cut, then picks the lowest-energy snapped edge. Without `--words`, Buddy still splits on energy pauses or, if those cannot be read, on the max length. Pass `--max-piece-seconds 6.5` to keep the unsplit pause plan.

With the voice wav, that snap is **9 pieces** (6 speech, 3 idle), not the 8 a words-only plan prints. Nine means the first speech run splits twice: the quietest snapped edge in the opening run (the end of the long "a", 2.625s) leaves more than 3s, and that remainder is cut again. The unit-test wav puts a deep dip on that edge and a shallower dip on every other snapped edge. Its dry-run is:

| # | kind | time | keep | render | guide |
|---|---|---|---|---|---|
| 0 | silence_idle | 0.000–0.875 | 21 | 25 | leading still, not pinned |
| 1 | speech | 0.875–2.625 | 42 | 49 | previous frame |
| 2 | speech | 2.625–4.875 | 54 | 65 | previous frame, drop 8 |
| 3 | speech | 4.875–5.708 | 20 | 33 | previous frame, drop 8 |
| 4 | silence_idle | 5.708–6.250 | 13 | 17 | source still at frame 8, strength 0.65 |
| 5 | speech | 6.250–8.542 | 55 | 57 | previous frame |
| 6 | speech | 8.542–10.500 | 47 | 57 | previous frame, drop 8 |
| 7 | silence_idle | 10.500–11.250 | 18 | 25 | source still at frame 13, strength 0.65 |
| 8 | speech | 11.250–12.000 | 18 | 25 | previous frame |

A deeper dip on a different snapped edge moves the later speech cuts. The piece count stays 9 while the first cut leaves more than 3s. A words-only plan with no wav (every edge equally quiet, so the latest legal frame wins) stays at 8 pieces, cuts at 3.500s and 8.917s, render lengths 25, 65, 65, 17, 65, 49, 25, 25.

`--max-piece-seconds 6.5` is the unsplit pause plan: 6 pieces, `join=none`, render lengths 25, 121, 17, 105, 25, 25 frames (318 frames total). #34 v2 was 5 jobs and about 28 minutes. The idle plan adds the leading idle as a real job. Expect on the order of **32–36 minutes** for 6 jobs, longer for the 9-piece wav plan. `--anchor hybrid --reframe on` is the experimental plan that scored 4/10 (speech pieces zoomed in and snapped back, joins ghosted); with the 6.5s piece cap it renders 25, 129, 25, 113, 33, 33 frames (358) because each join crossfades 8 overlap frames.

Likeness lock (opt-in, one seed-42 render). Speech stays on the previous frame. Silences at least 0.5s glide back to the still (`LTXVAddGuide` strength 0.65, four frames before the last kept frame) when that node is registered; shorter breaths stay plain idle. Otherwise the dry-run says the in-silence crossfade is in use. `--reframe off` so a measured scale correction does not fight that glide. `--max-piece-seconds 3` with the wav is the 9-piece plan above.

```bash
python -m master_agent run \
  "The ringmaster clown speaks directly to camera in this exact voice recording, saying word for word: 'I would like to make this a sound sample of what my voice is, so we could clone my voice, so we could clone whatever we want. Hey, thank you.' Lips tightly synced to the voice with clear articulation; his lips press fully closed on every m, b and p sound, and his mouth rests closed during pauses. His face paint, red ringmaster coat and axe-cane stay locked and unchanged; minimal head motion. Static locked-off camera, no push-in, no zoom. Smoky circus tent behind him. Portrait framing." \
  --variant ltx25_a2v \
  --image ringmaster_notitle.png \
  --audio scott_voice_sample_12s.wav \
  --words words.json \
  --line "I would like to make this a sound sample of what my voice is, so we could clone my voice, so we could clone whatever we want. Hey, thank you." \
  --width 704 --height 960 --seed 42 \
  --silence-mode idle --anchor pause-reset --reframe off \
  --max-piece-seconds 3 \
  --tripod --no-interview --storyboard off --no-judge \
  --llm-panel local --panel-judge ollama \
  --dry-run
```
