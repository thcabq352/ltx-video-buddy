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
| `--lipdub-overlap` | `8` (`LIPDUB_OVERLAP_FRAMES`) | Frames of overlap trimmed at a speech-to-speech seam. |
| `--words` | none | JSON word timestamps (`[{w,s,e}]` or `{"words":[...]}`). Splits prefer pauses and never cut a word. A word longer than the cap is kept whole. |
| `--tripod` | off | Locked-off talking head. Prompt, negative, and stage-1 `LTXVImgToVideoInplace` strength `0.7 → 0.85`. Stage 2 stays `1.0`. The graph has no camera-motion input. |
| `--dry-run` | | Prints the segment plan and validates each Comfy piece. Nothing is queued. |

H3 (`h3_r2v`) is unchanged: one clip, capped at 12s.

## Continuity and silence

- Speech after speech is conditioned on the previous slice's last kept frame. The overlap frames are rendered, then dropped, so the timeline does not duplicate frames and the muxed audio stays the original file.
- Speech after a pause is conditioned on the pause bridge's last frame, not the original still.
- Leading silence (≥ 250ms) is a hold of the source still (the rest mouth).
- An internal or trailing pause is a 9-frame a2v bridge from the previous frame ("close the mouth"), using the real pause audio, then a hold of that bridge's last frame for the rest of the pause.
- Frame grid is `round(duration * fps)`. 12.0s at 24 fps is 288 frames. Every render length is `8n+1`. Pieces tile `[0, timeline)`.

Prompt-only "mouth closed during pauses" did not hold on the 12s tower pass, so silence is structural (plate or bridge), not a prompt sentence.

`--tripod` is what keeps the head and coat from drifting so a still-hold or bridge does not pop. It is opt-in. With the flag off, a short clip's prompt and strength are unchanged.

## Provenance

`outputs/<run>/shot-1.buddy.json` is still `buddy.clip.provenance/v1`. `params.lipdub` records `split_points_s`, `continuity`, `silence_handling`, `tripod`, `full_audio_mux`, and one record per segment (seed, attempt, time range, source frame, place frames). There is no second schema.

The Rust `ClipProvenance` struct in `your-video-buddy` does not list `params.lipdub` yet. If that struct denies unknown fields, it needs `lipdub: Option<...>` the same way it already allows `voice_sample`. This repo does not fork the schema id.

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
  --width 352 --height 480 --seed 7 \
  --tripod --no-interview --storyboard off --no-judge \
  --llm-panel local --panel-judge ollama \
  --dry-run
```

Drop `--dry-run` to render. `--words` should be the faster-whisper timestamps for that wav (pauses near 0–0.9s, 5.7–6.25s, and 10.6–11.15s). Without `--words`, Buddy still splits on energy pauses or, if those cannot be read, on the max length.
