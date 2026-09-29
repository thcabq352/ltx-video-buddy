# LTX 2.3 outpaint emitted 1 frame

- **status:** recorded
- **machine:** both
- **priority:** check-before-retry
- **symptom:** LTX 2.3 outpaint emitted 1 frame instead of the requested 25.
  Verified 2026-09-29.
- **bad approach:** The pad mask was a still single mask, and SaveVideo / blend
  used `trim_to_shortest`, so the output collapsed to 1 frame. Do not retry
  outpaint with a still mask plus `trim_to_shortest` on either machine. The
  same 1-frame class hit the first #41 and #43 smokes before the fixes below.
- **fix:** PR #44, commit `d19b414`, branch
  `cursor/fix-ltx23-inoutpaint-smoke-fbb2`. Repeat the outpaint pad mask to
  the latent length via `VHS_DuplicateMasks` (nodes 82/83) into dilate, and
  set blend `trim_to_shortest` to false. Related earlier fixes on the same
  PR: `resize_type.width` / `resize_type.height` / `resize_type.crop` on
  `ResizeImageMaskNode`, trim the source to `--frames`, and SaveVideo
  `format.codec`. Re-smoke A3 after the fix: PASS, 25 frames, 1.0417 s at
  24 fps, 448×768, peak VRAM ~15516 MiB, model
  `ltx-2.3-22b-dev-fp8.safetensors`, seed 42. Continuity good. A mild
  bottom-band edge remained. No flicker.
- **applies when:** LTX 2.3 outpaint at 25 frames, and the same still-mask
  plus `trim_to_shortest` setup that failed the first #41 and #43 smokes.
