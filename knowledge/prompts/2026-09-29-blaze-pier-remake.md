# blaze pier remake

- **status:** recorded
- **machine:** both
- **goal:** Story-grade vertical remake of the rough Clearwater pier clip with Scott and Blaze locked
- **model:** LTX 2.5 Ingredients IC-LoRA, `ltx25_msr`
- **key parameters:** safe default 448×800, 97 frames, 24 fps, cfg 1, guide strength 1. Rough source was 720×1280×241. That size is legal `8n+1` and is not the 16GB default.
- **workflow:** video_buddy/workflows/ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json
- **prompt:** see Positive below. The CLI loads this section. A `--prompt` override replaces it for one run.
- **result note:** Brief locked from CoS review of the rough vertical (flat Blaze and turtle cutouts, caption-style words). No tower render is recorded in this change.

## Positive

Vertical 9:16 Clearwater Beach pier in bright daylight, story-grade live action. Scott is in the foreground, locked to the Scott reference still: red baseball cap worn backwards, dark sunglasses, black t-shirt with a blue diamond on the chest, one stable face, natural skin, two hands. A woman in a dark bikini stands in the foreground with him, one body, stable limbs. Blaze is locked to the Blaze reference still and stands on the pier boards as a solid green leafy mascot with a face, stubby limbs, feet, weight, and a contact shadow. A small cartoon turtle is solid on the pier, not a flat cutout. Gulf water, pier pilings, sunny Florida light, continuous motion. Upbeat energy only. No burned-in words, no captions, no subtitles.

## Negative

morphing, identity drift, face swap, melting face, extra fingers, extra limbs, fused hands, sticker, flat 2d mascot, paper cutout, pasted overlay, cartoon sticker on the pier, readable text, subtitles, captions, lower third, burned-in words, misspelled text, letters, watermark, logo, flicker, unlocked face
