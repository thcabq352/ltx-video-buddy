# blaze concert remake

- **status:** recorded
- **machine:** both
- **goal:** Story-grade vertical remake of the rough stadium concert clip with Scott and Blaze locked
- **model:** LTX 2.5 Ingredients IC-LoRA, `ltx25_msr`
- **key parameters:** safe default 448×800, 97 frames, 24 fps, cfg 1, guide strength 1. Rough source was 720×1280×241. That size is legal `8n+1` and is not the 16GB default.
- **workflow:** video_buddy/workflows/ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json
- **prompt:** see Positive below. The CLI loads this section. A `--prompt` override replaces it for one run.
- **result note:** Brief locked from CoS review of the rough vertical (face and limb morphs, flat Blaze paste, jumbotron text "MHETLIICCA"). No tower render is recorded in this change.

## Positive

Vertical 9:16 night stadium concert, story-grade live action. Scott is locked to the Scott reference still: red baseball cap worn backwards, dark sunglasses, black t-shirt with a blue diamond on the chest, one stable face, natural skin, two hands with five fingers. A woman in a dark bikini stands next to him, one body, stable limbs, not a morph. Blaze is locked to the Blaze reference still and stands in the crowd as a solid green leafy mascot with a face, stubby limbs, feet, weight, and a contact shadow on the rock bed. A small cartoon turtle is also solid and sitting on the rocks, in the world, not a cutout. Stadium lights, a cheering crowd, continuous motion. The jumbotron is only abstract color and light, with no letters, no words, and no logos.

## Negative

morphing, identity drift, face swap, melting face, extra fingers, extra limbs, fused hands, duplicated Scott, sticker, flat 2d mascot, paper cutout, pasted overlay, cartoon sticker, readable text, subtitles, captions, jumbotron text, misspelled text, letters, watermark, logo, metallica, mhetliicca, skull on the screen, geometry soup, flicker, unlocked face
