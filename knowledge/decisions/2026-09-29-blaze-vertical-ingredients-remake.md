# Blaze vertical remakes stay on Ingredients IC-LoRA

- **status:** recorded
- **machine:** both
- **date:** 2026-09-29
- **tried:** Restyle the rough mp4 with `ltx25_v2v_ic_lora`. Plain `ltx25_t2v_i2v` with no stills. The shipped `ltx25_msr` Ingredients graph (`LTXAddVideoICLoRAGuide` + `LTXICLoRALoaderModelOnly`). A pic1–pic4 node `ComfyUILTX25MSRMultiReferenceGuide` (named in older merge notes).
- **chosen:** `ltx25_msr`, two chained guide stills (Scott, then Blaze), safe canvas 448×800×97. Source 720×1280×241 is opt-in and unmeasured. Recipe ids `blaze-concert` and `blaze-pier`.
- **why:** Cached object_info has the Ingredients guide and loader and does not have the multi-ref guide class. V2V would start from the morphing frames. T2V without stills does not lock Scott. The landscape 30s intro is a different job: this recipe caps at 241 frames and uses its own filename prefixes.
- **revisit when:** a tower smoke records `sec/step` plus ffprobe for 448×800×97, or a later object_info actually lists `ComfyUILTX25MSRMultiReferenceGuide`.
- **graph:** video_buddy/workflows/ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json
- **playbook:** knowledge/workflows/2026-09-29-blaze-vertical-remake-ltx25-msr.md
