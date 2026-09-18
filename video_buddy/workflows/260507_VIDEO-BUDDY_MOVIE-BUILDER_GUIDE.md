# LTX 2.3 Movie Builder Workflow — Guide (Mickmumpitz, May 4)

Movie Builder stays on **LTX 2.3**. It is **not** one of the LTX 2.5 default-catalog
graphs (`ltx25_*`). Queue it with `python -m master_agent comfy run --mode template --template vb_movie_builder` — the director will not auto-route here. For LTX 2.5 T2V/I2V see [`ltx-2.5/README.md`](ltx-2.5/README.md).

Full movie-builder pipeline in ComfyUI: every shot is its own group, every group
pulls from one shared reference at the top of the workflow, and a final assembler
stitches the shots into a finished cut. Swap the reference image and voice sample,
hit Queue Prompt, and the same story renders with a different lead.

Runs on **LTX 2.3** for video + audio generation and **Flux 2 Klein** for the
startframe. Free version covers shot-by-shot movie creation; the advanced version
adds 360° environment generation for a controllable environment.

Workflow sections (color code): white = Input/Output/Settings/Model Loaders,
purple = Modular Groups, yellow = Important Notes, red = Asset Creation,
green = 360 Environment (advanced only).

## Setup: resolution, character & voice references

The Settings group at the top is the only place for global parameters.

- Image gen: 1152 px default. Raise if the GPU has headroom — image gen is cheap.
- Video gen: 720p default for iteration speed. Full HD possible but per-shot
  render time scales hard.
- fps: 24 default.

### Main character reference
Plug two images into the *Load Protagonist Images* group: a face close-up and a
full-body shot, ideally at a 3/4 angle (better side-view consistency than flat
front). The subgraph strips the background and produces three references:
FACE (face only), BODY (body only), FACEBODY (both blended). FACEBODY is what
most shots pull from; FACE/BODY are for shots needing only one (wide shot where
the face is too small, or tight close-up where the body confuses the model).
A second protagonist group exists for multi-character scenes.

### Voice reference
In *Audio Reference 01*, load a clean 5-second voice sample for LTX's
audio-reference LoRA. It drives voice cloning for any shot with the voice toggle
on. Trim nodes below the loader are pre-configured.

## Image generation per shot

Each shot group contains an image-gen subgraph and a video-gen subgraph, run in
sequence (still first, then video conditioned on the still).

- Reference inputs — up to 4. Connect FACEBODY to input 1; remaining three for
  additional characters, environment references, or assets.
- Prompt — natural-language scene description. Keep the protagonist description
  vague-but-anchored: write "the protagonist, in a field at golden hour", NOT
  "a woman with red hair and blue eyes wearing a leather jacket". The reference
  image carries identity; the vague prompt is what lets you swap the lead later
  without rewriting every shot. (You can get specific if the character is fixed.)
- Save Image — play here computes only the still; video stays unrendered.

Iterate a shot via prompt tweaks or the image sampler seed, then move to video.

## Video generation & voice acting

- Shot length — seconds; converted to frames by the workflow.
- Seeds — two seeds, one per sampler stage (LTX 2.3 distilled uses a two-stage
  path).
- Native vs upscale — native renders at the chosen video resolution; upscale
  uses LTX's spatial upscaler at 2x for higher final resolution.
- Voice reference toggle — on for dialogue shots, off for ambient/silent.

For dialogue shots, write the spoken line into the dialogue prompt field. The
voice reference clones the speaker; the dialogue text drives what they say.

**Voice mismatch fix:** if the voice comes out wrong, change the seed on stage 2
of the LTX sampler first; if that doesn't fix it, change stage 1 too. Stage-2
seed changes often fix voice/lip-sync without losing the visual.

## Shot Duplicator, color match, asset groups

- Shot Duplicator — select the source shot group, click *Add Shot*. The new
  group gets the same loaders, samplers, and reference connections, plus a
  Color Match node referencing Shot 01 — keeps the grade consistent across the
  whole movie. Manual duplicates do NOT wire this automatically.
- The shotlist (Shot Duplicator output) is reorderable; the final movie reads
  the shotlist order as the cut order.
- Asset groups — for props/vehicles/environment pieces: duplicate an Asset
  Creator group, prompt without character refs to design the asset (optionally
  feed a sketch via Load Image), then plug the output into a spare reference
  slot on any shot group.

## Advanced (Patreon): 360 environments, shot-reverse-shot

- 360 environment generation — prompt or load a reference; runs Flux 2 Klein
  with `flux-2-klein-9B-360-erp-outpaint-lora` to produce a seamless 360°
  equirectangular image. The Panorama Viewer node previews it in ComfyUI.
  The *Load Input Image* toggle switches between an existing reference and the
  outpaint-from-prompt pipeline.
- Shot-reverse-shot via Environment Crop (Olm DragCrop) — crop background
  regions per character framing from the same 360, load each crop into the
  matching shot group alongside character references. Lighting, geometry, and
  palette match across cuts.

**Required pack (not cosmetic):** `OlmDragCrop` sits on the first-shot
encode path (`ImageExists` → reference latent → `KSampler` → later I2V).
Install ComfyUI-Olm-DragCrop (or the pack that registers `OlmDragCrop`).
Buddy does **not** silent-bypass this class — missing pack fails Hands /
`validate_workflow` instead of 400 on `/prompt`. `PanoramaViewerNode` is
preview-only and soft-bypasses when unregistered.

## Final movie output / assembler

- ShotAssembler concatenates every shot in shotlist order into one video.
- Save individual shots toggle — keep on when editing in DaVinci/Premiere.
- Folder label — names each run (draft_01, draft_02, …); without it sequential
  runs overwrite each other.
- resolution_mode on ShotAssembler unifies shots rendered at different
  resolutions.

## Required models (filenames must match loader nodes)

Image generation (Flux 2 Klein, 9B default):
- `flux-2-klein-9b-fp8` → models/diffusion_models/
- `Flux2-Klein-9B-consistency-V2` (LoRA) → models/loras/
- `flux2-vae` → models/vae/
- `qwen_3_8b_fp8mixed` (text encoder) → models/text_encoders/

4B variant (smaller, more permissive license, faster, lower quality) — swap both:
- `flux-2-klein-4b-fp8` → models/diffusion_models/
- `qwen_3_4b_fp4_flux2` (text encoder) → models/text_encoders/
- advanced 360: also swap LoRA to `flux-2-klein-4B-360-erp-outpaint-lora_V1`

Video + audio (LTX 2.3):
- `ltx-2.3-22b-dev_transformer_only_fp8_scaled` → models/diffusion_models/
- `ltx-2.3-22b-distilled-lora-384-1.1` → models/loras/
- `ltx-2.3-id-lora-talkvid-3k` → models/loras/
- `ltx-2.3-spatial-upscaler-x2-1.1` → models/latent_upscale_models/
- `gemma_3_12B_it_fp8_scaled` (text encoder) → models/text_encoders/
- `ltx-2.3_text_projection_bf16` → models/text_encoders/
- `LTX23_video_vae_bf16` → models/vae/
- `LTX23_audio_vae_bf16` → models/vae/

Advanced only:
- `flux-2-klein-9B-360-erp-outpaint-lora_V1` → models/loras/

Low-VRAM alternative:
- `LTX-2.3-dev-Q4_K_S.gguf` → models/diffusion_models/ (swap the checkpoint
  loader for the GGUF loader; lower quality, much smaller VRAM footprint)

## Required custom nodes

Install via ComfyUI Manager → Install Missing Custom Nodes:
- ComfyUI-Mickmumpitz-Nodes — REQUIRED: ShotAssembler, ShotVideoOutput,
  ShotDuplicator, ShotOrder, ResolutionPicker, MickmumpitzLabel.
- ComfyUI-LTXVideo (Lightricks)
- ComfyUI-KJNodes
- RES4LYF — install manually from the Manager list if it errors via Install Missing.
- ComfyUI-VideoHelperSuite
- ComfyUI Easy Use — easy ifElse for the Load/Generate toggle (advanced).
- rgthree's ComfyUI Nodes
- Olm DragCrop — advanced only.
- ComfyUI-Panorama-Viewer — advanced only.

LTX 2.3 audio reference nodes need a recent ComfyUI build.

## Troubleshooting

- Custom nodes won't install via Manager → install RES4LYF manually.
- LTX audio reference nodes missing → update ComfyUI.
- Wrong voice on a shot → stage-2 seed first, then stage-1.
- Character drifts across shots → 3/4 body reference + short character
  description in each prompt.
- OOM on video gen → GGUF model versions; lower video resolution from 720p.
- Color grade jumps between shots → confirm Color Match references Shot 01 in
  every duplicated group (Shot Duplicator wires it; manual duplicates don't).
- Background changes when zooming in → reference the previous shot's image in
  addition to character refs.
- Final movie has mixed resolutions → set resolution_mode on ShotAssembler.
- Sequential runs overwrite → unique Folder label per run.
- 360 outpaint seams → increase padding in ImagePadForOutpaint; check LoRA
  filename matches.

## Tips & gotchas

- GGUF versions trade quality for much smaller VRAM — recommended for older GPUs.
- Use the top toggle to disable video generation and render ALL images first —
  avoids repeatedly loading/unloading Flux and LTX between shots.
- Reference inputs are agnostic: character in slot 1, asset in slot 2, two
  characters, character + background plate — the model treats them all as
  visual conditioning.
