# LoRA training (scenes & styles)

Train your own **scene LoRAs** (a location, set, or lighting look) and
**style LoRAs** (lego, claymation, pixel-art, …) for the workflows in this
project, using [ostris/ai-toolkit](../ai-toolkit) on the local 16GB GPU.

## Layout

```
training/
  configs/
    train_lora_ltx23_16gb.yaml   # LTX-2.3 22B  (LIP-SYNC / master_agent pipeline)
    train_lora_wan22_16gb.yaml   # Wan 2.2 14B  (WAN-2.2-VID pipeline)
  datasets/
    scenes/<scene_name>/         # stills and/or short clips of ONE scene
    styles/<style_name>/         # 20-100 images of ONE style (lego, clay, ...)
```

## 1. Prepare a dataset

- Put images (`.jpg/.png/.webp`) and/or short clips (`.mp4`) in one folder.
- Next to every file, a same-named `.txt` caption: `shot01.png` + `shot01.txt`.
- Start every caption with your trigger token (set as `trigger_word` in the
  config — use a nonsense string like `l3g0styl3`, not a real word).
- Captions describe what is IN the frame, not the style/scene itself — the
  LoRA learns the constant part. Example: `l3g0styl3 a bear walking through
  a forest, wide shot`.
- Scene LoRA: 10-40 varied angles/lighting of the same scene work well.
  Style LoRA: 20-100 images in the target style.
- Good source material: Blender viewport renders (the bundled Blender 5.3
  alpha in this project), frames from `outputs/`, or AI-generated sets from
  the Qwen-Edit / Krea workflows.

## 2. Train

Edit a copy of a config (change `name`, `trigger_word`, `folder_path`), then:

```bash
cd ai-toolkit
./venv/Scripts/python.exe run.py ../training/configs/train_lora_ltx23_16gb.yaml
```

- Checkpoints save every 250 steps to `ai-toolkit/output/<name>/`.
- Samples render alongside so you can watch the LoRA converge.
- 16GB notes: keep `num_frames: 1` for image training; video LoRA training
  works but keep clips short (~25 frames) and resolutions low.
- First Wan 2.2 run downloads ~28GB of base weights from HuggingFace.

## 3. Use the LoRA

Copy the final `.safetensors` into the ComfyUI-visible model tree:

```bash
# LTX-2.3 LoRAs
cp ai-toolkit/output/<name>/<name>.safetensors models/loras/ltx/
# Wan LoRAs
cp ai-toolkit/output/<name>/<name>.safetensors models/loras/wan/
```

Then add/select it in a `LoraLoader` node (strength 0.6-1.0) in the LTX or
Wan workflow, and include your trigger token in the prompt.
Restart ComfyUI (or its model list) so the new file appears in the dropdown.
