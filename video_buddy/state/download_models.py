"""Download model files required by the Mickmumpitz workflows (2606*/2607*).

Uses huggingface_hub.hf_hub_download (resumable) + shutil.copy to the
project model dirs that ComfyUI sees via extra_model_paths.yaml.

Re-runnable: skips files already at the destination with the expected size.
"""
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

# (repo_id, repo_filename, dest_path_relative_to_project_root)
DOWNLOADS = [
    # --- text encoders ---
    ("Comfy-Org/flux2-klein-9B",
     "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
     "models/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
    ("Comfy-Org/Ideogram-4",
     "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
     "models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
    ("Comfy-Org/Krea-2",
     "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
     "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    # --- diffusion models ---
    ("Comfy-Org/Ideogram-4",
     "diffusion_models/ideogram4_fp8_scaled.safetensors",
     "models/diffusion_models/ideogram4_fp8_scaled.safetensors"),
    ("Comfy-Org/Ideogram-4",
     "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
     "models/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"),
    ("Comfy-Org/Krea-2",
     "diffusion_models/krea2_turbo_nvfp4.safetensors",
     "models/diffusion_models/krea2_turbo_nvfp4.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
     "models/diffusion_models/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
     "models/diffusion_models/wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
    # --- vae ---
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "split_files/vae/wan_2.1_vae.safetensors",
     "models/vae/wan_2.1_vae.safetensors"),
    # --- SeedVR2 (folder per custom node constants.py: models/SEEDVR2) ---
    ("numz/SeedVR2_comfyUI",
     "ema_vae_fp16.safetensors",
     "models/SEEDVR2/ema_vae_fp16.safetensors"),
    ("cmeka/SeedVR2-GGUF",
     "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
     "models/SEEDVR2/seedvr2_ema_7b_sharp-Q4_K_M.gguf"),
    # --- face detector (Impact Pack convention: models/ultralytics/bbox) ---
    ("Bingsu/adetailer",
     "face_yolov8m.pt",
     "models/ultralytics/bbox/face_yolov8m.pt"),
    # --- public LoRAs referenced by the WAN 2.2 workflow ---
    ("Kijai/WanVideo_comfy",
     "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
     "models/loras/wan/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"),
    ("Instara/instareal-wan-2.2",
     "Instareal_high.safetensors",
     "models/loras/wan/Instareal_high.safetensors"),
    ("Instara/instareal-wan-2.2",
     "Instareal_low.safetensors",
     "models/loras/wan/Instareal_low.safetensors"),
    ("kennydaglish/Wan22Loras",
     "stock_photography_wan22_LOW_v1.safetensors",
     "models/loras/wan/stock_photography_wan22_LOW_v1.safetensors"),
    ("untitled700/lenovoUltraRealWan",
     "Lenovo.safetensors",
     "models/loras/wan/Lenovo.safetensors"),
    # --- Qwen Image Edit 360 workflow (251018) ---
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "models/text_encoders/qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/vae/qwen_image_vae.safetensors",
     "models/vae/qwen/qwen_image_vae.safetensors"),
    # GGUF: unet folder aliases to models/diffusion_models (extra_model_paths.yaml)
    ("QuantStack/Qwen-Image-Edit-2509-GGUF",
     "Qwen-Image-Edit-2509-Q5_0.gguf",
     "models/diffusion_models/gguf/Qwen-Image-Edit-2509-Q5_0.gguf"),
    ("unsloth/Qwen-Image-Edit-2511-GGUF",
     "qwen-image-edit-2511-Q5_0.gguf",
     "models/diffusion_models/gguf/qwen-image-edit-2511-Q5_0.gguf"),
    ("lightx2v/Qwen-Image-Lightning",
     "Qwen-Image-Lightning-4steps-V2.0.safetensors",
     "models/loras/qwen/Qwen-Image-Lightning-4steps-V2.0.safetensors"),
    ("lightx2v/Qwen-Image-Edit-2511-Lightning",
     "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
     "models/loras/qwen/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"),
    ("mickmumpitz/QWEN-EDIT_360",
     "251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors",
     "models/loras/qwen/251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors"),
    # --- Wan VACE (AI-VFX workflows) ---
    # note: no public true-fp32 file exists; e4m3fn is the canonical fp8 build
    ("Inner-Reflections/VACE_Skyreels_V3_R2V_Merge",
     "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors",
     "models/diffusion_models/wan/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors"),
    ("mickmumpitz/VACE_Skyreels_V3_R2V_Merge-GGUF",
     "wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf",
     "models/diffusion_models/gguf/wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1-Q4_K_M.gguf"),
    ("vrgamedevgirl84/Wan14BT2VFusioniX",
     "FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
     "models/loras/wan/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"),
    # --- LTX 2.3 lip-sync LoRAs ---
    ("Kijai/LTX2.3_comfy",
     "loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
     "models/loras/ltx/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"),
    ("Lightricks/LTX-2-19b-IC-LoRA-Union-Control",
     "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
     "models/loras/ltx/ltx-2-19b-ic-lora-union-control-ref0.5.safetensors"),
    ("Lightricks/LTX-2-19b-IC-LoRA-Detailer",
     "ltx-2-19b-ic-lora-detailer.safetensors",
     "models/loras/ltx-2-19b-ic-lora-detailer.safetensors"),
    ("Lightricks/LTX-2.3",
     "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
     "models/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
    # --- CCC 4.1 Krea2-Edit ---
    ("Comfy-Org/Krea-2",
     "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
     "models/diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    ("Comfy-Org/Krea-2",
     "text_encoders/qwen3vl_4b_bf16.safetensors",
     "models/text_encoders/qwen3vl_4b_bf16.safetensors"),
    ("conradlocke/krea2-identity-edit",
     "krea2_identity_edit_v1_2.safetensors",
     "models/loras/krea/krea2_identity_edit_v1_2.safetensors"),
    # --- Z-Image Turbo CN (AI-RENDERER) ---
    ("Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
     "models/diffusion_models/z_image/z_image_turbo_bf16.safetensors"),
    ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
     "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors",
     "models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors"),
    # --- BATCH4: Mickmumpitz MOVIE-BUILDER workflow (260507) ---
    ("nomadoor/flux-2-klein-9B-360-erp-outpaint-lora",
     "flux-2-klein-9B-360-erp-outpaint-lora_V1.safetensors",
     "models/loras/flux-2-klein-9B-360-erp-outpaint-lora_V1.safetensors"),
    ("QuantStack/LTX-2.3-GGUF",
     "LTX-2.3-dev/LTX-2.3-dev-Q4_K_S.gguf",
     "models/diffusion_models/gguf/LTX-2.3-dev-Q4_K_S.gguf"),
    ("Comfy-Org/ltx-2.3",
     "split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors",
     "models/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"),
    ("Comfy-Org/ltx-2",
     "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
     "models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"),
    ("Kijai/LTX2.3_comfy",
     "vae/LTX23_video_vae_bf16.safetensors",
     "models/vae/LTX23_video_vae_bf16.safetensors"),
    ("Lightricks/LTX-2.3",
     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
     "models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_local_weight(fname, dest):
    """Reuse a file already on disk (dest, Comfy trees, HF cache, aliases)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from master_agent.models.weights import find_weight_file

        found = find_weight_file(os.path.basename(fname.replace("\\", "/")))
        if found is not None and found.is_file() and found.stat().st_size > 0:
            return str(found)
    except Exception:
        return None
    return None


def fetch(repo, fname, dest_rel):
    dest = os.path.join(ROOT, dest_rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        existing = _find_local_weight(fname, dest)
        if existing:
            log(
                f"SKIP download of {os.path.basename(dest)} — local file found at "
                f"{existing} (not re-downloading)"
            )
            return dest_rel, "SKIP"
        if os.path.islink(dest):
            os.remove(dest)  # dangling symlink left by an earlier failed move
        cached = hf_hub_download(repo_id=repo, filename=fname)
        # HF cache entries are symlinks into blobs/; move the REAL blob so the
        # cache doesn't double disk usage (shutil.move would recreate a broken
        # relative symlink at the destination instead).
        blob = os.path.realpath(cached)
        if blob != cached and os.path.exists(blob):
            try:
                os.replace(blob, dest)
            except OSError:
                # cross-drive cache (e.g. HF cache symlinked to another disk)
                shutil.copyfile(blob, dest)
                os.remove(blob)
            try:
                os.remove(cached)  # drop the now-dangling cache symlink
            except OSError:
                pass
        else:
            shutil.move(cached, dest)
        log(f"DONE  {dest_rel} ({os.path.getsize(dest)} bytes) <- {repo}/{fname}")
        return dest_rel, "DONE"
    except GatedRepoError:
        log(f"GATED {dest_rel} <- {repo}/{fname} (needs HF token + license accept)")
        return dest_rel, "GATED"
    except EntryNotFoundError:
        log(f"MISS  {dest_rel} <- {repo}/{fname} (file not in repo)")
        return dest_rel, "MISS"
    except Exception as e:
        log(f"FAIL  {dest_rel} <- {repo}/{fname}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return dest_rel, "FAIL"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    log(f"starting {len(DOWNLOADS)} downloads with {workers} workers")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, r, f, d): d for r, f, d in DOWNLOADS}
        for fut in as_completed(futs):
            dest, status = fut.result()
            results[dest] = status
    log("==== SUMMARY ====")
    for dest, status in sorted(results.items()):
        log(f"{status:5s} {dest}")


if __name__ == "__main__":
    main()
