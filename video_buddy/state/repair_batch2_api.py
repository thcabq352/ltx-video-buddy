"""One-shot repair of the second batch of converted Mickmumpitz *_api.json files.

Same class of converter bugs as batch 1 (state/repair_vb_api.py), plus:
- dangling subgraph-output links (LTX 2.3 subgraph output slot 2 is unconnected
  in the author's own file; the converter still emitted links to the dropped
  instance id) -> drop those inputs
- private assets referenced by LoadImage/LoadVideo/LoadAudio -> repoint to
  bundled example assets (copied into ComfyUI/input/) or existing placeholders
- 'wan-14B_vace_..._fp32_v1.safetensors' does not exist publicly (the mirror
  file with that name is a byte-identical rename of fp16) -> use the e4m3fn
  build the v1-3 workflow already uses
- gemma fp8_e4m3fn text encoder not downloaded; the project already carries
  gemma_3_12B_it_fp4_mixed.safetensors (also linked in the workflow's own notes)

Idempotent: safe to re-run.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- explicit per-file input overrides -------------------------------------
REPOINT = {
    "workflows/251018_VIDEO-BUDDY_QWEN-IMAGE-EDIT-360_1-1_api.json": {
        "213": {"image": "ComfyUI_01828_.png"},
    },
    "workflows/260330_AI-VFX-STARTIMAGE_1-0_api.json": {
        "21": {"video": "warehouse_src_30fps.mp4"},
        "22": {"image": "ComfyUI_01828_.png"},
        # author's loader values lacked our subfolder prefixes
        "24": {"clip_name": "qwen\\qwen_2.5_vl_7b_fp8_scaled.safetensors"},
        "30": {"vae_name": "qwen\\qwen_image_vae.safetensors"},
        "29": {"gguf_name": "gguf\\qwen-image-edit-2511-Q5_0.gguf"},
    },
    "workflows/260330_VIDEO-BUDDY_AI-VFX_1-0_ADV_api.json": {
        "282": {"image": "ComfyUI_01828_.png"},
        "265": {"video": "warehouse_src_30fps.mp4"},
        "3380": {"video": "warehouse_src_30fps.mp4"},
        # no public fp32 build of the VACE R2V merge exists
        "3326": {"unet_name": "wan\\wan-14B_vace_skyreels_v3_R2V_e4m3fn_v1.safetensors"},
        "3327": {"lora_name": "wan\\Wan2.1_T2V_14B_FusionX_LoRA.safetensors"},
    },
    "workflows/260330_VIDEO-BUDDY_AI-VFX_PREPROCESS_1-0_api.json": {
        "265": {"video": "warehouse_src_30fps.mp4"},
        # CreateShapeImageOnPath: converter dropped frame_width/height widgets,
        # shifting everything after them by two slots
        "3420": {
            "shape_height": 32,
            "shape_color": "black",
            "bg_color": "black",
            "blur_radius": 0.5,
            "intensity": 1,
            "size_multiplier": 1,
            "trailing": 2,
            "border_color": "white",
        },
        # SAM3Segment was unknown at convert time -> widgets never mapped
        "3537": {
            "prompt": "person",
            "output_mode": "Merged",
            "confidence_threshold": 0.35,
            "max_segments": 0,
            "segment_pick": 0,
            "mask_blur": 0,
            "mask_offset": 0,
            "device": "Auto",
            "invert_output": False,
            "unload_model": False,
            "background": "Color",
            "background_color": "#808080",
        },
    },
    "workflows/260608_VIDEO-BUDDY_AI-VFX_1-3_ADV_api.json": {
        "3349": {"image": "ComfyUI_01828_.png"},
        "282": {"image": "ComfyUI_01828_.png"},
        "3380": {"video": "warehouse_src_30fps.mp4"},
        "265": {"video": "warehouse_src_30fps.mp4"},
        "3327": {"lora_name": "wan\\Wan2.1_T2V_14B_FusionX_LoRA.safetensors"},
    },
    "workflows/260603_LTX2-3_3D-RENDERING_LIP-SYNC_v08_api.json": {
        "6108": {"image": "BusinessWoman_START-IMAGE.webp"},
        "6102": {"image": "BusinessWoman_START-IMAGE.webp"},
        "6139": {"image": "BusinessWoman_START-IMAGE.webp"},
        "6094": {"video": "BusinessWoman_CLAY.mp4"},
        "6059": {"video": "BusinessWoman_MASK.mp4"},
        "5893": {"video": "BusinessWoman_DEPTH.mp4"},
        "5883": {"audio": "BusinessWoman_AUDIO.mp3"},
    },
    "workflows/260729_VIDEO-BUDDY_CCC_4-1_KREA2-EDIT_BETA_PUBLISH_api.json": {
        "535": {"image": "ComfyUI_01828_.png"},
        "6292": {"image": "ComfyUI_01828_.png"},
        # ImageCollectUnpack — old order (source, fit, target_size);
        # 'Dataset' choice removed in current pack version
        "1560": {"source": "first", "target_size": "first", "fit": "stretch"},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/030_BearMinimum_RightThisWay/030_BearMinimum_RightThisWay_api.json": {
        "6108": {"image": "ComfyUI_02005_.png"},
        "6102": {"image": "ComfyUI_02005_.png"},
        # SetNode 6250 ('MouthMask') is frontend-only; the converter dropped it
        # without rewiring -> point mask at its source (ComfySwitchNode 6244)
        "6282": {"mask": ["6244", 0]},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/BusinessWoman/BusinessWoman_Workflow_api.json": {
        "6102": {"image": "BusinessWoman_START-IMAGE.webp"},
        "6139": {"image": "BusinessWoman_START-IMAGE.webp"},
    },
    # --- batch 3 (AI-RENDERER / Z-Image / RTX-SR) ---
    "workflows/AI-RENDERING-EXAMPLE FILES/260303_VIDEO-BUDDY_Z-IMAGE_TURBO_CN_1-1_api.json": {
        "58": {"image": "ComfyUI_01828_.png"},
        "74": {"video": "dance_line_v03_0001-0250.mkv"},
        "88": {"vae_name": "ae.safetensors"},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/260402_VIDEO-BUDDY_AI-RENDERER_SMPL_2-1_api.json": {
        "265": {"video": "dance_depth_v03_0001-0250.mkv"},
        "282": {"image": "ComfyUI_01828_.png"},
        "3345": {"video": "dance_line_v03_0001-0250.mkv"},
        # SetNode 3002 ('control') is frontend-only -> rewire to the
        # 'Blend Outputs' subgraph's internal output node
        "3001": {"images": ["3272:3278", 0]},
        "3327": {"lora_name": "wan\\Wan2.1_T2V_14B_FusionX_LoRA.safetensors"},
        # WanVideoVACEStartToEndFrame unknown at convert time -> widget shift
        "3228:3261": {"end_image": None, "inpaint_mask": None,
                      "empty_frame_level": 0.5, "start_index": 0, "end_index": -1},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/260402_VIDEO-BUDDY_AI-RENDERER_ADV_2-1_api.json": {
        "265": {"video": "warehouse_src_30fps.mp4"},
        "282": {"image": "ComfyUI_01828_.png"},
        "3345": {"video": "dance_depth_v03_0001-0250.mkv"},
        "3327": {"lora_name": "wan\\Wan2.1_T2V_14B_FusionX_LoRA.safetensors"},
        "3228:3261": {"end_image": None, "inpaint_mask": None,
                      "empty_frame_level": 0.5, "start_index": 0, "end_index": -1},
        "3228:3299": {"model": "gimmvfi_r_arb_lpips_fp32.safetensors",
                      "precision": "fp32", "torch_compile": False},
        "3228:3301": {"ds_factor": 1, "interpolation_factor": 2,
                      "seed": 653337867442944},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/260225_VIDEO-BUDDY_AI-RENDERER_ADV_2-0_api.json": {
        "265": {"video": "warehouse_src_30fps.mp4"},
        "282": {"image": "ComfyUI_01828_.png"},
        "3345": {"video": "dance_depth_v03_0001-0250.mkv"},
        "3327": {"lora_name": "wan\\Wan2.1_T2V_14B_FusionX_LoRA.safetensors"},
        "3228:3261": {"end_image": None, "inpaint_mask": None,
                      "empty_frame_level": 0.5, "start_index": 0, "end_index": -1},
        "3228:3299": {"model": "gimmvfi_r_arb_lpips_fp32.safetensors",
                      "precision": "fp32", "torch_compile": False},
        "3228:3301": {"ds_factor": 1, "interpolation_factor": 2,
                      "seed": 653337867442944},
    },
    "workflows/AI-RENDERING-EXAMPLE FILES/260330_VIDEO-BUDDY_NVIDIA-RTX-SUPER-RESOLUTION_1-0_api.json": {
        "12": {"video": "warehouse_src_30fps.mp4"},
        "26": {"resize_type": "target dimensions", "quality": "ULTRA"},
    },
}

LTX_APIS = [
    "workflows/260603_LTX2-3_3D-RENDERING_LIP-SYNC_v08_api.json",
    "workflows/AI-RENDERING-EXAMPLE FILES/030_BearMinimum_RightThisWay/030_BearMinimum_RightThisWay_api.json",
    "workflows/AI-RENDERING-EXAMPLE FILES/050_BearMinimum_Bar/050_BearMinimum_Bar_workflow_api.json",
    "workflows/AI-RENDERING-EXAMPLE FILES/BusinessWoman/BusinessWoman_Workflow_api.json",
]

GEMMA_SWAP = ("gemma_3_12B_it_fp8_e4m3fn.safetensors",
              "gemma_3_12B_it_fp4_mixed.safetensors")

CCC41 = "workflows/260729_VIDEO-BUDDY_CCC_4-1_KREA2-EDIT_BETA_PUBLISH_api.json"
CCC41_UI = "workflows/260729_VIDEO-BUDDY_CCC_4-1_KREA2-EDIT_BETA_PUBLISH.json"


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def save(rel, data):
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def drop_dead_links(wf):
    """Remove inputs that link to a node id not present in the workflow."""
    n = 0
    for node in wf.values():
        for key in list(node.get("inputs", {})):
            val = node["inputs"][key]
            if isinstance(val, list) and val and str(val[0]) not in wf:
                del node["inputs"][key]
                n += 1
    return n


def swap_gemma(wf):
    n = 0
    for node in wf.values():
        for key, val in node.get("inputs", {}).items():
            if val == GEMMA_SWAP[0]:
                node["inputs"][key] = GEMMA_SWAP[1]
                n += 1
    return n


def main():
    for rel, fixes in REPOINT.items():
        wf = load(rel)
        changed = 0
        for nid, vals in fixes.items():
            if nid in wf:
                for key, val in vals.items():
                    if val is None:
                        wf[nid]["inputs"].pop(key, None)  # delete bogus input
                    else:
                        wf[nid]["inputs"][key] = val
                changed += 1
            else:
                print(f"WARN {rel}: node {nid} not found")
        save(rel, wf)
        print(f"{rel}: {changed} node(s) repointed")

    for rel in LTX_APIS:
        wf = load(rel)
        dead = drop_dead_links(wf)
        gemma = swap_gemma(wf)
        save(rel, wf)
        print(f"{rel}: {dead} dead link(s) dropped, {gemma} gemma swap(s)")

    # JoinStringMulti in CCC 4.1: converter shifted [inputcount, delimiter,
    # return_list] -> restore delimiter/return_list from the UI widgets_values.
    ui = load(CCC41_UI)
    ui_widgets = {}
    stacks = [ui.get("nodes", [])]
    stacks += [s.get("nodes", []) for s in ui.get("definitions", {}).get("subgraphs", [])]
    for nodes in stacks:
        for x in nodes:
            if x.get("type") == "JoinStringMulti":
                ui_widgets[str(x["id"])] = x.get("widgets_values") or []
    wf = load(CCC41)
    n = 0
    for nid, node in wf.items():
        if node.get("class_type") != "JoinStringMulti":
            continue
        inp = node["inputs"]
        if isinstance(inp.get("delimiter"), str) and inp.get("return_list") is not None:
            continue
        wv = ui_widgets.get(nid.split(":")[-1], [])
        inp["delimiter"] = wv[1] if len(wv) > 1 and isinstance(wv[1], str) else " "
        inp["return_list"] = wv[2] if len(wv) > 2 and isinstance(wv[2], bool) else False
        n += 1
    save(CCC41, wf)
    print(f"{CCC41}: {n} JoinStringMulti node(s) repaired")

    # Krea2 subgraph instances: the converter (run before the krea2 packs were
    # installed) botched the Krea2Edit nodes — GroundedEncode.image/prompt got
    # pointed at ResolutionSelector INT outputs, image_b/source_latent_b got
    # raw widget ints, and the internal VAEEncode feeding
    # ModelPatch.source_latent was dropped entirely.
    ui_all = load(CCC41_UI)
    ui_links = {l[0]: l for l in ui_all["links"]}
    krea_subs = {s["id"] for s in ui_all["definitions"]["subgraphs"]
                 if any(x["type"].startswith("Krea2Edit") for x in s["nodes"])}
    inst_slots = {}  # instance id -> {slot name: (origin node, origin slot)}
    for n in ui_all["nodes"]:
        if n["type"] in krea_subs:
            inst_slots[str(n["id"])] = {
                i["name"]: (l[1], l[2])
                for i in n.get("inputs", [])
                if (l := ui_links.get(i.get("link")))
            }
    wf = load(CCC41)

    def is_valid_image_link(val):
        return (isinstance(val, list) and str(val[0]) in wf
                and wf[str(val[0])].get("class_type") != "ResolutionSelector")

    instances = {}
    for nid, node in wf.items():
        if ":" in nid and node.get("class_type") in (
            "Krea2EditGroundedEncode", "Krea2EditModelPatch"):
            instances.setdefault(nid.split(":")[0], []).append(nid)
    fixed = 0
    for inst, nids in instances.items():
        ges = [i for i in nids if wf[i]["class_type"] == "Krea2EditGroundedEncode"]
        mps = [i for i in nids if wf[i]["class_type"] == "Krea2EditModelPatch"]
        # canonical slot-0 (pixels) resolution: the converter got
        # ModelPatch.source_image right even where GroundedEncode.image broke
        good_image = None
        for mp in mps:
            if is_valid_image_link(wf[mp]["inputs"].get("source_image")):
                good_image = list(wf[mp]["inputs"]["source_image"])
                break
        if good_image is None:
            for ge in ges:
                if is_valid_image_link(wf[ge]["inputs"].get("image")):
                    good_image = list(wf[ge]["inputs"]["image"])
                    break
        text_slot = inst_slots.get(inst, {}).get("text")
        for ge in ges:
            inp = wf[ge]["inputs"]
            if good_image and not is_valid_image_link(inp.get("image")):
                inp["image"] = list(good_image)
                fixed += 1
            p = inp.get("prompt")
            if isinstance(p, list) and wf.get(str(p[0]), {}).get("class_type") == "ResolutionSelector":
                if text_slot and str(text_slot[0]) in wf:
                    inp["prompt"] = [str(text_slot[0]), text_slot[1]]
                else:
                    inp["prompt"] = ""
                fixed += 1
            elif "prompt" not in inp:
                inp["prompt"] = ""  # UI widget value was empty
                fixed += 1
            if "image_b" in inp and not is_valid_image_link(inp["image_b"]):
                del inp["image_b"]  # slot was unlinked; converter left garbage
                fixed += 1
        for mp in mps:
            inp = wf[mp]["inputs"]
            inp.pop("ref_boost_mask", None)  # MASK input that received a bogus int
            inp.setdefault("ref_boost_a", 1.0)
            inp.setdefault("fit_mode", "fit")
            for bad in ("source_latent_b", "source_image_b"):
                if bad in inp and not is_valid_image_link(inp[bad]):
                    del inp[bad]
                    fixed += 1
            if "source_latent" not in inp and is_valid_image_link(inp.get("source_image")):
                ve = f"{inst}:9016"  # fresh id, no clash with UI ids
                if ve not in wf:
                    wf[ve] = {"inputs": {"pixels": list(inp["source_image"]),
                                         "vae": list(inp["vae"])},
                              "class_type": "VAEEncode",
                              "_meta": {"title": "VAEEncode (restored: source_latent)"}}
                inp["source_latent"] = [ve, 0]
                fixed += 1
    # ImpactInversedSwitch now has a single output; the UI linked slot 1
    for nid, node in wf.items():
        for key, val in node.get("inputs", {}).items():
            if (isinstance(val, list) and str(val[0]) in wf
                    and wf[str(val[0])].get("class_type") == "ImpactInversedSwitch"
                    and val[1] != 0):
                node["inputs"][key] = [val[0], 0]
                fixed += 1
    save(CCC41, wf)
    print(f"{CCC41}: {fixed} Krea2 edit node(s) repaired")


if __name__ == "__main__":
    main()
