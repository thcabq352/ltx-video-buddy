"""Probe live/cached /object_info against Buddy's wired surfaces.

Discovery here is *read-only*: Buddy does not auto-queue a node just because
it appears in object_info. The catalog is the gap matrix for tower packs
vs director / patcher / template / CPU-only paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from master_agent.comfy.cli_run import unwrap_workflow
from master_agent.config import OBJECT_INFO_CACHE, WORKFLOW_FILES, WORKFLOWS_DIR

# How far a capability can currently be driven from Buddy code.
# director  = choose_variant / run pipeline will pick it
# patcher   = load_and_patch_workflow has a field map / heuristic
# template  = shipped API graph can be queued via comfy run --template
# upscale   = dedicated post-stage
# fractal   = CPU numpy path (not Comfy)
# none      = not wired
Surface = str


@dataclass(frozen=True)
class Capability:
    id: str
    label: str
    class_types: tuple[str, ...] = ()
    name_contains: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()
    notes: str = ""


# Exact YES from live tower dump 2026-09-13 (Comfy Desk, 4114 classes).
# The markdown dump was not on this VM; names came from the operator follow-up.
TOWER_LIVE_YES: tuple[str, ...] = (
    "WanFunInpaintToVideo",
    "Wan22FunControlToVideo",
    "LanPaint_KSampler",
    "IPAdapterFaceID",
    "ControlNetLoader",
    "CreateVoronoiMask",
    "Image Perlin Power Fractal",
    "TeaCache",
    "WanVideoTeaCache",
)
TOWER_LIVE_RELATED: dict[str, tuple[str, ...]] = {
    "warp": ("GetWarpedNoiseFromVideo",),  # no exact VideoNoiseWarp
    "mmaudio": ("MMAudioModelLoader", "MMAudioSampler", "MMAudioVoCoder"),
}

# Tower claims vs repo wiring. Prefer live exact names over the older cache.
CAPABILITY_CATALOG: tuple[Capability, ...] = (
    Capability(
        "ltx25_t2v",
        "LTX 2.5 distilled T2V / I2V / FLF / MSR / A2V / T2A",
        class_types=(
            "LTXVLatentUpsampler",
            "LTXVICLoRALoader",
            "ComfyUILTX25MSRICLoRALoader",
            "ComfyUILTX25MSRMultiReferenceGuide",
        ),
        surfaces=("director", "patcher", "template"),
        templates=(
            "ltx25_t2v_i2v",
            "ltx25_t2v_i2v_two_stage",
            "ltx25_flf2v",
            "ltx25_msr",
            "ltx25_v2v_ic_lora",
            "ltx25_a2v",
            "ltx25_t2a",
        ),
        notes=(
            "Default catalog (PR #6). No env flag. Inventory-first loaders: "
            "GGUF Q4 → NVFP4 (VRAM≥14) → int8-convrot → bf16."
        ),
    ),
    Capability(
        "h3_t2v",
        "MiniMax H3 omni T2V / I2V / FLF / R2V (native stereo)",
        class_types=("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "UnetLoaderGGUF"),
        surfaces=("director", "patcher", "template"),
        templates=("h3_t2v", "h3_i2v", "h3_flf", "h3_r2v"),
        notes=(
            "Default catalog. fl2va = t2v/i2v/flf, ref2va = r2v. "
            "16GB pick: GGUF Q4_K DiT + Comfy TE (NVFP4/int8). CFG stays 1.0. "
            "H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). "
            "For tight lip-sync to an exact recording, use ltx25_a2v."
        ),
    ),
    Capability(
        "ltx_t2v",
        "LTX 2.3 T2V / I2V (base / eros / directors)",
        class_types=("EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent"),
        surfaces=("director", "patcher", "template"),
        templates=("base", "eros", "directors"),
        notes="Primary Rainey path. 8n+1 frame law + DOWNSCALE_LADDER.",
    ),
    Capability(
        "ltx_lipsync",
        "LTX lip-sync / LipDub",
        class_types=("LTXICLoRALoaderModelOnly", "LoadVideo"),
        surfaces=("director", "patcher", "template"),
        templates=("lipsync", "ltx23_lipsync_v08"),
        notes="Director forces lipsync when a source video is attached.",
    ),
    Capability(
        "wan22_t2v",
        "Wan 2.2 two-stage T2V (native UNET high/low)",
        class_types=("WanVideoNAG", "EmptyHunyuanLatentVideo"),
        surfaces=("director", "patcher", "template"),
        templates=("wan22", "vb_wan22_vid"),
        notes=(
            "Mickmumpitz native graph, not WanVideoWrapper Sampler. T2V UNETs only. "
            "16GB: QuantStack GGUF Q4_K_S when present, else Comfy-Org fp8 + Lightx2v. "
            "Sequential high/low. Dual bf16 is not a default. "
            "Wan TeaCache stays bypass-only (do not inject onto UNET→KSampler)."
        ),
    ),
    Capability(
        "k3nk_wan_aio",
        "K3NK WAN 2.2 AIO I2V HIGH/LOW fp8",
        class_types=(),
        name_contains=("k3nk",),
        surfaces=(),
        notes=(
            "HYPOTHESIS: tower weights. Hub search found only K3NK LoRAs, not an AIO "
            "I2V pack — do not invent filenames. Not a 16GB default. Use wan22 T2V."
        ),
    ),
    Capability(
        "wan_wrapper",
        "WanVideoWrapper (Sampler / Encode / TeaCache CACHEARGS)",
        class_types=("WanVideoSampler", "WanVideoModelLoader", "WanVideoTeaCache"),
        surfaces=(),
        notes="119 wrapper nodes in cached object_info; no Buddy graph uses WanVideoSampler.",
    ),
    Capability(
        "wan_fun_inpaint",
        "WAN Fun Inpaint",
        class_types=("WanFunInpaintToVideo",),
        surfaces=("director", "patcher", "template"),
        templates=("wan_fun_inpaint",),
        notes=(
            "Template wan_fun_inpaint: LoadImage start plate + LoadImageMask "
            "→ SetLatentNoiseMask on the WanFunInpaintToVideo latent. The core "
            "node has no mask widget (it builds a temporal concat_mask from "
            "start_image); the uploaded mask is the spatial noise mask. UNET is "
            "the attested Wan 2.2 high-noise file, not an invented K3NK AIO."
        ),
    ),
    Capability(
        "wan_fun_control",
        "WAN Fun Control",
        class_types=("Wan22FunControlToVideo", "WanFunControlToVideo"),
        surfaces=("template",),
        templates=("vb_zimage_turbo_cn",),
        notes=(
            "Live tower YES: Wan22FunControlToVideo. No Buddy API graph. "
            "Z-Image Fun-Controlnet-Union is gitignored under AI-RENDERING-EXAMPLE FILES/."
        ),
    ),
    Capability(
        "wan_teacache",
        "Wan TeaCache / EasyCache",
        class_types=("TeaCache", "WanVideoTeaCache", "WanVideoTeaCacheKJ", "WanVideoEasyCache"),
        surfaces=("bypass",),
        notes=(
            "Live tower YES: TeaCache + WanVideoTeaCache. WanVideoTeaCache outputs "
            "CACHEARGS — do not inject onto native UNET→KSampler. welltop-cn "
            "`TeaCache` on LTX graphs is inject-when-registered via "
            "graph_ops.ensure_teacache; missing class stays WARN+rewire bypass."
        ),
    ),
    Capability(
        "lightx2v",
        "LightX2V distill LoRAs",
        class_types=(),
        name_contains=("lightx2v",),
        surfaces=("template",),
        templates=("wan22",),
        notes=(
            "Baked into wan22 Power Lora Loader widgets. 16GB path: "
            "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32 when present."
        ),
    ),
    Capability(
        "vace",
        "Wan VACE / Phantom compositor",
        class_types=("WanVacePhantomSimpleV2", "WanVideoVACEStartToEndFrame", "WanVaceToVideo"),
        surfaces=("director", "patcher", "template"),
        templates=("vb_aivfx_adv", "vb_aivfx_adv_13", "vb_ai_renderer_smpl"),
        notes=(
            "Director-routed (vfx / aivfx / possession / vace). "
            "16GB default is vb_aivfx_adv_13 (Q4_K_M GGUF). v1.0 e4m3fn is heavy. "
            "Field map writes IterPromptBuilder.string_1. Example renderer files gitignored."
        ),
    ),
    Capability(
        "stand_in",
        "Wan Stand-In identity",
        class_types=("WanVideoAddStandInLatent",),
        surfaces=(),
        notes="Wrapper-only node. No Buddy template.",
    ),
    Capability(
        "lanpaint",
        "LanPaint (LanPaint_KSampler)",
        class_types=("LanPaint_KSampler",),
        surfaces=("bypass", "patcher"),
        notes=(
            "Live tower YES: LanPaint_KSampler (not a bare LanPaint class). "
            "No graph. Patcher writes seed/steps/cfg if a template uses it; "
            "validator bypasses if the pack is missing."
        ),
    ),
    Capability(
        "video_noise_warp",
        "GetWarpedNoiseFromVideo (warp family)",
        class_types=("GetWarpedNoiseFromVideo",),
        surfaces=("bypass",),
        notes=(
            "Live tower: NO exact VideoNoiseWarp. Use GetWarpedNoiseFromVideo. "
            "No Buddy graph. Soft-bypass if a graph names it and the pack is missing."
        ),
    ),
    Capability(
        "sam2",
        "SAM2 masks",
        class_types=("SAM2Segment", "ImpactSAM2VideoDetectorSEGS"),
        surfaces=(),
        notes="Cached object_info has SAM2Segment. AI-VFX preprocess uses SAM3Segment instead.",
    ),
    Capability(
        "sam3_vfx_preprocess",
        "SAM3 / DepthCrafter AI-VFX preprocess",
        class_types=("SAM3Segment", "DepthCrafter"),
        surfaces=("director", "template"),
        templates=("vb_aivfx_preprocess",),
        notes=(
            "Director-routed (sam3 / depthcrafter / vfx preprocess). "
            "Queues with baked control-video widgets; no Fun Inpaint loop."
        ),
    ),
    Capability(
        "ipadapter_faceid",
        "IPAdapter_plus + FaceID",
        class_types=(
            "IPAdapterUnifiedLoader",
            "IPAdapter",
            "IPAdapterFaceID",
            "easy ipadapterApplyFaceIDKolors",
        ),
        surfaces=("template",),
        templates=(),
        notes=(
            "Live tower YES: IPAdapterFaceID. SDXL ADV UI graphs also have "
            "IPAdapterUnifiedLoader (not API, not routed). No director/patcher path. "
            "Deferred: a queueable FaceID API graph needs its own conversion of "
            "those UI files; it is not a small add-on to Fun Inpaint."
        ),
    ),
    Capability(
        "controlnet_sd15",
        "ControlNet depth/canny SD1.5",
        class_types=("ControlNetLoader", "ControlNetApplyAdvanced", "CannyEdgePreprocessor"),
        surfaces=("template",),
        notes=(
            "Live tower YES: ControlNetLoader. In ltx23_lipsync_v08 API + SDXL ADV UI. "
            "Director will not pick a ControlNet path."
        ),
    ),
    Capability(
        "realistic_vision",
        "Realistic Vision SD1.5 checkpoint",
        class_types=(),
        name_contains=("realisticvision", "realistic_vision"),
        surfaces=(),
        notes="HYPOTHESIS: tower checkpoint. Not in MODEL_FILES or inventory snapshot.",
    ),
    Capability(
        "fractal_comfy",
        "Voronoi / Perlin fractal IMAGE nodes",
        class_types=("CreateVoronoiMask", "Image Perlin Power Fractal", "Image Perlin Noise"),
        surfaces=(),
        notes=(
            "Live tower YES: CreateVoronoiMask, Image Perlin Power Fractal. "
            "Buddy fractal path is CPU numpy, not these nodes. "
            "Deferred: a Comfy Voronoi/Perlin plate is a separate graph from "
            "the CPU fractal path and from the Fun Inpaint template."
        ),
    ),
    Capability(
        "fractal_cpu",
        "CPU Mandelbrot/Julia fractal (zoom/inpaint/outpaint)",
        class_types=(),
        surfaces=("fractal",),
        notes="master_agent/fractal — no Comfy, no recursive inpaint loop.",
    ),
    Capability(
        "raft",
        "RAFT optical flow",
        class_types=("RAFT", "RAFTFlow"),
        surfaces=(),
        notes="HYPOTHESIS: tower pack. Cached object_info has no RAFT* class (Recraft* is unrelated).",
    ),
    Capability(
        "qwen_krea_edit",
        "Qwen / Krea image edit",
        class_types=(
            "TextEncodeQwenImageEditPlus",
            "Krea2EditGroundedEncode",
            "Krea2EditModelPatch",
        ),
        surfaces=("director", "patcher", "template"),
        templates=("krea2_img", "vb_qwen_edit_360", "vb_aivfx_startimage", "vb_ccc41_krea2"),
        notes=(
            "Director-routed. krea2_img / qwen 360 / aivfx start-image have field maps. "
            "16GB: Krea NVFP4, Qwen Edit GGUF Q5_0. CCC 4.1 is heavy (safer: krea2_img)."
        ),
    ),
    Capability(
        "vb_movie_builder",
        "MickMumpitz Movie Builder (LTX 2.3 ADV)",
        class_types=("ShotAssembler", "OlmDragCrop"),
        surfaces=("director", "patcher", "template"),
        templates=("vb_movie_builder",),
        notes=(
            "OlmDragCrop is on the first-shot encode path (ImageExists → "
            "reference latent → KSampler). Required pack: ComfyUI-Olm-DragCrop "
            "(or equivalent). Missing class fails Hands / validate — do not "
            "silent-bypass or /prompt 400s. PanoramaViewerNode is preview-only "
            "and soft-bypasses when unregistered."
        ),
    ),
    Capability(
        "mickmumpitz_vfx",
        "MickMumpitz AI-VFX / CCC",
        class_types=("WanVacePhantomSimpleV2", "CCC_PromptStudio", "ShotAssembler"),
        surfaces=("director", "patcher", "template"),
        templates=(
            "vb_aivfx_adv",
            "vb_aivfx_adv_13",
            "vb_ccc_adv",
            "vb_ccc41_krea2",
        ),
        notes=(
            "Director-routed (ccc / aivfx). CCC ADV is HEAVY on 16GB — safer: flux. "
            "CCC ADV has no safe prompt widget — baked defaults; prefer --template. "
            "Movie Builder is a separate capability (requires OlmDragCrop). "
            "Some example files gitignored."
        ),
    ),
    Capability(
        "mmaudio",
        "MMAudio (video→audio)",
        class_types=("MMAudioModelLoader", "MMAudioSampler", "MMAudioVoCoder"),
        surfaces=("bypass",),
        notes=(
            "Live tower: NO exact class named MMAudio. Related: MMAudioModelLoader / "
            "Sampler / VoCoder. Music pipeline is beat-detect + mux. Soft-bypass if a "
            "graph names MMAudio* and the pack is missing."
        ),
    ),
    Capability(
        "seedvr2",
        "SeedVR2 upscale",
        class_types=("SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel"),
        surfaces=("upscale", "template"),
        templates=("upscale_seedvr2_api.json",),
        notes="master_agent/upscale.py default method=seedvr2 (in-repo JSON). RTX is gated on a gitignored file and raises FileNotFoundError when missing.",
    ),
    Capability(
        "sage_attention",
        "Sage-attention patches",
        class_types=(
            "LTX2MemoryEfficientSageAttentionPatch",
            "WanVideoMemoryEfficientSageAttentionPatch",
            "PathchSageAttentionKJ",
        ),
        surfaces=(),
        notes="HYPOTHESIS: tower launch flag. Patch nodes exist in cache; no Buddy graph inserts them.",
    ),
)


@dataclass
class CapabilityRow:
    capability: Capability
    in_object_info: list[str] = field(default_factory=list)
    in_templates: list[str] = field(default_factory=list)
    director: bool = False
    verdict: str = "none"
    gap: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.capability.id,
            "label": self.capability.label,
            "in_buddy": self.verdict,
            "how": list(self.capability.surfaces) or ["none"],
            "object_info_hits": self.in_object_info,
            "templates": self.in_templates,
            "director": self.director,
            "gap": self.gap,
            "notes": self.capability.notes,
        }


def _iter_api_workflows(root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            wf = unwrap_workflow(data)
        except ValueError:
            continue
        if not any(isinstance(n, dict) and "class_type" in n for n in wf.values()):
            continue
        yield path.relative_to(root).as_posix(), wf


def index_template_class_types(root: Path = WORKFLOWS_DIR) -> dict[str, set[str]]:
    """filename/slug → class_types present in that API graph."""
    from master_agent.comfy.workflow_patcher import _load_manifests

    by_rel: dict[str, set[str]] = {}
    for rel, wf in _iter_api_workflows(root):
        by_rel[rel] = {
            str(n.get("class_type"))
            for n in wf.values()
            if isinstance(n, dict) and n.get("class_type")
        }
    # also index manifest slugs
    out: dict[str, set[str]] = dict(by_rel)
    for slug, meta in (_load_manifests() or {}).items():
        if not isinstance(meta, dict):
            continue
        filename = meta.get("file")
        if filename in by_rel:
            out[str(slug)] = by_rel[filename]
    for slug, filename in WORKFLOW_FILES.items():
        if filename in by_rel:
            out[slug] = by_rel[filename]
    return out


def _class_hits(cap: Capability, registry: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    keys = list(registry.keys()) if isinstance(registry, dict) else []
    for ct in cap.class_types:
        if ct in registry:
            hits.append(ct)
    for token in cap.name_contains:
        needle = token.lower()
        for key in keys:
            if needle in str(key).lower() and key not in hits:
                hits.append(key)
    return hits


def _template_hits(cap: Capability, index: dict[str, set[str]]) -> list[str]:
    hits: list[str] = []
    wanted = set(cap.class_types)
    for slug in cap.templates:
        classes = index.get(slug) or index.get(slug.replace("\\", "/"))
        if classes:
            hits.append(slug)
            continue
        # path-style template id
        if slug in index:
            hits.append(slug)
    if wanted:
        for slug, classes in index.items():
            if classes & wanted and slug not in hits:
                hits.append(slug)
    return sorted(hits)


def _verdict(cap: Capability, row: CapabilityRow) -> tuple[str, str]:
    surfaces = set(cap.surfaces)
    if "director" in surfaces:
        return "yes", ""
    if "fractal" in surfaces:
        return "partial", "CPU-only; no Comfy fractal/inpaint loop."
    if "upscale" in surfaces and row.in_templates:
        return "partial", "Post-stage only; not director-routed."
    if "bypass" in surfaces:
        if row.in_object_info:
            return "partial", cap.notes
        return "no", "Aliases not in this object_info; bypass would drop them if a graph asked."
    if "template" in surfaces and row.in_templates:
        return "partial", "Template/manual comfy run only; director will not pick this."
    if "patcher" in surfaces:
        return "partial", "Patcher/field-map exists; director allowlist does not include it."
    if row.in_object_info:
        return "no", "Nodes exist on this Comfy snapshot but Buddy has no graph or injector."
    return "no", cap.notes or "Not referenced in repo graphs or object_info snapshot."


def probe_capabilities(
    object_info: dict[str, Any] | None = None,
    *,
    workflows_dir: Path = WORKFLOWS_DIR,
) -> list[CapabilityRow]:
    registry = object_info if isinstance(object_info, dict) else {}
    index = index_template_class_types(workflows_dir)
    director = set(WORKFLOW_FILES)
    rows: list[CapabilityRow] = []
    for cap in CAPABILITY_CATALOG:
        row = CapabilityRow(
            capability=cap,
            in_object_info=_class_hits(cap, registry),
            in_templates=_template_hits(cap, index),
            director=bool(set(cap.templates) & director) or "director" in cap.surfaces,
        )
        row.verdict, row.gap = _verdict(cap, row)
        rows.append(row)
    return rows


def format_matrix(rows: list[CapabilityRow], *, source: str = "unknown") -> str:
    lines = [
        f"Video Buddy Comfy capability matrix  (object_info: {source})",
        "",
        f"{'Capability':<42} {'In Buddy?':<10} {'How':<28} Gap / fix",
        "-" * 110,
    ]
    for row in rows:
        how = ",".join(row.capability.surfaces) or "none"
        gap = row.gap.replace("\n", " ")
        lines.append(f"{row.capability.label:<42} {row.verdict:<10} {how:<28} {gap}")
    lines.append("")
    lines.append(
        "Director allowlist: " + ", ".join(sorted(WORKFLOW_FILES)) + "."
    )
    lines.append(
        "object_info is validation + this probe — not automatic graph synthesis."
    )
    return "\n".join(lines) + "\n"


def load_object_info_for_probe(
    *, prefer_live: bool = True
) -> tuple[dict[str, Any], str]:
    from master_agent.comfy.client import ComfyClient

    try:
        return ComfyClient().load_object_info(
            cache_path=OBJECT_INFO_CACHE, prefer_live=prefer_live
        )
    except Exception:
        if OBJECT_INFO_CACHE.is_file():
            return json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8")), "cache"
        return {}, "empty"


def run_probe(*, prefer_live: bool = True) -> tuple[list[CapabilityRow], str]:
    info, source = load_object_info_for_probe(prefer_live=prefer_live)
    return probe_capabilities(info), source


__all__ = [
    "CAPABILITY_CATALOG",
    "TOWER_LIVE_RELATED",
    "TOWER_LIVE_YES",
    "Capability",
    "CapabilityRow",
    "format_matrix",
    "index_template_class_types",
    "probe_capabilities",
    "run_probe",
]
