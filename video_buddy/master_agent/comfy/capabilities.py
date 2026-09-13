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


# Tower claims vs repo wiring. class_types are Comfy node names as of the
# cached object_info snapshot — live tower may differ (see AUDIT.md).
CAPABILITY_CATALOG: tuple[Capability, ...] = (
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
        notes="Mickmumpitz native graph, not WanVideoWrapper Sampler. T2V UNETs only.",
    ),
    Capability(
        "k3nk_wan_aio",
        "K3NK WAN 2.2 AIO I2V HIGH/LOW fp8",
        class_types=(),
        name_contains=("k3nk",),
        surfaces=(),
        notes="HYPOTHESIS: tower weights. Not in MODEL_FILES or inventory snapshot.",
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
        surfaces=(),
        notes="Present in cached object_info; no shipped API graph or patcher field.",
    ),
    Capability(
        "wan_fun_control",
        "WAN Fun Control",
        class_types=("WanFunControlToVideo", "Wan22FunControlToVideo"),
        surfaces=("template",),
        templates=("vb_zimage_turbo_cn",),
        notes="Z-Image Fun-Controlnet-Union is gitignored under AI-RENDERING-EXAMPLE FILES/.",
    ),
    Capability(
        "wan_teacache",
        "Wan TeaCache / EasyCache",
        class_types=("WanVideoTeaCache", "WanVideoTeaCacheKJ", "WanVideoEasyCache"),
        surfaces=("bypass",),
        notes=(
            "Bypass-only (validator). WanVideoTeaCache outputs CACHEARGS, not MODEL — "
            "do not inject onto LTX or native UNET→KSampler graphs."
        ),
    ),
    Capability(
        "lightx2v",
        "LightX2V distill LoRAs",
        class_types=(),
        name_contains=("lightx2v",),
        surfaces=("template",),
        templates=("wan22",),
        notes="Baked into wan22 Power Lora Loader widgets; patcher does not select them.",
    ),
    Capability(
        "vace",
        "Wan VACE / Phantom compositor",
        class_types=("WanVacePhantomSimpleV2", "WanVideoVACEStartToEndFrame", "WanVaceToVideo"),
        surfaces=("template",),
        templates=("vb_aivfx_adv", "vb_aivfx_adv_13", "vb_ai_renderer_smpl"),
        notes="Shipped AI-VFX API graphs. Not director-routed. Example renderer files gitignored.",
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
        "LanPaint",
        class_types=("LanPaint",),
        name_contains=("lanpaint",),
        surfaces=(),
        notes="HYPOTHESIS: tower pack. Not in cached object_info or any workflow JSON.",
    ),
    Capability(
        "video_noise_warp",
        "VideoNoiseWarp",
        class_types=("VideoNoiseWarp",),
        name_contains=("noisewarp", "video_noise_warp"),
        surfaces=(),
        notes="HYPOTHESIS: tower pack. Not in cached object_info or any workflow JSON.",
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
        surfaces=("template",),
        templates=("vb_aivfx_preprocess",),
        notes="Template only; no agent loop that feeds masks into Fun Inpaint.",
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
            "SDXL ADV UI graphs contain IPAdapterUnifiedLoader (not API, not routed). "
            "Cached object_info has easy-use FaceID wrappers only — not IPAdapter_plus native."
        ),
    ),
    Capability(
        "controlnet_sd15",
        "ControlNet depth/canny SD1.5",
        class_types=("ControlNetLoader", "ControlNetApplyAdvanced", "CannyEdgePreprocessor"),
        surfaces=("template",),
        notes="Nodes exist in cache. No director/patcher path. SDXL ADV UI graphs only.",
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
        notes="In cached object_info. Buddy fractal path is CPU numpy, not these nodes.",
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
        surfaces=("patcher", "template"),
        templates=("krea2_img", "vb_qwen_edit_360", "vb_aivfx_startimage", "vb_ccc41_krea2"),
        notes="Templates + krea2_img field map. Not in director allowlist.",
    ),
    Capability(
        "mickmumpitz_vfx",
        "MickMumpitz AI-VFX / Movie Builder / CCC",
        class_types=("WanVacePhantomSimpleV2", "CCC_PromptStudio", "ShotAssembler"),
        surfaces=("template",),
        templates=(
            "vb_aivfx_adv",
            "vb_aivfx_adv_13",
            "vb_movie_builder",
            "vb_ccc_adv",
            "vb_ccc41_krea2",
        ),
        notes="Large curated graphs. Director never routes here. Some example files gitignored.",
    ),
    Capability(
        "mmaudio",
        "MMAudio (video→audio)",
        class_types=("MMAudioSampler", "MMAudioModelLoader", "OviMMAudioVAELoader"),
        surfaces=(),
        notes=(
            "Cache has OviMMAudioVAELoader / WanVideoEmptyMMAudioLatents only. "
            "Music pipeline is beat-detect + mux, not MMAudio generation."
        ),
    ),
    Capability(
        "seedvr2",
        "SeedVR2 upscale",
        class_types=("SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel"),
        surfaces=("upscale", "template"),
        templates=("upscale_seedvr2_api.json",),
        notes="master_agent/upscale.py method=seedvr2. RTX path points at a gitignored file.",
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
    "Capability",
    "CapabilityRow",
    "format_matrix",
    "index_template_class_types",
    "probe_capabilities",
    "run_probe",
]
