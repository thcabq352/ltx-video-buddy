"""Argument parser and dispatcher for python -m master_agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from master_agent.config import LIPDUB_ANCHOR, LIPDUB_REFRAME, LIPDUB_SILENCE_MODE, ensure_dirs
from master_agent.cli.common import _DurationSet, _DimSet, _add_selector_flags
from master_agent.cli.studio import cmd_curriculum, cmd_about, cmd_kb, cmd_ui, cmd_budget, cmd_hermes
from master_agent.cli.doctor import cmd_setup, cmd_inventory, cmd_health, cmd_scan_models
from master_agent.cli.comfy import cmd_fetch_object_info, cmd_capabilities, cmd_validate, cmd_power_tune, cmd_diagnose, cmd_comfy
from master_agent.cli.run import cmd_run, cmd_brief
from master_agent.cli.rainey1 import cmd_rainey1_batch
from master_agent.rainey1.recipes import recipe_choices
from master_agent.cli.models import cmd_download_flux, cmd_download_models, cmd_workflows, cmd_models_manifest, cmd_models_select
from master_agent.cli.fractal import cmd_fractal
from master_agent.cli.music import cmd_mv, cmd_music, cmd_heartmula
from master_agent.cli.persona import cmd_persona, cmd_soul, cmd_character, cmd_lora
from master_agent.cli.agent import cmd_agent

def main(argv: list[str] | None = None) -> int:
    # Windows console is cp1252 — never crash on LLM-emitted unicode (e.g. →)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ensure_dirs()
    parser = argparse.ArgumentParser(prog="python -m master_agent", description="VIDEO BUDDY")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "curriculum",
        help="print LESSON_BUDDY_WORKS_HERE (L0→L5) and the Part 2 overnight gate",
    )
    p.add_argument("--json", action="store_true", help="machine-readable card")
    p.set_defaults(func=cmd_curriculum)

    p = sub.add_parser("about", help="print the VIDEO BUDDY identity card")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_about)

    p = sub.add_parser(
        "setup",
        aliases=["doctor"],
        help="check local deps + 16GB pack policy + LTX 2.5 / H3 weights (scan first; --fix-models after you agree)",
    )
    p.add_argument("--fix", action="store_true", help="create venv, pip install, Playwright, .env, ffmpeg; Ollama pulls only for models missing from `ollama list` and only with --yes or y/n")
    p.add_argument(
        "--fix-models",
        action="store_true",
        help="after the inventory, download confirmed-missing LTX 2.5 weights (explicit consent)",
    )
    p.add_argument("--yes", action="store_true", help="consent for Ollama pulls (--fix) and for --download")
    fetch = p.add_mutually_exclusive_group()
    fetch.add_argument("--scan-only", action="store_true", help="inventory and report only; never fetch weights or pull Ollama")
    fetch.add_argument("--use-existing", action="store_true", help="use weights already on disk; do not download")
    fetch.add_argument("--download", action="store_true", help="fetch confirmed-missing weights after --yes or a y/n prompt")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser(
        "inventory",
        help="list discovered model files (paths + roles) before doctor suggests downloads",
    )
    p.add_argument("--json", action="store_true", help="machine-readable inventory")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("workflows", help="list default catalog variants (no env flags)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--vram", action="store_true", help="16GB-class pack table (RTX 5060 Ti)")
    p.set_defaults(func=cmd_workflows)

    p = sub.add_parser("health", help="check ComfyUI reachability")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("fetch-object-info", help="cache /object_info to state/")
    p.set_defaults(func=cmd_fetch_object_info)

    p = sub.add_parser("scan-models", help="scan models/ into inventory")
    p.add_argument("--strict", action="store_true", help="exit 1 if any bundle is not runnable")
    p.set_defaults(func=cmd_scan_models)

    p = sub.add_parser("validate", help="validate workflow JSON")
    p.add_argument("file", nargs="?", help="workflow JSON path")
    p.add_argument("--all", "-a", action="store_true", help="validate all workflows/*.json")
    p.add_argument("--offline", action="store_true", help="use cached object_info only")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--strict",
        action="store_true",
        help="illegal LTX frame counts (not 8n+1, min 9) are ERROR instead of auto-correct",
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("kb", help="knowledge base: ingest | search | stats")
    p.add_argument("kb_command", choices=["ingest", "search", "stats"])
    p.add_argument("query", nargs="?", default="", help="search query")
    p.add_argument("-k", type=int, default=3, help="max hits (default 3)")
    p.add_argument("--workflows", action="store_true", help="search workflows instead of runs")
    p.add_argument(
        "--knowledge",
        action="store_true",
        help="search the git-synced knowledge/ collection",
    )
    p.set_defaults(func=cmd_kb)

    p = sub.add_parser("ui", help="web dashboard (FastAPI) on 127.0.0.1:8189")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8189)
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback --host and skip the Host/Origin check (no auth!)",
    )
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("run", help="orchestrated generation: patch -> validate -> submit -> judge")
    p.add_argument("request", help="what to generate (natural language)")
    p.add_argument(
        "--variant",
        help=(
            "force catalog variant (see: python -m master_agent workflows). "
            "H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). "
            "For tight lip-sync to an exact recording, use ltx25_a2v."
        ),
    )
    p.add_argument(
        "--preset",
        choices=["rainey1"],
        default=None,
        help=(
            "director recipe pack. rainey1 pins base and fills the prompt, "
            "negative, frames, and size from the recipe (a brief that already "
            "says rainey1 does the same without this flag)"
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        action=_DurationSet,
        help="seconds (default 5; photo + voice follows the audio unless this is set)",
    )
    p.set_defaults(duration_set=False)
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--width", type=int, default=768, action=_DimSet)
    p.add_argument("--height", type=int, default=512, action=_DimSet)
    p.set_defaults(width_set=False, height_set=False)
    p.add_argument("--video", help="source video file (lipsync, ltx23_inoutpaint, or ltx25_inoutpaint); uploaded to ComfyUI first")
    p.add_argument("--image", help="source image file (i2v); uploaded to ComfyUI first")
    p.add_argument(
        "--mask",
        help="inpaint mask image (white=fill); uploaded to ComfyUI first. "
        "wan_fun_inpaint generates fun_inpaint_mask.png when this is omitted. "
        "ltx23_inoutpaint and ltx25_inoutpaint generate a center mask when this is omitted. "
        "--outpaint builds the border mask itself",
    )
    p.add_argument(
        "--aspect",
        help="outpaint target aspect, for example 9:16 or 16:9 (with --video)",
    )
    p.add_argument(
        "--outpaint",
        action="store_true",
        help="extend the source video; requires --aspect or an explicit --width and --height",
    )
    p.add_argument("--audio", help="source audio file; uploaded to ComfyUI first")
    p.add_argument(
        "--line",
        default=None,
        help=(
            "Exact words for H3 voice mode (h3_r2v). Injected word for word. "
            "The --audio file is a 2–12 s voice sample; longer samples are trimmed "
            "to the loudest 12 s."
        ),
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="one render: skip the judge and the quality-bar revise (no second attempt)",
    )
    p.add_argument("--max-judge-rounds", type=int, default=None, help="judge retry budget")
    p.add_argument("--storyboard", choices=["smart", "always", "multi_only", "off"],
                   help="storyboard mode (default: env STORYBOARD_MODE or smart)")
    p.add_argument("--llm-panel",
                   help="storyboard LLM panel: preset (default|local | grok | "
                        "grok+local|both | grok+claude | duo) or comma list")
    p.add_argument("--panel-judge",
                   help="provider that picks the winning storyboard (default: env PANEL_JUDGE or ollama; llamacpp[:model] ok)")
    p.add_argument("--max-full-judge-rounds", type=int, default=None,
                   help="full-video judge re-gen budget (multi-segment)")
    p.add_argument("--dry-run", action="store_true",
                   help="storyboard + patch + validate all segments, no GPU queue. "
                        "Long ltx25_a2v prints the lipdub segment plan.")
    p.add_argument(
        "--tripod",
        action="store_true",
        help="locked-off camera for ltx25_a2v talking heads (no push-in, no drift)",
    )
    p.add_argument(
        "--lipdub-max-s",
        type=float,
        default=None,
        help="split ltx25_a2v audio longer than this many seconds (default 6.5, env LIPDUB_SEGMENT_MAX_S)",
    )
    p.add_argument(
        "--silence-min-s",
        type=float,
        default=None,
        help="pauses at least this long are closed-mouth idle or bridge pieces (default 0.25)",
    )
    p.add_argument(
        "--silence-mode",
        choices=["idle", "hold", "bridge"],
        default=LIPDUB_SILENCE_MODE if LIPDUB_SILENCE_MODE in ("idle", "hold", "bridge") else "idle",
        help=(
            "how ltx25_a2v treats a pause at least --silence-min-s long: "
            "idle (default, closed-mouth breathing render), "
            "hold (#34 still plate plus frozen tail), "
            "or bridge (9-frame mouth close, tail held)"
        ),
    )
    p.add_argument(
        "--anchor",
        choices=["source", "previous", "hybrid", "pause-reset"],
        default=(
            LIPDUB_ANCHOR
            if LIPDUB_ANCHOR in ("source", "previous", "hybrid", "pause-reset")
            else "previous"
        ),
        help=(
            "identity/framing anchor for a segmented lipdub. "
            "previous (default) is the last-frame chain: each piece starts "
            "from the previous last frame, overlap trimmed on speech-to-speech seams. "
            "hybrid (experimental) conditions on the source still and crossfades "
            "the previous frame across the overlap. "
            "source (experimental) uses the still only, with the same crossfade. "
            "pause-reset (opt-in) chains speech from the previous frame and, on each "
            "silence, pins the source still as a last-frame LTXVAddGuide so the "
            "look glides back during the pause"
        ),
    )
    p.add_argument(
        "--max-piece-seconds",
        type=float,
        default=None,
        help=(
            "split a speech run longer than this many seconds at the quietest "
            "audio frame (default 3.0, env LIPDUB_MAX_PIECE_S). "
            "Joins stay on the frame grid with the wav. "
            "Does not change the 6.5s threshold that decides whether to segment. "
            "Word edges are snapped onto the frame grid before the quietest legal frame is chosen."
        ),
    )
    p.add_argument(
        "--pause-reset-strength",
        type=float,
        default=None,
        help=(
            "LTXVAddGuide strength for --anchor pause-reset "
            "(default 0.65, env LIPDUB_PAUSE_RESET_STRENGTH). "
            "1.0 pinned the still so hard the head turned in about 0.2s"
        ),
    )
    p.add_argument(
        "--pause-reset-min-s",
        type=float,
        default=None,
        help=(
            "only pin the source still on a pause at least this long "
            "(default 0.5, env LIPDUB_PAUSE_RESET_MIN_S). "
            "Shorter breaths stay plain idle"
        ),
    )
    p.add_argument(
        "--reframe",
        choices=["on", "off"],
        default="on" if LIPDUB_REFRAME else "off",
        help=(
            "experimental: scale each piece back to the source still after render. "
            "Default off. Pass --reframe on to enable; the geometric snap-back "
            "can zoom speech pieces in and then jump back to the still"
        ),
    )
    p.add_argument(
        "--lipdub-overlap",
        type=int,
        default=None,
        help="overlap frames trimmed at speech-to-speech seams (default 8)",
    )
    p.add_argument(
        "--words",
        help="JSON word timestamps so long lipdub splits on pauses and never mid-word",
    )
    p.add_argument(
        "--heartmula-transcribe",
        action="store_true",
        help=(
            "when --words is omitted, transcribe --audio with HeartTranscriptor "
            "into the same {w,s,e} list. Dry-run plans only. "
            "Does not replace a faster-whisper --words file."
        ),
    )
    p.add_argument(
        "--self-improve-dry",
        action="store_true",
        help="closed judge→revise→rejudge loop (quality_bar a/c/d); no Comfy queue",
    )
    p.add_argument("--power-mode", action="store_true",
                   help="LLM graph ops after patch (object_info + RAG; validate-gated)")
    p.add_argument("--no-power-mode", action="store_true",
                   help="disable power mode even if POWER_MODE=1")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.add_argument(
        "--attach",
        help="previs buddy.comfy.attach/v1 / WorkflowPatchPlan JSON (patch after director)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "power-tune",
        help="dry-run power mode: patch variant + LLM graph ops + validate (no GPU)",
    )
    p.add_argument("request", help="creative brief / intent for the graph edit")
    p.add_argument("--variant", default="base",
                   help="workflow variant template (default catalog; see `workflows`)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], default="draft")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--provider", help="LLM provider (ollama|llamacpp|grok|auto)")
    p.add_argument("--json", action="store_true", help="print machine-readable result")
    p.add_argument("--out", help="write patched workflow JSON to this path")
    p.set_defaults(func=cmd_power_tune)

    p = sub.add_parser("persona", help="personas: list | show | set <slug>")
    p.add_argument("persona_command", choices=["list", "show", "set"])
    p.add_argument("name", nargs="?", default="", help="persona slug (show/set)")
    p.set_defaults(func=cmd_persona)

    p = sub.add_parser("soul", help="souls: list | show | set <slug>")
    p.add_argument("soul_command", choices=["list", "show", "set"])
    p.add_argument("name", nargs="?", default="", help="soul slug (show/set)")
    p.set_defaults(func=cmd_soul)

    p = sub.add_parser("brief", help="interview-only: turn a rough idea into a creative brief")
    p.add_argument("request", help="your rough idea (natural language)")
    p.add_argument("--go", action="store_true", help="generate immediately after the interview")
    p.add_argument("--duration", type=float, default=8.0, help="seconds if the brief doesn't say (--go)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile (--go)")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser(
        "fractal",
        help="procedural fractal video: zoom | inpaint | outpaint (CPU only)",
    )
    p.add_argument("request", nargs="?", default="", help="optional brief (for the run record)")
    p.add_argument("--mode", choices=["zoom", "inpaint", "outpaint"], default="zoom",
                   help="zoom=deep-zoom; inpaint=fill a hole; outpaint=expand canvas borders")
    p.add_argument("--image", help="source image for inpaint/outpaint")
    p.add_argument("--mask", help="inpaint mask image (white=fill); default: center ellipse")
    p.add_argument("--expand", type=int, default=128,
                   help="outpaint border thickness in pixels (default 128)")
    p.add_argument("--cover", type=float, default=0.4,
                   help="inpaint center-hole size as fraction of frame (default 0.4)")
    p.add_argument("--feather", type=int, default=28,
                   help="soft edge radius in pixels for inpaint/outpaint (default 28)")
    p.add_argument("--duration", type=float, default=20.0, help="seconds (default 20)")
    p.add_argument("--fps", type=int, default=None, help="frames per second (default 24)")
    p.add_argument("--width", type=int, default=768,
                   help="zoom mode width (inpaint/outpaint use the image size)")
    p.add_argument("--height", type=int, default=512,
                   help="zoom mode height (inpaint/outpaint use the image size)")
    p.add_argument("--target", default="seahorse",
                   choices=["seahorse", "elephant", "minibrot", "spiral"],
                   help="zoom target (default seahorse)")
    p.add_argument("--palette", default="fire",
                   choices=["fire", "ocean", "monochrome", "neon", "sunset"])
    p.add_argument("--julia", action="store_true", help="Julia-set mode (c wobbles with the beat)")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--audio", help="audio track: beat-reactive render + mux")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.set_defaults(func=cmd_fractal)

    p = sub.add_parser("music", help="beat-synced music video from an audio track")
    p.add_argument("request", help="creative brief (natural language)")
    p.add_argument("--audio", help="audio file (mp3/wav/...). Or HeartMuLa lyrics+tags")
    p.add_argument("--heartmula-lyrics", help="generate the track with HeartMuLa when --audio is omitted")
    p.add_argument("--heartmula-tags", help="comma-separated HeartMuLa tags, no spaces (piano,happy)")
    p.add_argument("--heartmula-duration", type=float, default=None, help="generated track seconds (default 30)")
    p.add_argument("--visual", choices=["shots", "fractal"], default="shots",
                   help="shots = ComfyUI generation per shot; fractal = CPU beat-reactive zoom")
    p.add_argument("--variant", help="force catalog variant (shots mode)")
    p.add_argument("--quality", choices=["draft", "balanced", "quality"], help="quality profile")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="one render per shot: skip the judge and the quality-bar revise",
    )
    p.add_argument("--max-judge-rounds", type=int, default=None, help="judge retry budget")
    p.add_argument("--llm-panel",
                   help="storyboard LLM panel: preset (default|local | grok | "
                        "grok+local|both | grok+claude | duo) or comma list")
    p.add_argument("--panel-judge",
                   help="provider that picks the winning storyboard")
    p.add_argument("--upscale", choices=["rtx", "seedvr2"], default=None,
                   help="post-stage upscale of the final video")
    p.add_argument("--no-interview", action="store_true",
                   help="skip the persona intake interview")
    p.set_defaults(func=cmd_music)

    p = sub.add_parser(
        "mv",
        help="Comfy/LTX music-video mode: beat plan → unique burns → Remotion (not Grok)",
    )
    mv_sub = p.add_subparsers(dest="mv_command", required=True)
    plan_p = mv_sub.add_parser("plan", help="write buddy.mv.beat_plan/v1 JSON from an audio track")
    plan_p.add_argument("--audio", required=True, help="audio file (mp3/wav/...)")
    plan_p.add_argument("--out", help="beat plan JSON (default out/beat_plan.json)")
    plan_p.add_argument("--fps", type=int, default=30, help="frame grid (MTV cut is 30)")
    plan_p.add_argument("--json", action="store_true", help="print the plan to stdout")
    plan_p.set_defaults(func=cmd_mv)
    rend_p = mv_sub.add_parser(
        "render",
        help="plan → Comfy/LTX burn (I2V if --image) → unique check → Remotion",
    )
    rend_p.add_argument("request", nargs="?", default="music video", help="creative brief")
    rend_p.add_argument("--audio", help="full-track audio (mp3/wav/...). Or HeartMuLa lyrics+tags")
    rend_p.add_argument("--heartmula-lyrics", help="generate the track when --audio is omitted")
    rend_p.add_argument("--heartmula-tags", help="comma-separated HeartMuLa tags, no spaces")
    rend_p.add_argument("--heartmula-duration", type=float, default=None, help="generated track seconds (default 30)")
    rend_p.add_argument("--heartmula-seed", type=int, default=None, help="torch seed before HeartMuLaGenPipeline")
    rend_p.add_argument("--out", default=str(Path("out") / "MV-FIXED.mp4"), help="1080p output")
    rend_p.add_argument("--plan", help="reuse a buddy.mv.beat_plan/v1 JSON (skip analyze)")
    rend_p.add_argument("--image", help="still / character lock → I2V (else T2V)")
    rend_p.add_argument("--variant", default="ltx25_t2v_i2v", help="LTX catalog slug")
    rend_p.add_argument("--prompt", help="override brief (else positional request)")
    rend_p.add_argument("--seed", type=int, help="base seed (windows offset uniquely)")
    rend_p.add_argument("--width", type=int, default=768, help="Comfy burn width")
    rend_p.add_argument("--height", type=int, default=512, help="Comfy burn height")
    rend_p.add_argument("--fps", type=int, default=30, help="Remotion / beat-plan fps")
    rend_p.add_argument(
        "--dry-run",
        action="store_true",
        help="plan + unique mock burns + Remotion wiring; no GPU / Comfy / Node",
    )
    rend_p.add_argument("--work-dir", dest="work_dir", help="scratch dir for clips + props")
    rend_p.add_argument("--json", action="store_true")
    rend_p.set_defaults(func=cmd_mv)

    p = sub.add_parser(
        "heartmula",
        help="HeartMuLa: generate a track from lyrics+tags, or transcribe word timestamps",
    )
    hm = p.add_subparsers(dest="heartmula_command", required=True)
    gen_p = hm.add_parser("generate", help="lyrics + tags → wav via heartlib (dry-run needs no package)")
    gen_p.add_argument("--lyrics", required=True, help="lyric text, or a path heartlib can read")
    gen_p.add_argument("--tags", required=True, help="comma-separated tags without spaces")
    gen_p.add_argument("--duration", type=float, default=30.0, help="seconds (heartlib example uses 240)")
    gen_p.add_argument("--seed", type=int, default=None, help="torch.manual_seed before the pipeline")
    gen_p.add_argument("--out", required=True, help="output wav path")
    gen_p.add_argument("--topk", type=int, default=50)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=1.5)
    gen_p.add_argument(
        "--max-seq-len",
        dest="max_seq_len",
        type=int,
        default=None,
        help=(
            "backbone KV window. Default sizes to the clip and the card. "
            "heartlib's 8192 OOMs on 16GB during the GQA expand; 512 is the 5s smoke"
        ),
    )
    gen_p.add_argument(
        "--low-vram",
        action="store_true",
        help="size the KV window as if the card is 16GB",
    )
    gen_p.add_argument("--dry-run", action="store_true", help="print the plan; do not import heartlib")
    gen_p.set_defaults(func=cmd_heartmula)
    tr_p = hm.add_parser("transcribe", help="audio → words.json for lipdub --words")
    tr_p.add_argument("--audio", required=True, help="music or vocal file")
    tr_p.add_argument("--out", required=True, help="words JSON path")
    tr_p.add_argument("--dry-run", action="store_true", help="print the plan; do not import heartlib")
    tr_p.set_defaults(func=cmd_heartmula)

    p = sub.add_parser("download-flux", help="download Flux fp8 weights (~17GB, one-time)")
    p.set_defaults(func=cmd_download_flux)

    p = sub.add_parser(
        "download-models",
        help="scan 16GB-class packs (LTX 2.5 / H3 / Wan / VACE / Krea / Flux / Qwen); download missing only with --yes",
    )
    p.add_argument("--ltx25", action="store_true", default=True, help="LTX 2.5 distilled split pack (default)")
    p.add_argument("--h3", action="store_true", help="MiniMax H3 GGUF + Comfy TE/VAE pack")
    p.add_argument(
        "--heartmula",
        action="store_true",
        help="HeartMuLa + HeartCodec + HeartTranscriptor attested slots (list missing; --yes fetches)",
    )
    p.add_argument("--wan", action="store_true", help="Wan 2.2 GGUF/fp8 + Lightx2v 16GB pack")
    p.add_argument("--vace", action="store_true", help="VACE Skyreels Q4_K_M GGUF")
    p.add_argument("--krea", action="store_true", help="Krea-2 turbo NVFP4")
    p.add_argument("--qwen", action="store_true", help="Qwen-Image-Edit GGUF Q5_0")
    p.add_argument("--flux-pack", dest="flux_pack", action="store_true", help="Flux.1-dev GGUF/fp8")
    p.add_argument(
        "--bundle",
        help="weight bundle id (ltx25_*|h3_*|wan22|vace|krea2|flux|qwen_edit)",
    )
    p.add_argument("--yes", action="store_true", help="consent: download the missing mandatory set")
    p.add_argument("--optional", action="store_true", help="also fetch optional Hub files (distilled LoRA 450, temporal upscaler)")
    p.add_argument("--json", action="store_true")
    fetch = p.add_mutually_exclusive_group()
    fetch.add_argument("--scan-only", action="store_true", help="inventory and report only; never fetch (overrides --yes)")
    fetch.add_argument("--use-existing", action="store_true", help="keep whatever is on disk; do not download")
    fetch.add_argument("--download", action="store_true", help="fetch confirmed-missing files after --yes or a y/n prompt")
    p.add_argument(
        "--selector",
        action="store_true",
        help="open the LTX 2.3|2.5 model selector instead of a pack scan",
    )
    _add_selector_flags(p, include_optional=False)
    p.set_defaults(func=cmd_download_models)

    p = sub.add_parser(
        "models",
        help="model selector: manifest | select (LTX 2.3|2.5 checklist; no comfy install)",
    )
    models_sub = p.add_subparsers(dest="models_command", required=True)
    manifest_p = models_sub.add_parser("manifest", help="print the capability-grouped catalog JSON")
    manifest_p.set_defaults(func=cmd_models_manifest)
    select_p = models_sub.add_parser(
        "select",
        help="radio 2.3|2.5, required and optional rows, running total, disk gate",
    )
    _add_selector_flags(select_p)
    select_p.add_argument("--yes", action="store_true", help="consent: download the selected missing set")
    select_p.add_argument("--json", action="store_true")
    select_p.add_argument(
        "--scan-only",
        action="store_true",
        help="print the checklist only; do not download or write selector state",
    )
    select_p.set_defaults(func=cmd_models_select)

    p = sub.add_parser("character", help="CCC stage: create | list")
    p.add_argument("character_command", choices=["create", "list"])
    p.add_argument("description", nargs="?", default="", help="character description")
    p.add_argument("--name", help="override the LLM-chosen character name")
    p.add_argument("--shots", type=int, default=None, help="cap sheet shots (default: all)")
    p.add_argument("--no-vision", action="store_true", help="skip vision identity curation")
    p.add_argument("--train", action="store_true", help="chain into LoRA train+validate loop")
    p.set_defaults(func=cmd_character)

    p = sub.add_parser("lora", help="Flux LoRA stage: setup | train | validate")
    p.add_argument("lora_command", choices=["setup", "train", "validate"])
    p.add_argument("name", nargs="?", default="", help="character name")
    p.add_argument("--steps", type=int, default=0, help="override training steps")
    p.add_argument("--lr", type=float, default=0.0, help="override learning rate")
    p.add_argument("--rank", type=int, default=0, help="override lora rank")
    p.add_argument("--attempt", type=int, default=1, help="attempt number (retry ladder)")
    p.add_argument("--no-resume", action="store_true", help="start training from scratch")
    p.add_argument("--validate", action="store_true", help="validate after training")
    p.set_defaults(func=cmd_lora)

    p = sub.add_parser(
        "comfy",
        help="drive ComfyUI: ingest/learn/dry-run/run graphs, managed start/stop/status/restart, or opt-in update",
    )
    p.add_argument(
        "comfy_command",
        choices=[
            "run",
            "attach",
            "ingest",
            "learn",
            "dry-run",
            "promote",
            "start",
            "stop",
            "status",
            "restart",
            "update",
        ],
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        help="ingest: API workflow JSON path; learn/dry-run/promote: slug under state/ingested/",
    )
    p.add_argument("--mode", choices=["raw", "template", "generate"], default="generate")
    p.add_argument("--json", dest="workflow_json", help="pasted/path API workflow JSON (raw)")
    p.add_argument("--template", help="template slug or path under workflows/")
    p.add_argument("--variant", default="base")
    p.add_argument(
        "--slug",
        default=None,
        help="ingest: name under state/ingested/ (default: the JSON file stem)",
    )
    p.add_argument(
        "--from",
        dest="ingest_from",
        default=None,
        help="ingest from a live Comfy history entry: history:PROMPT_ID",
    )
    p.add_argument(
        "--no-family-route",
        action="store_true",
        help="ingest/learn/dry-run/run: keep a matched inoutpaint, sulphur, or lipsync graph on the generic path",
    )
    p.add_argument(
        "--variant-name",
        default=None,
        help="promote: draft id in workflows/manifests.yaml (default: the ingested slug). Never a catalog default.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="promote: write the draft even when readiness lists missing nodes or models, or overwrite an existing workflows/<variant>.json",
    )
    p.add_argument(
        "--llm-assist",
        action="store_true",
        help=(
            "ingest/learn: ask a local LLM (llama.cpp, then Ollama) to name "
            "low-confidence widgets. Off by default. No cloud call."
        ),
    )
    p.add_argument(
        "--ingested",
        default=None,
        help="comfy run: queue a learned graph from state/ingested/SLUG. Not a catalog variant.",
    )
    p.add_argument(
        "--queue",
        action="store_true",
        help="ingest: queue after storing. Default is dry-run next and no queue.",
    )
    p.add_argument("--prompt", default="")
    p.add_argument("--negative", dest="negative_prompt", default=None)
    p.add_argument(
        "--scott",
        help="Scott identity still for a blaze remake (uploaded before queue, not committed)",
    )
    p.add_argument(
        "--blaze",
        help="Blaze identity still for a blaze remake (uploaded before queue, not committed)",
    )
    p.add_argument("--video", help="source video; uploaded before queue")
    p.add_argument("--mask", help="inpaint mask (white=regenerate); uploaded before queue")
    p.add_argument("--aspect", help="outpaint target aspect, for example 9:16")
    p.add_argument(
        "--outpaint",
        action="store_true",
        help="extend the source video; requires --aspect or --width and --height",
    )
    p.add_argument("--width", type=int, default=None, action=_DimSet)
    p.add_argument("--height", type=int, default=None, action=_DimSet)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(width_set=False, height_set=False)
    p.add_argument("--set", action="append", default=[], metavar="NODE.FIELD=VALUE",
                   help="expert override, e.g. 12.steps=8 (repeatable)")
    p.add_argument(
        "--vae",
        default=None,
        help=(
            "VAE filename override. A taeltx*/tae* preview VAE is refused "
            "when the graph uses tiled decode."
        ),
    )
    p.add_argument("--out", help="write prepared workflow JSON (still queues unless --prepare)")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="lint and write JSON only; do not queue Comfy",
    )
    p.add_argument(
        "--recipe",
        help=(
            "comfy run: blaze remake id (blaze-concert, blaze-pier). "
            "comfy attach: previs buddy.comfy.attach/v1 / WorkflowPatchPlan JSON"
        ),
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="attach: POST the patched graph to Comfy /prompt (off by default)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="attach: patch + validate only (default; mutually exclusive with --submit)",
    )
    p.add_argument("--runs-dir", dest="runs_dir", help=argparse.SUPPRESS)
    p.add_argument(
        "--workspace",
        default=None,
        help="managed Comfy workspace (start/stop/status/restart); "
        "default MANAGED_COMFY_ROOT or PROJECT_ROOT/ComfyUI",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="managed Comfy listen port (default 8188). Loopback only.",
    )
    p.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="managed health-check URL (default COMFYUI_URL)",
    )
    p.add_argument(
        "--extra-model-paths",
        dest="extra_model_paths",
        default=None,
        help="existing YAML passed to ComfyUI after -- (start/restart). "
        "When omitted, Buddy writes state/extra_model_paths.yaml and passes that. "
        "Does not write into an attached install.",
    )
    p.add_argument(
        "--write-yaml-into-external",
        dest="write_yaml_into_external",
        action="store_true",
        help="status/start/restart: also copy extra_model_paths.yaml into the "
        "attached Comfy root (EXTERNAL_COMFY_ROOT or COMFYUI_ROOT). "
        "Off by default. Never writes weight files.",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="start/restart: return after comfy launch, before /system_stats is ready",
    )
    p.add_argument(
        "--no-watch",
        action="store_true",
        help="start/restart: do not spawn the crash-restart watchdog",
    )
    p.add_argument(
        "--as-json",
        dest="as_json",
        action="store_true",
        help="start/stop/status/restart/update: print JSON (comfy --json remains the workflow path)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="comfy update: opt in. Without this flag the command only reports stale packs.",
    )
    p.add_argument(
        "--core",
        dest="update_core",
        action="store_true",
        help="comfy update: with --yes, update ComfyUI core after a node snapshot",
    )
    p.add_argument(
        "--nodes",
        dest="update_nodes",
        action="store_true",
        help="comfy update: with --yes, update custom nodes after a node snapshot",
    )
    p.add_argument(
        "--cli",
        dest="update_cli",
        action="store_true",
        help="comfy update: with --yes, pip-install the pinned comfy-cli (does not float to latest)",
    )
    p.set_defaults(func=cmd_comfy)

    p = sub.add_parser(
        "diagnose",
        help="9-frame hull fire: prepare + lint, then short queue; prints sec/step",
    )
    p.add_argument("--variant", default="base")
    p.add_argument("--prompt", default="garden proof")
    p.add_argument("--steps", type=int, default=None, help="sampler steps (clamped to 6-8)")
    p.add_argument("--seed", type=int, default=None, help="fixed diagnose seed (default DIAGNOSE_SEED)")
    p.add_argument("--width", type=int, default=None, help="refused above hull until sec/step exists")
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--prepare",
        action="store_true",
        help="lint the 9-frame graph only; do not queue Comfy",
    )
    p.add_argument("--json", action="store_true", help="machine-readable result")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser(
        "rainey1-batch",
        help="Rainey1 generate → junk filter → judge → top-K cut",
        description=(
            "Queue one Rainey1 seed at a time, drop junk under ~100KB or "
            "under 3 frames before the judge, then copy the top-K keepers "
            "to --out with a buddy.clip.provenance/v1 sidecar.\n\n"
            "Recipes load from orchestrator/presets/rainey1.json. "
            "The judge uses judge/prompts/rainey1.md. "
            "Frames are 8n+1 (minimum 9). OOM walks DOWNSCALE_LADDER "
            "downward inside the single-seed orchestrator run.\n\n"
            "--dry-run resolves the recipe and prints the table. It queues "
            "zero Comfy jobs and writes no clips.\n\n"
            "GPU batch is for the operator tower after this draft is merged "
            "and approved.\n\n"
            "Phase 1 (from video_buddy/, Comfy :8188 up, after approve):\n"
            "  python -m master_agent rainey1-batch --recipe lock_open "
            "--seeds 42,43,44,45,46,47,48,49 --top-k 3 "
            "--out outputs/rainey1/phase1/lock_open --no-interview\n"
            "  python -m master_agent rainey1-batch --recipe breach "
            "--seeds 42,43,44,45,46,47 --top-k 3 "
            "--out outputs/rainey1/phase1/breach --no-interview\n"
            "  python -m master_agent rainey1-batch --recipe density "
            "--seeds 42,43,44,45,46,47 --top-k 3 "
            "--out outputs/rainey1/phase1/density --no-interview\n\n"
            "Phase 2 LoRA train YAML is not part of this command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--recipe",
        required=True,
        choices=recipe_choices(),
        help="recipe slug (lock_open, breach, density, myth_16x9, story_9x16)",
    )
    p.add_argument(
        "--seeds",
        required=True,
        help="comma-separated integer seeds, for example 42,43,44,45",
    )
    p.add_argument("--top-k", dest="top_k", type=int, default=2, help="keepers to copy (default 2)")
    p.add_argument(
        "--out",
        default=None,
        help="keeper directory (default outputs/rainey1/topcut/<recipe>)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the table; queue zero Comfy jobs; write no clips",
    )
    p.add_argument(
        "--no-interview",
        action="store_true",
        help="accepted for tower-command compatibility; this subcommand never interviews",
    )
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="exit 0 when every seed drops (dry-run already exits 0)",
    )
    p.add_argument(
        "--llm-judge",
        dest="llm_judge",
        action="store_true",
        help="use the text judge (rainey1 rubric)",
    )
    p.add_argument(
        "--no-llm-judge",
        dest="llm_judge",
        action="store_false",
        help="heuristic look score only; skip the text/vision judge",
    )
    p.set_defaults(func=cmd_rainey1_batch, llm_judge=True)

    p = sub.add_parser("budget", help="render shift budget: status | reset-shift")
    p.add_argument("budget_command", choices=["status", "reset-shift"])
    p.add_argument("--json", action="store_true", help="machine-readable snapshot")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser(
        "agent",
        help="call an agent tool and print JSON (gateway entry point; `agent list` shows tools)",
    )
    p.add_argument("tool", help="tool name from `agent list`, or list")
    p.add_argument(
        "--args",
        default=None,
        help="JSON object of keyword arguments, or @path/to/args.json",
    )
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("hermes", help="Hermes profile ltx: status | register")
    p.add_argument("hermes_command", choices=["status", "register"])
    p.add_argument("--hermes-home", help="override HERMES_HOME / ~/.hermes")
    p.add_argument("--force", action="store_true", help="overwrite custom profiles/ltx/SOUL.md")
    p.add_argument("--json", action="store_true", help="machine-readable status")
    p.set_defaults(func=cmd_hermes)

    p = sub.add_parser(
        "capabilities",
        help="print Comfy capability gap matrix (object_info vs Buddy wiring)",
    )
    p.add_argument("--offline", action="store_true", help="use cached object_info only")
    p.add_argument("--json", action="store_true", help="machine-readable matrix")
    p.set_defaults(func=cmd_capabilities)

    args = parser.parse_args(argv)
    from master_agent.control.versioned_config import announce_config

    # `agent` reserves stdout for its JSON result.
    banner_stream = sys.stderr if args.command == "agent" else sys.stdout
    print(announce_config(), file=banner_stream, flush=True)
    return args.func(args)

