"""Hermes MCP server — exposes VIDEO BUDDY as sub-agent tools.

Run standalone (Hermes spawns it over stdio):
    .venv/Scripts/python.exe master_agent/mcp_server.py

Tools: health, create_video, plan_storyboard, judge_asset,
search_workflows, search_runs, kb_ingest, list_models, validate_workflow,
create_character, train_lora.

NOTE: stdio is the protocol channel — anything the pipeline prints would
corrupt it. ``__main__`` hands the real stdout to the transport once and
points ``sys.stdout`` at stderr for the rest of the process.

Tools are plain sync functions (importable and callable directly). When
registered with FastMCP they run in a worker thread so a long render does
not block pings or other calls on the event loop.

Set ``LLAMACPP_AUTOSTART=0`` in the gateway's environment to stop this
process from launching its own llama-server (useful when several gateways
each spawn an MCP child).
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

# Allow running as a plain script (no package context / cwd independence)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from mcp.server.fastmcp import FastMCP  # noqa: E402
except ImportError:
    FastMCP = None


class _ToolPassthrough:
    """Call tool functions directly when the mcp package is not installed.

    ``python -m`` / the script entry still need mcp. Unit tests import
    ``create_video`` without starting the stdio server.
    """

    def tool(self):
        def deco(fn):
            return fn

        return deco

    def run(self) -> None:
        raise ImportError(
            "mcp is required to run the master-agent server. "
            "Install video_buddy/requirements.txt."
        )


mcp = FastMCP("master-agent") if FastMCP is not None else _ToolPassthrough()


class _ThreadedTools:
    """Register sync tools with FastMCP behind an async worker-thread wrapper.

    The decorated name stays the original sync function, so tests and other
    callers (e.g. a future single gateway) can call it directly.
    """

    def __init__(self, server):
        self._server = server

    def tool(self):
        def deco(fn):
            if FastMCP is None:
                return fn

            @functools.wraps(fn)
            async def _threaded(*args, **kwargs):
                import anyio

                return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

            self._server.tool()(_threaded)
            return fn

        return deco


tools = _ThreadedTools(mcp)


def run_stdio() -> None:
    """Serve MCP over stdio with the protocol on the real stdout only.

    Everything else that prints (pipeline logs, worker threads, child
    libraries) goes to stderr, for the whole process.
    """
    if FastMCP is None:
        mcp.run()
        return
    import io

    import anyio
    from mcp.server.stdio import stdio_server

    protocol_out = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()), "wb"), encoding="utf-8")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    async def _serve() -> None:
        async with stdio_server(stdout=anyio.wrap_file(protocol_out)) as (read_stream, write_stream):
            server = mcp._mcp_server
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)


@tools.tool()
def health() -> dict:
    """ComfyUI reachability + GPU VRAM, local LLM (llama.cpp, Ollama), opt-in Grok credentials (read-only), KB counts."""
    from master_agent.comfy.client import ComfyClient
    from master_agent.kb.store import (
        COLLECTION_KNOWLEDGE,
        COLLECTION_RUNS,
        COLLECTION_WORKFLOWS,
        collection_count,
    )
    from master_agent.llm import attach_llm_health

    from master_agent.control.versioned_config import get_versioned_config

    snap = get_versioned_config().snapshot()
    out: dict = {
        "comfyui": None,
        "kb": {},
        "config_hash": snap["hash"],
        "config": snap["values"],
    }
    attach_llm_health(out)
    try:
        stats = ComfyClient().health()
        devices = stats.get("devices") or []
        out["comfyui"] = {
            "up": True,
            "gpus": [
                {
                    "name": d.get("name"),
                    "vram_free_gb": round((d.get("vram_free") or 0) / 1e9, 1),
                    "vram_total_gb": round((d.get("vram_total") or 0) / 1e9, 1),
                }
                for d in devices
            ],
        }
    except Exception as e:
        out["comfyui"] = {"up": False, "error": str(e)[:200]}
    out["kb"] = {
        "workflows": collection_count(COLLECTION_WORKFLOWS),
        "runs": collection_count(COLLECTION_RUNS),
        "knowledge": collection_count(COLLECTION_KNOWLEDGE),
    }
    return out


@tools.tool()
def create_video(
    request: str,
    duration_s: float | None = None,
    quality: str = "draft",
    variant: str | None = None,
    llm_panel: str | None = None,
    dry_run: bool = False,
    image_path: str | None = None,
    audio_path: str | None = None,
    video_path: str | None = None,
    line: str | None = None,
) -> dict:
    """Generate a video end-to-end (director routing, storyboard panel,
    per-segment judge, stitch, full judge). dry_run=True plans and validates
    without spending GPU. image_path + audio_path (no video) is a talking clip
    and defaults to ltx25_a2v. Naming MiniMax, Hailuo, H3, or ref2va — or
    passing variant h3_r2v — still routes to h3_r2v.
    H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). For tight lip-sync to an exact recording, use ltx25_a2v.
    Pass line (the exact words) with a 2–12 s audio_path. Under 2 s is rejected
    before queue; over 12 s is trimmed. That warning is returned on the response
    when the route is h3_r2v. A missing line sets line_warning and does not fail.
    Returns paths, scores and judge notes."""
    from pathlib import Path

    for label, raw in (
        ("image", image_path),
        ("audio", audio_path),
        ("video", video_path),
    ):
        if raw and not Path(raw).is_file():
            return {"status": "error", "error": f"{label} not found: {raw}"}

    note = None
    if duration_s is None:
        if image_path and audio_path and not video_path:
            from master_agent.orchestrator.talking import duration_following_audio

            duration_s, note = duration_following_audio(audio_path)
        else:
            duration_s = 5.0
    image_name = Path(image_path).name if image_path else None
    audio_name = Path(audio_path).name if audio_path else None
    video_name = Path(video_path).name if video_path else None
    from master_agent.orchestrator.talking import h3_r2v_audio_warning

    voice_warn = h3_r2v_audio_warning(
        request,
        variant=variant,
        has_image=bool(image_path),
        has_audio=bool(audio_path),
        has_video=bool(video_path),
    )
    spoken_line = (line or "").strip()
    voice_sample = None
    line_warning = None
    from master_agent.orchestrator.talking import is_h3_voice_route

    h3_voice = is_h3_voice_route(
        request,
        variant=variant,
        has_image=bool(image_path),
        has_audio=bool(audio_path),
        has_video=bool(video_path),
    )
    if h3_voice and audio_path and not dry_run:
        from master_agent.orchestrator.h3_voice import VoiceSampleError, h3_voice_preflight

        try:
            pre = h3_voice_preflight(
                request=request,
                variant=variant,
                audio_path=audio_path,
                has_image=bool(image_path),
                has_video=bool(video_path),
                line=spoken_line or None,
            )
        except VoiceSampleError as exc:
            return {
                "status": "error",
                "error": str(exc),
                "warning": voice_warn,
                "notes": voice_warn,
            }
        audio_path = pre.audio_path
        if audio_path:
            audio_name = Path(audio_path).name
        voice_sample = pre.voice_sample
        spoken_line = pre.spoken_line or ""
        line_warning = pre.line_warning
        if pre.trimmed_note:
            note = f"{note}; {pre.trimmed_note}" if note else pre.trimmed_note
    elif h3_voice:
        from master_agent.orchestrator.h3_voice import (
            h3_missing_line_warning,
            resolve_spoken_line,
        )

        spoken_line = resolve_spoken_line(spoken_line or None, request, None)
        line_warning = h3_missing_line_warning(spoken_line, h3_voice=True)

    if dry_run:
        from master_agent.orchestrator.pipeline import dry_run_pipeline

        code = dry_run_pipeline(
            request,
            variant=variant,
            duration_s=float(duration_s),
            quality=quality,
            llm_panel=llm_panel,
            image_name=image_name,
            audio_name=audio_name,
            video_name=video_name,
            spoken_line=spoken_line or None,
        )
        return {
            "dry_run": True,
            "ok": code == 0,
            "status": "dry-run" if code == 0 else "error",
            "error": None if code == 0 else "dry-run failed",
            "audio_note": note,
            "warning": voice_warn,
            "notes": voice_warn,
            "line_warning": line_warning,
            "spoken_line": spoken_line or None,
        }

    from master_agent.comfy.client import ComfyClient, ComfyClientError
    from master_agent.orchestrator.pipeline import run_pipeline

    client = ComfyClient()
    try:
        if image_path:
            image_name = client.upload_image(Path(image_path))
        if audio_path:
            audio_name = client.upload_audio(Path(audio_path))
        if video_path:
            video_name = client.upload_image(Path(video_path))
    except (ComfyClientError, OSError) as exc:
        return {"status": "error", "error": str(exc), "warning": voice_warn, "notes": voice_warn}

    result = run_pipeline(
        request,
        variant=variant,
        duration_s=float(duration_s),
        quality=quality,
        llm_panel=llm_panel,
        image_name=image_name,
        audio_name=audio_name,
        audio_path=audio_path,
        video_name=video_name,
        spoken_line=spoken_line or None,
        voice_sample=voice_sample,
        client=client,
    )
    return {
        "status": result.status,
        "video_path": result.video_path,
        "segment_paths": result.segment_paths,
        "segment_scores": result.segment_scores,
        "full_judge_score": result.full_judge_score,
        "full_judge_pass": result.full_judge_pass,
        "full_judge_notes": result.full_judge_notes,
        "storyboard": result.storyboard,
        "panel_meta": result.panel_meta,
        "error": result.error,
        "audio_note": note,
        "warning": voice_warn,
        "notes": voice_warn,
        "line_warning": line_warning,
        "spoken_line": spoken_line or None,
    }


@tools.tool()
def plan_storyboard(
    request: str,
    duration_s: float = 8.0,
    quality: str = "draft",
    llm_panel: str | None = None,
) -> dict:
    """Plan only: segment split + LLM-panel storyboard (with KB recall).
    No GPU, no validation — fast way to preview what a run would shoot."""
    from master_agent.orchestrator.pipeline import _plan_storyboard, plan_story_segments

    segs = plan_story_segments(duration_s, quality=quality)
    logs: list[str] = []
    cards, style, meta = _plan_storyboard(
        request,
        segs,
        variant=None,
        quality=quality,
        llm_panel=llm_panel,
        panel_judge=None,
        log=logs.append,
    )
    return {
        "segments": segs,
        "global_style": style,
        "shots": [c.to_dict() for c in cards],
        "panel_meta": meta,
        "log": logs,
    }


@tools.tool()
def judge_asset(video_path: str, request: str = "", full_video: bool = False) -> dict:
    """Grade an existing video file: heuristics + text-LLM + vision legs."""
    from master_agent.judge.judge import judge_full_video, judge_segment
    from master_agent.judge.probe import analyze

    if full_video:
        res = judge_full_video(user_request=request, storyboard=None, video_path=video_path)
    else:
        heuristic, issues = analyze(video_path)
        res = judge_segment(
            user_request=request,
            ltx_prompt=request,
            video_path=video_path,
            heuristic_score=heuristic,
            heuristic_issues=issues,
        )
    return res.to_dict()


@tools.tool()
def search_workflows(query: str, k: int = 3) -> list[dict]:
    """Semantic search over the workflow template knowledge base."""
    from master_agent.kb.store import COLLECTION_WORKFLOWS, search

    return search(COLLECTION_WORKFLOWS, query, k=k)


@tools.tool()
def search_runs(query: str, k: int = 3) -> list[dict]:
    """Semantic search over past generation runs (what worked, judge notes)."""
    from master_agent.kb.store import COLLECTION_RUNS, search

    return search(COLLECTION_RUNS, query, k=k)


@tools.tool()
def kb_ingest() -> dict:
    """Bulk-load workflows, run records, and git-synced knowledge/ markdown."""
    from master_agent.kb.ingest import ingest_all_runs, ingest_knowledge, ingest_workflows

    knowledge = ingest_knowledge()
    return {
        "workflows": ingest_workflows(),
        "runs": ingest_all_runs(),
        "knowledge": knowledge.docs,
        "knowledge_skipped": knowledge.skipped,
    }


@tools.tool()
def list_models() -> dict:
    """Local model inventory summary (LTX weights, bundles, runnable state)."""
    from master_agent.models.inventory import format_summary, scan_inventory

    inv = scan_inventory()
    return {
        "models_found": len(inv.files) if hasattr(inv, "files") else None,
        "bundles": inv.bundles,
        "summary": format_summary(inv),
    }


@tools.tool()
def validate_workflow(path: str) -> dict:
    """Validate a workflow JSON against ComfyUI's node registry + local models."""
    from master_agent.comfy.validator import validate_workflow_file

    report = validate_workflow_file(Path(path))
    return report.to_dict()


@tools.tool()
def create_character(
    description: str,
    name: str = "",
    shots: int = 0,
    train: bool = False,
) -> dict:
    """CCC stage: character bible -> Flux character sheet (vision-curated for
    identity consistency) -> captioned LoRA dataset. Set train=True to also
    kick off Flux LoRA training (hours on 16GB). Returns bible, sheet scores
    and dataset summary."""
    from master_agent.character import (
        build_dataset,
        generate_character_sheet,
        make_character_bible,
    )

    bible = make_character_bible(description)
    if name:
        bible.name = name
    sheet = generate_character_sheet(bible, shots=shots or None)
    dataset = build_dataset(bible.name)
    out = {
        "bible": bible.to_dict(),
        "hero": sheet.get("hero"),
        "kept": len(sheet.get("kept") or []),
        "scores": sheet.get("scores"),
        "dataset": dataset,
    }
    if train and dataset.get("ok"):
        from master_agent.lora import train_lora

        out["training"] = train_lora(bible.name)
    return out


@tools.tool()
def train_lora(
    character_name: str,
    steps: int = 0,
    lr: float = 0.0,
    rank: int = 0,
    validate: bool = True,
) -> dict:
    """Train a Flux LoRA for an existing character (ai-toolkit, resumable).
    Long-running (hours). When validate=True, scores the trained LoRA on
    held-out scenes against the character's hero image afterwards."""
    from master_agent.lora import train_lora as _train
    from master_agent.lora import validate_lora

    overrides = {
        k: v
        for k, v in (("steps", steps), ("lr", lr), ("rank", rank))
        if v
    }
    result = _train(character_name, overrides=overrides or None)
    if validate and result.get("ok"):
        result["validation"] = validate_lora(character_name)
    return result


if __name__ == "__main__":
    import time as _time

    _t = _time.time()
    # Warm heavy native deps on the MAIN thread before the event loop starts.
    # Loading them later inside anyio worker threads stalls ~30s per DLL on
    # this machine (240s+ per tool call). Main-thread import is ~1s.
    try:
        import numpy  # noqa: F401
        import chromadb  # noqa: F401

        from master_agent.kb.store import get_client

        get_client()
    except Exception as e:
        print(f"[mcp_server] warmup failed (tools will fall back): {e}", file=sys.stderr)
    # Tool bodies run in worker threads, so import their modules here too.
    for _mod in ("master_agent.orchestrator.pipeline", "master_agent.judge.judge"):
        try:
            __import__(_mod)
        except Exception as e:
            print(f"[mcp_server] warmup import {_mod} failed: {e}", file=sys.stderr)
    print(f"[mcp_server] warmup {_time.time() - _t:.1f}s", file=sys.stderr)
    try:
        from master_agent.control.versioned_config import announce_config

        print(announce_config(), file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        from master_agent.kb.ingest import schedule_knowledge_ingest

        schedule_knowledge_ingest(stderr=True)
    except Exception as e:
        print(f"[mcp_server] knowledge ingest not scheduled: {e}", file=sys.stderr)
    try:
        from master_agent.llm import prepare_local_llm

        prepare_local_llm()
    except Exception as e:
        print(f"[mcp_server] llama.cpp not started: {e}", file=sys.stderr)
    run_stdio()
