"""Hermes MCP server — exposes VIDEO BUDDY as sub-agent tools.

Run standalone (Hermes spawns it over stdio):
    .venv/Scripts/python.exe master_agent/mcp_server.py

Tools: health, create_video, plan_storyboard, judge_asset,
search_workflows, search_runs, kb_ingest, list_models, validate_workflow,
create_character, train_lora.

NOTE: stdio is the protocol channel — anything the pipeline prints would
corrupt it, so tool bodies redirect stdout to stderr.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

# Allow running as a plain script (no package context / cwd independence)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("master-agent")


def _quiet(fn, *args, **kwargs):
    """Run fn with stdout redirected to stderr (protects the stdio channel)."""
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


@mcp.tool()
def health() -> dict:
    """ComfyUI reachability + GPU VRAM, Ollama status, KB doc counts."""
    from master_agent.comfy.client import ComfyClient
    from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, collection_count
    from master_agent.llm import provider_available

    from master_agent.control.versioned_config import get_versioned_config

    snap = get_versioned_config().snapshot()
    out: dict = {
        "comfyui": None,
        "ollama": provider_available("ollama"),
        "kb": {},
        "config_hash": snap["hash"],
        "config": snap["values"],
    }
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
    }
    return out


@mcp.tool()
def create_video(
    request: str,
    duration_s: float = 5.0,
    quality: str = "draft",
    variant: str | None = None,
    llm_panel: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Generate a video end-to-end (director routing, storyboard panel,
    per-segment judge, stitch, full judge). dry_run=True plans and validates
    without spending GPU. Returns paths, scores and judge notes."""
    if dry_run:
        from master_agent.orchestrator.pipeline import dry_run_pipeline

        code = _quiet(
            dry_run_pipeline,
            request,
            variant=variant,
            duration_s=duration_s,
            quality=quality,
            llm_panel=llm_panel,
        )
        return {"dry_run": True, "ok": code == 0}

    from master_agent.orchestrator.pipeline import run_pipeline

    result = _quiet(
        run_pipeline,
        request,
        variant=variant,
        duration_s=duration_s,
        quality=quality,
        llm_panel=llm_panel,
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
    }


@mcp.tool()
def plan_storyboard(
    request: str,
    duration_s: float = 8.0,
    quality: str = "draft",
    llm_panel: str | None = None,
) -> dict:
    """Plan only: segment split + LLM-panel storyboard (with KB recall).
    No GPU, no validation — fast way to preview what a run would shoot."""
    from master_agent.config import plan_segment_durations
    from master_agent.orchestrator.pipeline import _plan_storyboard

    segs = plan_segment_durations(duration_s, quality=quality)
    logs: list[str] = []
    cards, style, meta = _quiet(
        _plan_storyboard,
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


@mcp.tool()
def judge_asset(video_path: str, request: str = "", full_video: bool = False) -> dict:
    """Grade an existing video file: heuristics + text-LLM + vision legs."""
    from master_agent.judge.judge import judge_full_video, judge_segment
    from master_agent.judge.probe import analyze

    if full_video:
        res = _quiet(judge_full_video, user_request=request, storyboard=None, video_path=video_path)
    else:
        heuristic, issues = analyze(video_path)
        res = _quiet(
            judge_segment,
            user_request=request,
            ltx_prompt=request,
            video_path=video_path,
            heuristic_score=heuristic,
            heuristic_issues=issues,
        )
    return res.to_dict()


@mcp.tool()
def search_workflows(query: str, k: int = 3) -> list[dict]:
    """Semantic search over the workflow template knowledge base."""
    from master_agent.kb.store import COLLECTION_WORKFLOWS, search

    return search(COLLECTION_WORKFLOWS, query, k=k)


@mcp.tool()
def search_runs(query: str, k: int = 3) -> list[dict]:
    """Semantic search over past generation runs (what worked, judge notes)."""
    from master_agent.kb.store import COLLECTION_RUNS, search

    return search(COLLECTION_RUNS, query, k=k)


@mcp.tool()
def kb_ingest() -> dict:
    """Bulk-load workflows + all run records into the knowledge base."""
    from master_agent.kb.ingest import ingest_all_runs, ingest_workflows

    return {"workflows": ingest_workflows(), "runs": ingest_all_runs()}


@mcp.tool()
def list_models() -> dict:
    """Local model inventory summary (LTX weights, bundles, runnable state)."""
    from master_agent.models.inventory import scan_inventory

    inv = _quiet(scan_inventory)
    return {
        "models_found": len(inv.files) if hasattr(inv, "files") else None,
        "bundles": inv.bundles,
        "summary": _quiet(lambda: __import__("master_agent.models.inventory", fromlist=["format_summary"]).format_summary(inv)),
    }


@mcp.tool()
def validate_workflow(path: str) -> dict:
    """Validate a workflow JSON against ComfyUI's node registry + local models."""
    from master_agent.comfy.validator import validate_workflow_file

    report = _quiet(validate_workflow_file, Path(path))
    return report.to_dict()


@mcp.tool()
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

    bible = _quiet(make_character_bible, description)
    if name:
        bible.name = name
    sheet = _quiet(
        generate_character_sheet,
        bible,
        shots=shots or None,
    )
    dataset = _quiet(build_dataset, bible.name)
    out = {
        "bible": bible.to_dict(),
        "hero": sheet.get("hero"),
        "kept": len(sheet.get("kept") or []),
        "scores": sheet.get("scores"),
        "dataset": dataset,
    }
    if train and dataset.get("ok"):
        from master_agent.lora import train_lora

        out["training"] = _quiet(train_lora, bible.name)
    return out


@mcp.tool()
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
    result = _quiet(_train, character_name, overrides=overrides or None)
    if validate and result.get("ok"):
        result["validation"] = _quiet(validate_lora, character_name)
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
    print(f"[mcp_server] warmup {_time.time() - _t:.1f}s", file=sys.stderr)
    try:
        from master_agent.control.versioned_config import announce_config

        print(announce_config(), file=sys.stderr, flush=True)
    except Exception:
        pass
    mcp.run()  # stdio transport
