"""Importable agent API — every capability the LTX bot gateway calls.

Plain sync functions returning JSON-safe dicts/lists. No server, no
transport: the gateway imports these or runs ``python -m master_agent
agent <tool> --args '<json>'`` (see ``cli/agent.py``), which prints the
return value as JSON on stdout.

GPU renders take a cross-process lock on ``STATE_DIR/gpu`` so two callers
(two gateway turns, a CLI run started by hand) never queue the card twice.
A busy GPU returns ``status="busy"`` immediately instead of blocking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

UPSCALE_METHODS = (None, "rtx", "seedvr2")
STORYBOARD_MODES = (None, "smart", "always", "multi_only", "off")
QUALITIES = ("draft", "balanced", "quality")
CONTROL_KEYS = (
    "judge_strictness",
    "learning_rate",
    "cost_vram_threshold_gb",
    "render_budget_cap_vram_min",
    "judge_score_threshold",
    "persona",
    "soul",
)


def _gpu_lock_path() -> Path:
    from master_agent.config import STATE_DIR

    return STATE_DIR / "gpu"


def about() -> dict:
    """VIDEO BUDDY identity card."""
    from master_agent.about import studio_about

    return studio_about()


def health() -> dict:
    """ComfyUI reachability + GPU VRAM, local LLM (llama.cpp, Ollama), opt-in Grok credentials (read-only), KB counts."""
    from master_agent.comfy.client import ComfyClient
    from master_agent.control.versioned_config import get_versioned_config
    from master_agent.kb.store import (
        COLLECTION_KNOWLEDGE,
        COLLECTION_RUNS,
        COLLECTION_WORKFLOWS,
        collection_count,
    )
    from master_agent.llm import attach_llm_health

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
    seed: int | None = None,
    storyboard: str | None = None,
    upscale: str | None = None,
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
    seed fixes the sampler seed; storyboard is smart|always|multi_only|off;
    upscale (rtx|seedvr2) post-processes the final video.
    Returns paths, scores and judge notes. status="busy" means another
    render holds the GPU; nothing was queued."""
    if not (request or "").strip():
        return {"status": "error", "error": "request must not be empty"}
    if quality not in QUALITIES:
        return {"status": "error", "error": "quality must be draft|balanced|quality"}
    if upscale not in UPSCALE_METHODS:
        return {"status": "error", "error": "upscale must be rtx|seedvr2"}
    if storyboard not in STORYBOARD_MODES:
        return {"status": "error", "error": "storyboard must be smart|always|multi_only|off"}

    from master_agent.media_paths import media_path_error

    for label, raw in (
        ("image", image_path),
        ("audio", audio_path),
        ("video", video_path),
    ):
        fenced = media_path_error(label, raw)
        if fenced:
            return {"status": "error", "error": fenced}
        if raw and not Path(raw).is_file():
            return {"status": "error", "error": f"{label} not found: {raw}"}

    from master_agent.orchestrator.talking import (
        h3_r2v_audio_warning,
        is_h3_voice_route,
        media_route_error,
        preview_media_variant,
    )

    media = {
        "has_image": bool(image_path),
        "has_audio": bool(audio_path),
        "has_video": bool(video_path),
    }
    route_err = media_route_error(preview_media_variant(request, variant=variant, **media), **media)
    if route_err:
        return {"status": "error", "error": route_err}

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

    voice_warn = h3_r2v_audio_warning(request, variant=variant, **media)
    spoken_line = (line or "").strip()
    voice_sample = None
    line_warning = None
    h3_voice = is_h3_voice_route(request, variant=variant, **media)
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

    from master_agent.fileutil import file_lock

    with file_lock(_gpu_lock_path(), blocking=False) as held:
        if not held:
            return {
                "status": "busy",
                "error": "another render holds the GPU; try again when it finishes",
                "warning": voice_warn,
                "notes": voice_warn,
            }
        return _render(
            request,
            variant=variant,
            duration_s=float(duration_s),
            quality=quality,
            llm_panel=llm_panel,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            spoken_line=spoken_line or None,
            voice_sample=voice_sample,
            seed=seed,
            storyboard=storyboard,
            upscale=upscale,
            extra={
                "audio_note": note,
                "warning": voice_warn,
                "notes": voice_warn,
                "line_warning": line_warning,
                "spoken_line": spoken_line or None,
            },
        )


def _render(
    request: str,
    *,
    variant: str | None,
    duration_s: float,
    quality: str,
    llm_panel: str | None,
    image_path: str | None,
    audio_path: str | None,
    video_path: str | None,
    spoken_line: str | None,
    voice_sample: dict | None,
    seed: int | None,
    storyboard: str | None,
    upscale: str | None,
    extra: dict,
) -> dict:
    from master_agent.comfy.client import ComfyClient, ComfyClientError
    from master_agent.orchestrator.pipeline import run_pipeline

    client = ComfyClient()
    image_name = audio_name = video_name = None
    try:
        if image_path:
            image_name = client.upload_image(Path(image_path))
        if audio_path:
            audio_name = client.upload_audio(Path(audio_path))
        if video_path:
            video_name = client.upload_image(Path(video_path))
    except (ComfyClientError, OSError) as exc:
        return {"status": "error", "error": str(exc), **{k: extra[k] for k in ("warning", "notes")}}

    result = run_pipeline(
        request,
        variant=variant,
        duration_s=duration_s,
        quality=quality,
        seed=seed,
        llm_panel=llm_panel,
        image_name=image_name,
        audio_name=audio_name,
        audio_path=audio_path,
        video_name=video_name,
        spoken_line=spoken_line,
        voice_sample=voice_sample,
        storyboard_mode=storyboard,
        client=client,
    )
    out = {
        "status": result.status,
        "run_id": getattr(result, "run_id", None),
        "video_path": result.video_path,
        "segment_paths": result.segment_paths,
        "segment_scores": result.segment_scores,
        "full_judge_score": result.full_judge_score,
        "full_judge_pass": result.full_judge_pass,
        "full_judge_notes": result.full_judge_notes,
        "storyboard": result.storyboard,
        "panel_meta": result.panel_meta,
        "error": result.error,
        **extra,
    }
    if upscale and result.video_path and result.status not in ("error", "paused"):
        from master_agent.upscale import upscale_video

        out["upscaled_path"] = str(
            upscale_video(result.video_path, method=upscale, run_id=getattr(result, "run_id", None))
        )
    return out


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


def judge_asset(video_path: str, request: str = "", full_video: bool = False) -> dict:
    """Grade an existing video file: heuristics + text-LLM + vision legs."""
    from master_agent.judge.judge import judge_full_video, judge_segment
    from master_agent.judge.probe import analyze
    from master_agent.media_paths import media_path_error

    fenced = media_path_error("video_path", video_path)
    if fenced:
        return {"status": "error", "error": fenced}

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


def search_workflows(query: str, k: int = 3) -> list[dict]:
    """Semantic search over the workflow template knowledge base."""
    from master_agent.kb.store import COLLECTION_WORKFLOWS, search

    return search(COLLECTION_WORKFLOWS, query, k=k)


def search_runs(query: str, k: int = 3) -> list[dict]:
    """Semantic search over past generation runs (what worked, judge notes)."""
    from master_agent.kb.store import COLLECTION_RUNS, search

    return search(COLLECTION_RUNS, query, k=k)


def search_knowledge(query: str, k: int = 3) -> list[dict]:
    """Semantic search over the git-synced knowledge/ markdown collection."""
    from master_agent.kb.store import COLLECTION_KNOWLEDGE, search

    return search(COLLECTION_KNOWLEDGE, query, k=k)


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


def list_runs(limit: int = 50) -> list[dict]:
    """Newest run records first: request, variant, status, judge score, video path."""
    from master_agent.config import RUNS_DIR

    limit = max(1, min(int(limit), 200))
    records = sorted(RUNS_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]
    out = []
    for path in records:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_pipeline = d.get("segment_paths") is not None
        out.append(
            {
                "file": path.name,
                "ts": path.stem.split("_")[0],
                "kind": d.get("kind") or ("pipeline" if is_pipeline else "run"),
                "run_id": d.get("run_id"),
                "request": d.get("request") or "",
                "variant": d.get("variant") or "",
                "status": d.get("status") or d.get("state") or "",
                "score": d.get("full_judge_score", d.get("judge_score")),
                "passed": bool(d.get("full_judge_pass") or d.get("judge_decision") == "accept"),
                "judge_reason": (d.get("full_judge_notes") or d.get("judge_reason") or "")[:300],
                "video_path": d.get("upscaled_path") or d.get("video_path") or None,
                "duration_s": sum(d.get("segment_durations") or []) or d.get("duration_s"),
            }
        )
    return out


def list_models() -> dict:
    """Local model inventory summary (LTX weights, bundles, runnable state)."""
    from master_agent.models.inventory import format_summary, scan_inventory

    inv = scan_inventory()
    files = list(getattr(inv, "files", None) or [])
    return {
        "models_found": len(files) if hasattr(inv, "files") else None,
        "bundles": inv.bundles,
        "files": [
            {"name": f.name, "size_gb": round(f.stat().st_size / 1e9, 2), "dir": f.parent.name}
            for f in files
            if f.is_file()
        ],
        "summary": format_summary(inv),
    }


def validate_workflow(path: str) -> dict:
    """Validate a workflow JSON against ComfyUI's node registry + local models."""
    from master_agent.comfy.validator import validate_workflow_file

    report = validate_workflow_file(Path(path))
    return report.to_dict()


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
        from master_agent.lora import train_lora as _train

        out["training"] = _train(bible.name)
    return out


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

    overrides = {k: v for k, v in (("steps", steps), ("lr", lr), ("rank", rank)) if v}
    result = _train(character_name, overrides=overrides or None)
    if validate and result.get("ok"):
        result["validation"] = validate_lora(character_name)
    return result


def control_get() -> dict:
    """Studio knobs (judge strictness/threshold, learning rate, VRAM cost gate,
    render budget cap/used, persona, soul) plus recent config history."""
    from master_agent.control.budget import get_project_budget
    from master_agent.control.versioned_config import get_versioned_config
    from master_agent.persona.persona import list_personas
    from master_agent.persona.soul import list_souls

    store = get_versioned_config()
    snap = store.snapshot()
    budget = get_project_budget().snapshot()
    values = snap["values"]
    return {
        "hash": snap["hash"],
        "judge_strictness": values.get("judge_strictness"),
        "learning_rate": values.get("learning_rate"),
        "cost_vram_threshold_gb": values.get("cost_vram_threshold_gb"),
        "judge_score_threshold": values.get("judge_score_threshold"),
        "render_budget_cap_vram_min": budget["cap"],
        "render_budget_used_vram_min": budget["used"],
        "persona": str(values.get("persona") or "ara"),
        "soul": str(values.get("soul") or "studio"),
        "personas": [{"slug": p.slug, "name": p.name} for p in list_personas()],
        "souls": [{"slug": s.slug, "name": s.name} for s in list_souls()],
        "budget": budget,
        "history": store.history(10),
    }


def control_set(
    judge_strictness: float | None = None,
    learning_rate: float | None = None,
    cost_vram_threshold_gb: float | None = None,
    render_budget_cap_vram_min: float | None = None,
    judge_score_threshold: float | None = None,
    persona: str | None = None,
    soul: str | None = None,
    reset_budget: bool = False,
    session: str = "agent",
) -> dict:
    """Change studio knobs (versioned, hash-stamped). reset_budget=True zeroes
    used VRAM-minutes and resumes generates that paused on the budget."""
    from master_agent.control.budget import get_project_budget
    from master_agent.control.versioned_config import get_versioned_config

    candidates = {
        "judge_strictness": judge_strictness,
        "learning_rate": learning_rate,
        "cost_vram_threshold_gb": cost_vram_threshold_gb,
        "render_budget_cap_vram_min": render_budget_cap_vram_min,
        "judge_score_threshold": judge_score_threshold,
        "persona": persona,
        "soul": soul,
    }
    updates = {k: v for k, v in candidates.items() if v is not None}
    store = get_versioned_config()
    if updates:
        store.set_values(updates, session=(session or "agent").strip() or "agent", sync_budget=True)
    resumed: list[str] = []
    if reset_budget:
        get_project_budget().reset_used()
        store.sync_used(0.0)
        from master_agent.orchestrator.pipeline import resume_after_budget_clear

        resumed = resume_after_budget_clear(background=False)
    payload = control_get()
    if resumed:
        payload["resumed"] = resumed
    return payload


def budget_status() -> dict:
    """Render shift budget snapshot: used, cap, paused, shift id, pending, ledger."""
    from master_agent.control.budget import get_project_budget

    return get_project_budget().snapshot()


def budget_reset_shift() -> dict:
    """Archive the shift ledger, zero used, and resume budget-paused generates."""
    from master_agent.control.budget import get_project_budget
    from master_agent.orchestrator.pipeline import resume_after_budget_clear

    store = get_project_budget()
    row = store.reset_shift()
    resumed = resume_after_budget_clear(background=False)
    return {"reset": row, "resumed": resumed, "budget": store.snapshot()}


TOOLS: dict[str, Callable[..., Any]] = {
    fn.__name__: fn
    for fn in (
        about,
        health,
        create_video,
        plan_storyboard,
        judge_asset,
        search_workflows,
        search_runs,
        search_knowledge,
        kb_ingest,
        list_runs,
        list_models,
        validate_workflow,
        create_character,
        train_lora,
        control_get,
        control_set,
        budget_status,
        budget_reset_shift,
    )
}


def call(tool: str, args: Optional[dict[str, Any]] = None) -> Any:
    """Dispatch ``tool`` with keyword ``args``. Raises KeyError for unknown tools."""
    fn = TOOLS[tool]
    return fn(**(args or {}))
