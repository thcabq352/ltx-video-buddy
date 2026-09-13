"""FastAPI app — the VIDEO BUDDY web dashboard.

Run: python -m master_agent ui --port 8189
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from master_agent.config import OUTPUTS_DIR, RUNS_DIR
from master_agent.web.jobs import MANAGER

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from master_agent.control.versioned_config import announce_config

    print(announce_config(), flush=True)
    yield


app = FastAPI(title="VIDEO BUDDY", docs_url=None, redoc_url=None, lifespan=_lifespan)


# ── models ──────────────────────────────────────────────


class JobRequest(BaseModel):
    request: str
    duration_s: float = 8.0
    quality: str = "draft"
    variant: Optional[str] = None
    llm_panel: Optional[str] = None
    dry_run: bool = False
    # Local paths under state/uploads/ (from /api/upload); uploaded into ComfyUI on run
    image_path: Optional[str] = None  # start frame for i2v
    audio_path: Optional[str] = None  # lipsync / audio conditioning
    video_path: Optional[str] = None  # source video (lipsync)
    upscale: Optional[str] = None
    seed: Optional[int] = None
    storyboard: Optional[str] = None


class JudgeRequest(BaseModel):
    video_path: str
    request: str = ""
    full_video: bool = False


class FractalRequest(BaseModel):
    request: str = ""
    mode: str = "zoom"  # zoom | inpaint | outpaint
    duration_s: float = 20.0
    fps: int = 24
    width: int = 768
    height: int = 512
    target: str = "seahorse"
    palette: str = "fire"
    julia: bool = False
    seed: Optional[int] = None
    audio_path: Optional[str] = None
    image_path: Optional[str] = None  # required for inpaint/outpaint
    mask_path: Optional[str] = None  # optional inpaint mask (white=fill)
    expand: int = 128  # outpaint border px
    cover: float = 0.4  # inpaint center-hole fraction
    feather: int = 28
    upscale: Optional[str] = None


class MusicRequest(BaseModel):
    request: str
    audio_path: str
    visual: str = "shots"
    quality: str = "draft"
    seed: Optional[int] = None
    upscale: Optional[str] = None


class IntakeRequest(BaseModel):
    session_id: Optional[str] = None
    request: Optional[str] = None  # initial idea (starts a session)
    message: Optional[str] = None  # subsequent user reply


class PowerTuneRequest(BaseModel):
    request: str
    variant: str = "base"
    quality: str = "draft"
    duration_s: float = 5.0
    seed: Optional[int] = None
    provider: Optional[str] = None


class ComfyPrepareRequest(BaseModel):
    mode: str = "template"
    workflow: Optional[dict[str, Any]] = None
    template: Optional[str] = None
    variant: Optional[str] = None
    prompt: str = ""
    overrides: Optional[dict[str, dict[str, Any]]] = None


class ComfyWorkflowBody(BaseModel):
    workflow: dict[str, Any]
    request: str = "comfy run"


# ── pages & media ───────────────────────────────────────


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/videos/{rel_path:path}")
def videos(rel_path: str):
    """Serve generated videos for playback (path-guarded to OUTPUTS_DIR)."""
    base = OUTPUTS_DIR.resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(404, "video not found")
    if not target.is_file():
        raise HTTPException(404, "video not found")
    return FileResponse(target)


# ── api ─────────────────────────────────────────────────


@app.get("/api/about")
def api_about() -> dict[str, Any]:
    from master_agent.about import studio_about

    return studio_about()


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    from master_agent.comfy.client import ComfyClient
    from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, collection_count
    from master_agent.llm import provider_available

    out: dict[str, Any] = {
        "ollama": provider_available("ollama"),
        "grok": provider_available("grok"),
        "kb": {
            "workflows": collection_count(COLLECTION_WORKFLOWS),
            "runs": collection_count(COLLECTION_RUNS),
        },
    }
    try:
        from master_agent.control.versioned_config import get_versioned_config

        out["config_hash"] = get_versioned_config().snapshot()["hash"]
    except Exception:
        pass
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
    return out


_FRACTAL_TARGETS = {"seahorse", "elephant", "minibrot", "spiral"}
_FRACTAL_PALETTES = {"fire", "ocean", "monochrome", "neon", "sunset"}
_UPSCALE_METHODS = {None, "rtx", "seedvr2"}
_STORYBOARD_MODES = {None, "smart", "always", "multi_only", "off"}


def _require_upload_path(path: Optional[str], label: str) -> Optional[str]:
    """Ensure optional media path exists and lives under state/uploads/."""
    if not path:
        return None
    from master_agent.config import STATE_DIR

    p = Path(path).resolve()
    uploads = (STATE_DIR / "uploads").resolve()
    try:
        p.relative_to(uploads)
    except ValueError:
        raise HTTPException(400, f"{label} must be an uploaded file under state/uploads/")
    if not p.is_file():
        raise HTTPException(400, f"{label} not found: {path}")
    return str(p)


@app.post("/api/jobs")
def api_submit_job(req: JobRequest):
    if not req.request.strip():
        raise HTTPException(400, "request must not be empty")
    if req.quality not in ("draft", "balanced", "quality"):
        raise HTTPException(400, "quality must be draft|balanced|quality")
    if req.variant not in (None, "", "auto"):
        from master_agent.comfy.catalog import is_known_variant

        if not is_known_variant(req.variant):
            raise HTTPException(400, f"unknown variant: {req.variant}")
    if req.upscale not in _UPSCALE_METHODS:
        raise HTTPException(400, "upscale must be rtx|seedvr2")
    if req.storyboard not in _STORYBOARD_MODES:
        raise HTTPException(400, "storyboard must be smart|always|multi_only|off")
    image_path = _require_upload_path(req.image_path, "image")
    audio_path = _require_upload_path(req.audio_path, "audio")
    video_path = _require_upload_path(req.video_path, "video")
    kind = "dry-run" if req.dry_run else "run"
    job = MANAGER.submit(
        kind,
        req.request.strip(),
        duration_s=req.duration_s,
        quality=req.quality,
        variant=req.variant,
        llm_panel=req.llm_panel,
        image_path=image_path,
        audio_path=audio_path,
        video_path=video_path,
        upscale=req.upscale,
        seed=req.seed,
        storyboard=req.storyboard,
    )
    return job.to_dict()


@app.get("/api/jobs")
def api_list_jobs():
    return [j.to_dict() for j in MANAGER.list()]


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = MANAGER.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_dict(with_log=True)


_FRACTAL_MODES = {"zoom", "inpaint", "outpaint"}


@app.post("/api/fractal")
def api_fractal(req: FractalRequest):
    mode = (req.mode or "zoom").strip().lower()
    if mode not in _FRACTAL_MODES:
        raise HTTPException(400, f"mode must be one of {sorted(_FRACTAL_MODES)}")
    if req.target not in _FRACTAL_TARGETS:
        raise HTTPException(400, f"target must be one of {sorted(_FRACTAL_TARGETS)}")
    if req.palette not in _FRACTAL_PALETTES:
        raise HTTPException(400, f"palette must be one of {sorted(_FRACTAL_PALETTES)}")
    if req.upscale not in _UPSCALE_METHODS:
        raise HTTPException(400, "upscale must be rtx|seedvr2")
    if mode in ("inpaint", "outpaint") and not req.image_path:
        raise HTTPException(400, f"mode={mode} requires image_path (upload an image first)")
    if req.duration_s <= 0 or req.duration_s > 600:
        raise HTTPException(400, "duration_s must be in (0, 600]")
    if req.expand < 0 or req.expand > 1024:
        raise HTTPException(400, "expand must be 0..1024")
    if not (0.05 <= req.cover <= 0.95):
        raise HTTPException(400, "cover must be 0.05..0.95")
    if req.feather < 0 or req.feather > 256:
        raise HTTPException(400, "feather must be 0..256")
    audio_path = _require_upload_path(req.audio_path, "audio") if req.audio_path else None
    image_path = _require_upload_path(req.image_path, "image") if req.image_path else None
    mask_path = _require_upload_path(req.mask_path, "mask") if req.mask_path else None
    job = MANAGER.submit(
        "fractal",
        req.request.strip() or f"fractal {mode} {req.target}/{req.palette}",
        mode=mode,
        duration_s=req.duration_s,
        fps=req.fps,
        width=req.width,
        height=req.height,
        target=req.target,
        palette=req.palette,
        julia=req.julia,
        seed=req.seed,
        audio_path=audio_path,
        image_path=image_path,
        mask_path=mask_path,
        expand=req.expand,
        cover=req.cover,
        feather=req.feather,
        upscale=req.upscale,
    )
    return job.to_dict()


@app.post("/api/music")
def api_music(req: MusicRequest):
    if not req.request.strip():
        raise HTTPException(400, "request must not be empty")
    if req.visual not in ("shots", "fractal"):
        raise HTTPException(400, "visual must be shots|fractal")
    if req.quality not in ("draft", "balanced", "quality"):
        raise HTTPException(400, "quality must be draft|balanced|quality")
    if req.upscale not in _UPSCALE_METHODS:
        raise HTTPException(400, "upscale must be rtx|seedvr2")
    if not (req.audio_path or "").strip():
        raise HTTPException(400, "audio_path required")
    audio_path = _require_upload_path(req.audio_path, "audio")
    job = MANAGER.submit(
        "music",
        req.request.strip(),
        audio_path=audio_path,
        visual=req.visual,
        quality=req.quality,
        seed=req.seed,
        upscale=req.upscale,
    )
    return job.to_dict()


# Max single upload size (images / audio / video for ComfyUI inputs)
_UPLOAD_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Save an uploaded media file under state/uploads/ (images, audio, video)."""
    from master_agent.config import STATE_DIR
    import uuid as _uuid

    uploads = STATE_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.bin").name
    # strip path tricks
    safe_name = safe_name.replace("\\", "/").split("/")[-1] or "upload.bin"
    dest = uploads / f"{_uuid.uuid4().hex[:8]}_{safe_name}"
    written = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > _UPLOAD_MAX_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file too large (max {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB)")
            f.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    return {
        "path": str(dest),
        "name": safe_name,
        "size_bytes": written,
        "content_type": file.content_type or "",
    }


# ── intake interview (persona chat) ─────────────────────

_intake_sessions: dict[str, Any] = {}


@app.get("/api/persona")
def api_persona():
    import master_agent.config as cfg
    from master_agent.persona.persona import list_personas, load_persona

    p = load_persona()
    return {
        "slug": p.slug,
        "name": p.name,
        "active": cfg.PERSONA,
        "items": [
            {"slug": i.slug, "name": i.name, "active": i.slug == cfg.PERSONA}
            for i in list_personas()
        ],
    }


@app.post("/api/persona")
def api_persona_set(payload: dict[str, Any]):
    from master_agent.persona.persona import set_active_persona

    slug = str(payload.get("slug") or "").strip()
    if not slug:
        raise HTTPException(400, "slug required")
    try:
        p = set_active_persona(slug, session=str(payload.get("session") or "studio"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return api_persona()


@app.get("/api/soul")
def api_soul():
    import master_agent.config as cfg
    from master_agent.persona.soul import list_souls, load_soul

    s = load_soul()
    return {
        "slug": s.slug,
        "name": s.name,
        "active": cfg.SOUL,
        "items": [
            {"slug": i.slug, "name": i.name, "active": i.slug == cfg.SOUL}
            for i in list_souls()
        ],
    }


@app.post("/api/soul")
def api_soul_set(payload: dict[str, Any]):
    from master_agent.persona.soul import set_active_soul

    slug = str(payload.get("slug") or "").strip()
    if not slug:
        raise HTTPException(400, "slug required")
    try:
        s = set_active_soul(slug, session=str(payload.get("session") or "studio"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return api_soul()


@app.post("/api/power-tune")
def api_power_tune(req: PowerTuneRequest):
    """Dry-run agent power mode: patch + LLM graph ops + validate (no GPU)."""
    from master_agent.comfy.power_mode import power_tune
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow
    from master_agent.config import get_quality_profile

    if not req.request.strip():
        raise HTTPException(400, "request must not be empty")
    from master_agent.comfy.catalog import is_known_variant

    if not is_known_variant(req.variant):
        raise HTTPException(400, f"unknown variant: {req.variant}")
    if req.quality not in ("draft", "balanced", "quality"):
        raise HTTPException(400, "quality must be draft|balanced|quality")
    profile = get_quality_profile(req.quality)
    try:
        wf, meta = load_and_patch_workflow(
            req.variant,
            prompt=req.request.strip(),
            duration_s=req.duration_s,
            seed=req.seed,
            steps=profile.get("steps"),
            width=int(profile.get("max_width") or 768),
            height=int(profile.get("max_height") or 512),
        )
    except Exception as e:
        raise HTTPException(400, f"patch failed: {e}") from e
    logs: list[str] = []
    result = power_tune(
        wf,
        request=req.request.strip(),
        provider=req.provider,
        log=logs.append,
    )
    out = result.to_dict()
    out["variant"] = req.variant
    out["log"] = logs
    out["patch_meta"] = {
        k: meta.get(k)
        for k in ("seed", "steps", "cfg", "width", "height", "frames")
        if k in meta
    }
    # Do not return full workflow by default (large); include node count only
    return out


@app.get("/api/comfy/templates")
def api_comfy_templates():
    from master_agent.comfy.cli_run import list_templates

    return {"items": list_templates()}


@app.get("/api/variants")
def api_variants():
    """Default catalog — same list as CLI ``workflows`` / Create-tab picker."""
    from master_agent.comfy.catalog import default_entries

    return {
        "items": [
            {
                "id": e.id,
                "path": e.path,
                "name": e.name,
                "description": e.description,
                "family": e.family,
                "modes": list(e.modes),
                "aliases": list(e.aliases),
            }
            for e in default_entries()
        ]
    }


@app.post("/api/comfy/prepare")
def api_comfy_prepare(req: ComfyPrepareRequest):
    from master_agent.comfy.cli_run import editable_fields, prepare_run

    mode = (req.mode or "template").strip().lower()
    if mode not in ("raw", "template", "generate"):
        raise HTTPException(400, "mode must be raw|template|generate")
    try:
        workflow = prepare_run(
            mode,
            workflow=req.workflow,
            template_path=req.template,
            variant=req.variant,
            prompt=req.prompt,
            overrides=req.overrides,
        )
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(400, str(e)) from e
    return {
        "ok": True,
        "mode": mode,
        "nodes": len(workflow),
        "fields": editable_fields(workflow),
        "workflow": workflow,
    }


@app.post("/api/comfy/lint")
def api_comfy_lint(req: ComfyWorkflowBody):
    from master_agent.comfy.cli_run import lint_report
    from master_agent.comfy.client import ComfyClient

    try:
        object_info, source = ComfyClient().load_object_info(prefer_live=True)
    except Exception as e:
        raise HTTPException(503, f"cannot load /object_info: {e}") from e
    report = lint_report(req.workflow, object_info)
    report["object_info"] = source
    return report


@app.post("/api/comfy/run")
def api_comfy_run(req: ComfyWorkflowBody):
    from master_agent.comfy.cli_run import unwrap_workflow

    try:
        workflow = unwrap_workflow(req.workflow)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not any(isinstance(node, dict) and "class_type" in node for node in workflow.values()):
        raise HTTPException(400, "workflow has no Comfy nodes")
    label = (req.request or "comfy run").strip() or "comfy run"
    job = MANAGER.submit("comfy-run", label, workflow=workflow)
    return job.to_dict()


@app.post("/api/intake")
def api_intake(req: IntakeRequest):
    from master_agent.persona.intake import IntakeSession

    if req.session_id:
        session = _intake_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(404, "intake session not found — start a new one")
        rep = session.reply(req.message or "")
        out: dict[str, Any] = {
            "session_id": req.session_id,
            "spoken": rep.spoken,
            "done": rep.done,
        }
        if rep.done and rep.brief:
            out["brief"] = rep.brief.to_dict()
            out["refined_request"] = rep.brief.to_request()
            _intake_sessions.pop(req.session_id, None)
        return out

    if not (req.request or "").strip():
        raise HTTPException(400, "pass request (new session) or session_id + message")
    session = IntakeSession(req.request.strip())
    _intake_sessions[session.id] = session
    # bound the registry
    while len(_intake_sessions) > 50:
        _intake_sessions.pop(next(iter(_intake_sessions)))
    return {
        "session_id": session.id,
        "persona": session.persona.name,
        "spoken": session.opening(),
        "done": False,
    }


@app.get("/api/runs")
def api_runs(limit: int = Query(50, le=200)) -> list[dict[str, Any]]:
    records = sorted(RUNS_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]
    out = []
    for path in records:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_pipeline = d.get("segment_paths") is not None
        kind = d.get("kind") or ("pipeline" if is_pipeline else "run")
        video = d.get("upscaled_path") or d.get("video_path") or ""
        out.append(
            {
                "file": path.name,
                "ts": path.stem.split("_")[0],
                "kind": kind,
                "run_id": d.get("run_id"),
                "request": d.get("request") or "",
                "variant": d.get("variant") or "",
                "status": d.get("status") or d.get("state") or "",
                "score": d.get("full_judge_score", d.get("judge_score")),
                "passed": bool(
                    d.get("full_judge_pass") or d.get("judge_decision") == "accept"
                ),
                "judge_reason": (d.get("full_judge_notes") or d.get("judge_reason") or "")[:300],
                "video_url": _video_url(video),
                "duration_s": sum(d.get("segment_durations") or []) or d.get("duration_s"),
            }
        )
    return out


def _video_url(video_path: str) -> Optional[str]:
    if not video_path:
        return None
    try:
        rel = Path(video_path).resolve().relative_to(OUTPUTS_DIR.resolve())
    except (ValueError, OSError):
        return None
    return f"/videos/{rel.as_posix()}"


@app.get("/api/kb/search")
def api_kb_search(q: str, collection: str = "runs", k: int = Query(5, le=20)):
    from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, search

    coll = COLLECTION_WORKFLOWS if collection == "workflows" else COLLECTION_RUNS
    return search(coll, q, k=k)


_models_cache: tuple[float, Any] = (0.0, None)


@app.get("/api/models")
def api_models():
    global _models_cache
    ts, cached = _models_cache
    if cached is not None and time.time() - ts < 60:
        return cached
    from master_agent.models.inventory import scan_inventory

    inv = scan_inventory()
    data = {
        "bundles": inv.bundles,
        "files": [
            {"name": f.name, "size_gb": round(f.stat().st_size / 1e9, 2), "dir": f.parent.name}
            for f in getattr(inv, "files", [])
        ]
        if hasattr(inv, "files")
        else [],
    }
    _models_cache = (time.time(), data)
    return data


@app.post("/api/judge")
def api_judge(req: JudgeRequest):
    from master_agent.judge.judge import judge_full_video, judge_segment
    from master_agent.judge.probe import analyze

    if req.full_video:
        res = judge_full_video(
            user_request=req.request, storyboard=None, video_path=req.video_path
        )
    else:
        heuristic, issues = analyze(req.video_path)
        res = judge_segment(
            user_request=req.request,
            ltx_prompt=req.request,
            video_path=req.video_path,
            heuristic_score=heuristic,
            heuristic_issues=issues,
        )
    return res.to_dict()


# A2A sits beside MCP (master_agent/mcp_server.py). Protocol lives in a2a/.
_A2A = None


def _a2a_store():
    global _A2A
    if _A2A is None:
        from master_agent.a2a.protocol import TaskStore

        _A2A = TaskStore()
    return _A2A


def _a2a_bind(request: Request) -> tuple[str, int]:
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port or 8189
    return host, port


@app.get("/.well-known/agent.json")
def a2a_agent_card(request: Request):
    from master_agent.a2a.protocol import agent_card

    host, port = _a2a_bind(request)
    return agent_card(host=host, port=port)


@app.post("/a2a")
def a2a_rpc(payload: dict[str, Any], request: Request):
    from master_agent.a2a.protocol import handle_rpc, submit_orchestrator

    store = _a2a_store()
    host, port = _a2a_bind(request)
    return handle_rpc(
        payload,
        store=store,
        submit=lambda tid, body: submit_orchestrator(tid, body, store),
        host=host,
        port=port,
    )


class ControlUpdate(BaseModel):
    judge_strictness: Optional[float] = None
    learning_rate: Optional[float] = None
    cost_vram_threshold_gb: Optional[float] = None
    render_budget_cap_vram_min: Optional[float] = None
    judge_score_threshold: Optional[float] = None
    persona: Optional[str] = None
    soul: Optional[str] = None
    reset_budget: bool = False
    session: Optional[str] = None


def _control_payload() -> dict[str, Any]:
    from master_agent.control.budget import get_project_budget
    from master_agent.control.versioned_config import get_versioned_config

    store = get_versioned_config()
    snap = store.snapshot()
    budget = get_project_budget().snapshot()
    values = snap["values"]
    from master_agent.persona.persona import list_personas
    from master_agent.persona.soul import list_souls

    persona = str(values.get("persona") or "ara")
    soul = str(values.get("soul") or "studio")
    return {
        "hash": snap["hash"],
        "judge_strictness": values.get("judge_strictness"),
        "learning_rate": values.get("learning_rate"),
        "cost_vram_threshold_gb": values.get("cost_vram_threshold_gb"),
        "judge_score_threshold": values.get("judge_score_threshold"),
        "render_budget_cap_vram_min": budget["cap"],
        "render_budget_used_vram_min": budget["used"],
        "persona": persona,
        "soul": soul,
        "personas": [{"slug": p.slug, "name": p.name} for p in list_personas()],
        "souls": [{"slug": s.slug, "name": s.name} for s in list_souls()],
        "budget": budget,
        "history": store.history(10),
    }


@app.get("/api/control")
def api_control_get():
    return _control_payload()


@app.post("/api/control")
def api_control_post(req: ControlUpdate):
    from master_agent.control.budget import get_project_budget
    from master_agent.control.versioned_config import get_versioned_config

    session = (req.session or "studio").strip() or "studio"
    updates = req.model_dump(exclude_none=True)
    updates.pop("reset_budget", None)
    updates.pop("session", None)
    store = get_versioned_config()
    if updates:
        store.set_values(updates, session=session, sync_budget=True)
    if req.reset_budget:
        get_project_budget().reset_used()
        store.sync_used(0.0)
    return _control_payload()
