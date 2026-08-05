"""FastAPI app — the Master Agent web dashboard.

Run: python -m master_agent ui --port 8189
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from master_agent.config import OUTPUTS_DIR, RUNS_DIR
from master_agent.web.jobs import MANAGER

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ComfyUI Master Agent", docs_url=None, redoc_url=None)


# ── models ──────────────────────────────────────────────


class JobRequest(BaseModel):
    request: str
    duration_s: float = 8.0
    quality: str = "draft"
    variant: Optional[str] = None
    llm_panel: Optional[str] = None
    dry_run: bool = False


class JudgeRequest(BaseModel):
    video_path: str
    request: str = ""
    full_video: bool = False


class FractalRequest(BaseModel):
    request: str = ""
    duration_s: float = 20.0
    fps: int = 24
    width: int = 768
    height: int = 512
    target: str = "seahorse"
    palette: str = "fire"
    julia: bool = False
    seed: Optional[int] = None
    audio_path: Optional[str] = None
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


# ── pages & media ───────────────────────────────────────


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/videos/{rel_path:path}")
def videos(rel_path: str):
    """Serve generated videos for playback (path-guarded to OUTPUTS_DIR)."""
    base = OUTPUTS_DIR.resolve()
    target = (base / rel_path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(404, "video not found")
    return FileResponse(target)


# ── api ─────────────────────────────────────────────────


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    from master_agent.comfy.client import ComfyClient
    from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, collection_count
    from master_agent.llm import provider_available

    out: dict[str, Any] = {
        "ollama": provider_available("ollama"),
        "kb": {
            "workflows": collection_count(COLLECTION_WORKFLOWS),
            "runs": collection_count(COLLECTION_RUNS),
        },
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
    return out


@app.post("/api/jobs")
def api_submit_job(req: JobRequest):
    if not req.request.strip():
        raise HTTPException(400, "request must not be empty")
    if req.quality not in ("draft", "balanced", "quality"):
        raise HTTPException(400, "quality must be draft|balanced|quality")
    if req.variant not in (None, "base", "directors", "eros", "lipsync", "wan22"):
        raise HTTPException(400, "unknown variant")
    kind = "dry-run" if req.dry_run else "run"
    job = MANAGER.submit(
        kind,
        req.request.strip(),
        duration_s=req.duration_s,
        quality=req.quality,
        variant=req.variant,
        llm_panel=req.llm_panel,
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


_FRACTAL_TARGETS = {"seahorse", "elephant", "minibrot", "spiral"}
_FRACTAL_PALETTES = {"fire", "ocean", "monochrome", "neon", "sunset"}
_UPSCALE_METHODS = {None, "rtx", "seedvr2"}


@app.post("/api/fractal")
def api_fractal(req: FractalRequest):
    if req.target not in _FRACTAL_TARGETS:
        raise HTTPException(400, f"target must be one of {sorted(_FRACTAL_TARGETS)}")
    if req.palette not in _FRACTAL_PALETTES:
        raise HTTPException(400, f"palette must be one of {sorted(_FRACTAL_PALETTES)}")
    if req.upscale not in _UPSCALE_METHODS:
        raise HTTPException(400, "upscale must be rtx|seedvr2")
    if req.audio_path and not Path(req.audio_path).is_file():
        raise HTTPException(400, f"audio not found: {req.audio_path}")
    job = MANAGER.submit(
        "fractal",
        req.request.strip() or f"fractal {req.target}/{req.palette}",
        duration_s=req.duration_s,
        fps=req.fps,
        width=req.width,
        height=req.height,
        target=req.target,
        palette=req.palette,
        julia=req.julia,
        seed=req.seed,
        audio_path=req.audio_path,
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
    if not Path(req.audio_path).is_file():
        raise HTTPException(400, f"audio not found: {req.audio_path}")
    job = MANAGER.submit(
        "music",
        req.request.strip(),
        audio_path=req.audio_path,
        visual=req.visual,
        quality=req.quality,
        seed=req.seed,
        upscale=req.upscale,
    )
    return job.to_dict()


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Save an uploaded media file (e.g. a music track) under state/uploads/."""
    from master_agent.config import STATE_DIR
    import uuid as _uuid

    uploads = STATE_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.bin").name
    dest = uploads / f"{_uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"path": str(dest), "name": safe_name}


# ── intake interview (persona chat) ─────────────────────

_intake_sessions: dict[str, Any] = {}


@app.get("/api/persona")
def api_persona():
    from master_agent.persona.persona import load_persona

    p = load_persona()
    return {"slug": p.slug, "name": p.name}


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
