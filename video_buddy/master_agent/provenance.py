"""ClipProvenance — ``buddy.clip.provenance/v1`` (Rust buddy-core aligned).

Sidecar: ``{output_dir}/shot-N.buddy.json`` next to planned ``shot-N.mp4``.
The same object is embedded on the orchestrator / pipeline run JSON.

Target schema is Rust ``ClipProvenance`` from sibling your-video-buddy
PR #7 (``cursor/close-improve-loop-1c25``, agent bc-39743643). That repo
was **not fetchable** from this environment (404). Field names below match
the PR #7 contract Scott listed — do not invent a second schema. If the
Rust struct lands extra keys, add them without dropping these.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.config import DEFAULT_FPS, OUTPUTS_DIR, get_variant_gen

CLIP_PROVENANCE_SCHEMA = "buddy.clip.provenance/v1"
SIDECAR_SUFFIX = ".buddy.json"
LEGACY_SIDECAR_SUFFIX = ".provenance.json"

REQUIRED_TOP = (
    "schema",
    "prompts",
    "engine",
    "params",
    "lineage",
    "judge",
    "revise_notes",
    "created_at",
    "output_path",
)
REQUIRED_PROMPTS = ("brief", "positive", "negative", "additives")
REQUIRED_ENGINE = ("backend", "workflow_id", "variant")
REQUIRED_PARAMS = ("seed", "steps", "cfg", "size", "fps", "duration", "refs")
REQUIRED_LINEAGE = (
    "shot_id",
    "attempt_id",
    "iteration",
    "parent_shot_id",
    "parent_attempt_id",
)
REQUIRED_JUDGE = ("score", "fail_reasons")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def shot_n(st: Any) -> int:
    idx = getattr(st, "shot_index", None)
    if idx:
        try:
            return max(1, int(idx))
        except (TypeError, ValueError):
            pass
    shot = getattr(st, "shot", None) or {}
    if isinstance(shot, dict) and shot.get("index") is not None:
        try:
            raw = int(shot["index"])
            return raw if raw >= 1 else raw + 1
        except (TypeError, ValueError):
            pass
    return 1


def shot_id_of(st: Any) -> str:
    return f"shot-{shot_n(st)}"


def attempt_id_of(shot_id: str, attempt: int) -> str:
    return f"{shot_id}.a{int(attempt)}"


def output_dir_of(st: Any) -> Path:
    raw = getattr(st, "output_dir", None)
    if raw:
        return Path(raw)
    run_id = str(getattr(st, "run_id", "") or "run")
    return OUTPUTS_DIR / run_id


def planned_clip_path(st: Any) -> Path:
    planned = getattr(st, "planned_clip", None)
    if planned:
        return Path(planned)
    return output_dir_of(st) / f"{shot_id_of(st)}.mp4"


def sidecar_path(clip_path: str | Path) -> Path:
    """``shot-N.buddy.json`` next to ``shot-N.mp4`` (or any clip stem)."""
    p = Path(clip_path)
    return p.with_name(f"{p.stem}{SIDECAR_SUFFIX}")


def sidecar_path_for_state(st: Any) -> Path:
    return sidecar_path(planned_clip_path(st))


def plan_clip_paths(st: Any) -> Path:
    """Seat planned ``shot-N.mp4`` + sidecar dir on the run state."""
    out = output_dir_of(st)
    out.mkdir(parents=True, exist_ok=True)
    stem = shot_id_of(st)
    clip = out / f"{stem}.mp4"
    st.output_dir = str(out)
    st.planned_clip = str(clip)
    st.shot_id = stem
    if not getattr(st, "video_path", None):
        st.video_path = str(clip)
    return clip


def sha256_file(path: str | Path | None) -> Optional[str]:
    """Hash only if the file exists; otherwise None (never invent)."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refs_from_state(st: Any) -> list[str]:
    refs: list[str] = []
    for attr in ("image_name", "video_name", "audio_name", "audio_path", "previs_source"):
        value = getattr(st, attr, None)
        if value:
            refs.append(str(value))
    return refs


def _additives_from_state(st: Any) -> list[str]:
    additives: list[str] = []
    for item in getattr(st, "revise_history", None) or []:
        if isinstance(item, dict):
            for delta in item.get("prompt_deltas") or []:
                if delta and str(delta) not in additives:
                    additives.append(str(delta))
    shot = getattr(st, "shot", None) or {}
    if isinstance(shot, dict):
        for key in ("camera", "action", "visuals", "continuity"):
            value = shot.get(key)
            if value and str(value) not in additives:
                additives.append(str(value))
    return additives


def judge_fail_reasons(st: Any) -> list[dict[str, Any]]:
    """Honest fail reasons. Quality-bar items use kind ``cpu_fail_rules``."""
    reasons: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fail in (getattr(st, "quality_bar", None) or {}).get("fails") or []:
        if not isinstance(fail, dict):
            continue
        key = f"cpu:{fail.get('id')}:{fail.get('code')}"
        seen.add(key)
        reasons.append(
            {
                "kind": "cpu_fail_rules",
                "id": fail.get("id"),
                "code": fail.get("code"),
                "detail": fail.get("detail") or "",
            }
        )
    for issue in getattr(st, "judge_issues", None) or []:
        text = str(issue)
        if text.startswith("quality_bar."):
            continue
        if text and text not in seen:
            seen.add(text)
            reasons.append({"kind": "judge", "detail": text})
    reason = str(getattr(st, "judge_reason", "") or "")
    if reason and reason not in seen and not reason.startswith("quality_bar"):
        reasons.append({"kind": "judge", "detail": reason})
    return reasons


def build_clip_provenance(
    st: Any,
    *,
    path: str | Path | None = None,
    revise_notes: str = "",
    hash_value: Optional[str] = None,
    created_at: Optional[str] = None,
    parent_shot_id: Optional[str] = None,
    parent_attempt_id: Optional[str] = None,
) -> dict[str, Any]:
    clip = path if path is not None else (
        getattr(st, "video_path", None) or planned_clip_path(st)
    )
    clip_str = str(clip) if clip else ""
    digest = hash_value if hash_value is not None else sha256_file(clip_str or None)
    attempt = int(getattr(st, "attempt", 1) or 1)
    sid = getattr(st, "shot_id", None) or shot_id_of(st)
    variant = str(getattr(st, "variant", None) or "")
    fps = getattr(st, "fps", None)
    if fps is None:
        fps = get_variant_gen(variant).get("fps", DEFAULT_FPS)
    parent_shot = parent_shot_id
    if parent_shot is None:
        parent_shot = getattr(st, "parent_shot_id", None) or None
    parent_attempt = parent_attempt_id
    if parent_attempt is None:
        parent_attempt = getattr(st, "parent_attempt_id", None) or None
    if parent_shot == "":
        parent_shot = None
    if parent_attempt == "":
        parent_attempt = None
    prompts: dict[str, Any] = {
        "brief": str(getattr(st, "request", "") or ""),
        "positive": str(getattr(st, "prompt", "") or getattr(st, "request", "") or ""),
        "negative": str(getattr(st, "negative_prompt", "") or ""),
        "additives": _additives_from_state(st),
    }
    spoken = str(getattr(st, "spoken_line", "") or "").strip()
    if spoken:
        prompts["spoken_line"] = spoken
    payload = {
        "schema": CLIP_PROVENANCE_SCHEMA,
        "prompts": prompts,
        "engine": {
            "backend": "comfy",
            "workflow_id": variant,
            "variant": variant,
        },
        "params": {
            "seed": getattr(st, "seed", None),
            "steps": getattr(st, "steps", None),
            "cfg": getattr(st, "cfg", None),
            "size": [
                int(getattr(st, "width", 0) or 0),
                int(getattr(st, "height", 0) or 0),
            ],
            "fps": int(fps) if fps is not None else DEFAULT_FPS,
            "duration": float(getattr(st, "duration_s", 0.0) or 0.0),
            "refs": _refs_from_state(st),
        },
        "lineage": {
            "shot_id": sid,
            "attempt_id": attempt_id_of(sid, attempt),
            "iteration": attempt,
            "parent_shot_id": parent_shot,
            "parent_attempt_id": parent_attempt,
        },
        "judge": {
            "score": float(getattr(st, "judge_score", 0.0) or 0.0),
            "fail_reasons": judge_fail_reasons(st),
        },
        "revise_notes": revise_notes or "",
        "created_at": created_at or utc_now(),
        "output_path": clip_str,
        "hash": digest,
    }
    sample = getattr(st, "voice_sample", None) or {}
    if isinstance(sample, dict) and sample.get("method"):
        payload["params"]["voice_sample"] = {
            "original_duration_s": sample.get("original_duration_s"),
            "start_s": sample.get("start_s"),
            "end_s": sample.get("end_s"),
            "method": sample.get("method"),
            "trimmed": bool(sample.get("trimmed")),
        }
    return payload


def missing_required(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_TOP:
        if key not in payload:
            missing.append(key)
    prompts = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
    for key in REQUIRED_PROMPTS:
        if key not in prompts:
            missing.append(f"prompts.{key}")
    engine = payload.get("engine") if isinstance(payload.get("engine"), dict) else {}
    for key in REQUIRED_ENGINE:
        if key not in engine:
            missing.append(f"engine.{key}")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    for key in REQUIRED_PARAMS:
        if key not in params:
            missing.append(f"params.{key}")
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    for key in REQUIRED_LINEAGE:
        if key not in lineage:
            missing.append(f"lineage.{key}")
    judge = payload.get("judge") if isinstance(payload.get("judge"), dict) else {}
    for key in REQUIRED_JUDGE:
        if key not in judge:
            missing.append(f"judge.{key}")
    return missing


def write_clip_provenance(clip_path: str | Path, payload: dict[str, Any]) -> Path:
    dest = sidecar_path(clip_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return dest


def read_clip_provenance(clip_path: str | Path | None) -> Optional[dict[str, Any]]:
    """Read ``.buddy.json`` (legacy ``.provenance.json`` fallback). None if missing."""
    if not clip_path:
        return None
    dest = sidecar_path(clip_path)
    if not dest.is_file():
        legacy = Path(clip_path).with_name(f"{Path(clip_path).stem}{LEGACY_SIDECAR_SUFFIX}")
        dest = legacy if legacy.is_file() else dest
    if not dest.is_file():
        return None
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_sidecar_for_state(st: Any) -> Optional[dict[str, Any]]:
    """Source of truth across attempts — planned shot sidecar first."""
    for candidate in (
        sidecar_path_for_state(st),
        sidecar_path(getattr(st, "video_path", None) or "")
        if getattr(st, "video_path", None)
        else None,
    ):
        if candidate is None:
            continue
        loaded = read_clip_provenance(Path(str(candidate)).with_suffix(".mp4"))
        if loaded:
            return loaded
        # candidate may already be the sidecar path
        if Path(candidate).is_file():
            try:
                data = json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                return data
    return None


def apply_sidecar_to_state(st: Any, payload: dict[str, Any]) -> None:
    """Restore prompts / params / parent lineage from sidecar before revise."""
    prompts = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
    if prompts.get("positive"):
        st.prompt = str(prompts["positive"])
    if "negative" in prompts:
        st.negative_prompt = str(prompts.get("negative") or "")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if params.get("seed") is not None:
        st.seed = params["seed"]
    if params.get("steps") is not None:
        st.steps = params["steps"]
    if params.get("cfg") is not None:
        st.cfg = params["cfg"]
    if params.get("fps") is not None:
        st.fps = params["fps"]
    if params.get("duration") is not None:
        st.duration_s = float(params["duration"])
    size = params.get("size")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        st.width, st.height = int(size[0]), int(size[1])
    elif isinstance(size, dict):
        if size.get("width"):
            st.width = int(size["width"])
        if size.get("height"):
            st.height = int(size["height"])
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    st.parent_shot_id = lineage.get("shot_id") or getattr(st, "parent_shot_id", "") or ""
    st.parent_attempt_id = lineage.get("attempt_id") or getattr(st, "parent_attempt_id", "") or ""


def persist_clip_provenance(
    st: Any,
    *,
    revise_notes: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Write sidecar (planned shot path) and record the object on ``st``."""
    plan_clip_paths(st)
    clip = path if path is not None else planned_clip_path(st)
    prior = read_clip_provenance(clip)
    created_at = None
    if prior:
        prior_attempt = (prior.get("lineage") or {}).get("iteration")
        if prior_attempt == int(getattr(st, "attempt", 1) or 1):
            created_at = prior.get("created_at")
    payload = build_clip_provenance(
        st, path=clip, revise_notes=revise_notes, created_at=created_at
    )
    st.provenance = payload
    history = getattr(st, "provenance_history", None)
    if history is None:
        st.provenance_history = [payload]
    else:
        history.append(payload)
    try:
        dest = write_clip_provenance(clip, payload)
        st.provenance_sidecar = str(dest)
    except OSError:
        pass
    return payload


def inherit_clip_provenance(
    src_clip: str | Path,
    dest_clip: str | Path,
    *,
    revise_notes: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy provenance onto a derived clip (trim / stitch / mux)."""
    payload = dict(read_clip_provenance(src_clip) or {})
    if extra:
        if "prompts" in extra or "engine" in extra or "params" in extra:
            for key, value in extra.items():
                if isinstance(value, dict) and isinstance(payload.get(key), dict):
                    merged = dict(payload[key])
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value
        else:
            prompts = dict(payload.get("prompts") or {})
            if extra.get("prompt"):
                prompts["brief"] = str(extra["prompt"])
                prompts.setdefault("positive", str(extra["prompt"]))
            if extra.get("brief"):
                prompts["brief"] = str(extra["brief"])
            payload["prompts"] = prompts
    if revise_notes:
        payload["revise_notes"] = revise_notes
    payload["schema"] = CLIP_PROVENANCE_SCHEMA
    payload.setdefault(
        "prompts",
        {"brief": "", "positive": "", "negative": "", "additives": []},
    )
    payload.setdefault(
        "engine", {"backend": "comfy", "workflow_id": "", "variant": ""}
    )
    payload.setdefault(
        "params",
        {
            "seed": None,
            "steps": None,
            "cfg": None,
            "size": [0, 0],
            "fps": DEFAULT_FPS,
            "duration": 0.0,
            "refs": [],
        },
    )
    payload.setdefault(
        "lineage",
        {
            "shot_id": "",
            "attempt_id": "",
            "iteration": 1,
            "parent_shot_id": None,
            "parent_attempt_id": None,
        },
    )
    payload.setdefault("judge", {"score": 0.0, "fail_reasons": []})
    payload.setdefault("revise_notes", revise_notes or "")
    payload.setdefault("created_at", utc_now())
    payload["output_path"] = str(dest_clip)
    payload["hash"] = sha256_file(dest_clip)
    write_clip_provenance(dest_clip, payload)
    return payload


def latest_revise_notes(st: Any) -> str:
    """What changed this attempt — reason plus prompt/param/shot deltas."""
    hist = getattr(st, "revise_history", None) or []
    if not hist:
        return ""
    last = hist[-1]
    if not isinstance(last, dict):
        return ""
    if last.get("revise_notes"):
        return str(last["revise_notes"])
    bits: list[str] = []
    if last.get("reason"):
        bits.append(str(last["reason"]))
    deltas = last.get("prompt_deltas") or []
    if deltas:
        bits.append("prompt: " + "; ".join(str(d) for d in deltas[:3]))
    params = last.get("param_deltas") or {}
    if params:
        bits.append("params: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    shot = last.get("shot_patch") or {}
    if shot:
        bits.append("shot: " + ",".join(str(k) for k in shot.keys()))
    if last.get("music_bed_attached"):
        bits.append("music_bed_attached")
    used = last.get("control_pack_used") or {}
    on = [k for k, v in used.items() if v]
    if on:
        bits.append("control: " + ",".join(on))
    return "; ".join(bits)
