"""ClipProvenance — prompts + full provenance next to every clip.

Wire-compatible with buddy-core ``ClipProvenance`` (Rust sibling
your-video-buddy). That schema was **not fetchable** from this environment
(repo 404 / agent bc-39743643 not in this workspace). Field names below are
the required contract; do not rename them. If the Rust struct lands extra
keys, add them without dropping these.

Sidecar: ``<clip_stem>.provenance.json`` next to the file.
Run row: the same object is embedded on the orchestrator / pipeline record
and keyed by ``path`` / ``hash``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

CLIP_PROVENANCE_SCHEMA = "buddy.clip.provenance/v1"

# Required wire fields (buddy-core ClipProvenance). Keep these names stable.
REQUIRED_FIELDS = (
    "prompt",
    "model",
    "workflow_id",
    "seed",
    "params",
    "attempt",
    "iteration",
    "judge_score",
    "judge_reasons",
    "revise_notes",
)


def sidecar_path(clip_path: str | Path) -> Path:
    p = Path(clip_path)
    return p.with_name(f"{p.stem}.provenance.json")


def sha256_file(path: str | Path | None) -> str:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return ""
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def params_from_state(st: Any) -> dict[str, Any]:
    return {
        "steps": getattr(st, "steps", None),
        "cfg": getattr(st, "cfg", None),
        "width": getattr(st, "width", None),
        "height": getattr(st, "height", None),
        "duration_s": getattr(st, "duration_s", None),
        "stg_scale": getattr(st, "stg_scale", None),
        "stg_blocks": getattr(st, "stg_blocks", None),
        "sampler_name": getattr(st, "sampler_name", None),
        "quality": getattr(st, "quality", None),
        "negative_prompt": getattr(st, "negative_prompt", None) or "",
    }


def model_from_state(st: Any) -> str:
    meta = getattr(st, "workflow_meta", None) or {}
    for key in ("checkpoint", "checkpoint_preferred", "default_pack"):
        value = meta.get(key)
        if value:
            return str(value)
    return str(getattr(st, "variant", None) or "")


def build_clip_provenance(
    st: Any,
    *,
    path: str | Path | None = None,
    revise_notes: str = "",
    hash_value: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ClipProvenance object. Always includes REQUIRED_FIELDS."""
    clip = path if path is not None else getattr(st, "video_path", None)
    clip_str = str(clip) if clip else ""
    digest = hash_value if hash_value is not None else sha256_file(clip_str or None)
    attempt = int(getattr(st, "attempt", 1) or 1)
    reasons = list(getattr(st, "judge_issues", None) or [])
    reason = str(getattr(st, "judge_reason", "") or "")
    if reason and reason not in reasons:
        reasons = [reason] + reasons
    payload = {
        "schema": CLIP_PROVENANCE_SCHEMA,
        "prompt": str(getattr(st, "prompt", "") or getattr(st, "request", "") or ""),
        "model": model_from_state(st),
        "workflow_id": str(getattr(st, "variant", None) or ""),
        "seed": getattr(st, "seed", None),
        "params": params_from_state(st),
        "attempt": attempt,
        "iteration": attempt,
        "judge_score": float(getattr(st, "judge_score", 0.0) or 0.0),
        "judge_reasons": reasons,
        "revise_notes": revise_notes or "",
        "path": clip_str,
        "hash": digest,
        "run_id": str(getattr(st, "run_id", "") or ""),
    }
    return payload


def write_clip_provenance(clip_path: str | Path, payload: dict[str, Any]) -> Path:
    dest = sidecar_path(clip_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return dest


def read_clip_provenance(clip_path: str | Path | None) -> Optional[dict[str, Any]]:
    """Read the sidecar for a clip path. None if missing / unreadable."""
    if not clip_path:
        return None
    dest = sidecar_path(clip_path)
    if not dest.is_file():
        return None
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def persist_clip_provenance(
    st: Any,
    *,
    revise_notes: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Write sidecar (when a clip exists) and record the object on ``st``."""
    clip = path if path is not None else getattr(st, "video_path", None)
    payload = build_clip_provenance(st, path=clip, revise_notes=revise_notes)
    st.provenance = payload
    history = getattr(st, "provenance_history", None)
    if history is None:
        st.provenance_history = [payload]
    else:
        history.append(payload)
    if clip:
        try:
            dest = write_clip_provenance(clip, payload)
            st.provenance_sidecar = str(dest)
        except OSError:
            pass
    return payload


_REQUIRED_DEFAULTS: dict[str, Any] = {
    "prompt": "",
    "model": "",
    "workflow_id": "",
    "seed": None,
    "params": {},
    "attempt": 1,
    "iteration": 1,
    "judge_score": 0.0,
    "judge_reasons": [],
    "revise_notes": "",
}


def inherit_clip_provenance(
    src_clip: str | Path,
    dest_clip: str | Path,
    *,
    revise_notes: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy provenance onto a derived clip (trim / stitch / mux) and rewrite keys."""
    payload = dict(read_clip_provenance(src_clip) or {})
    for key, default in _REQUIRED_DEFAULTS.items():
        payload.setdefault(key, default if not isinstance(default, (dict, list)) else default.copy())
    if extra:
        payload.update(extra)
    if revise_notes:
        payload["revise_notes"] = revise_notes
    payload["schema"] = CLIP_PROVENANCE_SCHEMA
    payload["path"] = str(dest_clip)
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
