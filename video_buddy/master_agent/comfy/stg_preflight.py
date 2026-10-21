"""STG preflight: ComfyUI >=0.40 breaks STG on ComfyUI-LTXVideo older than 1.1.0.

Lightricks/ComfyUI-LTXVideo#579, fixed in #583 (v1.1.0). STG stays on by
default. Only that known-bad pair sets ``stg`` to 0 on every
``GuiderParameters`` node for the run, with a warning. When either version
cannot be read the graph is left as is, also with a warning.

Versions come from the files on disk (``pyproject.toml`` /
``comfyui_version.py``), never from a network call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.comfy.updates import meets_floor, read_comfyui_version, read_node_version

log = logging.getLogger(__name__)

COMFYUI_MIN_VERSION = "0.40.0"
LTXVIDEO_MIN_VERSION = "1.1.0"
LTXVIDEO_NODE = "ComfyUI-LTXVideo"

KEEP, DISABLE, UNKNOWN = "keep", "disable", "unknown"
ISSUE = "Lightricks/ComfyUI-LTXVideo#579"
FIX = "python -m master_agent comfy update --yes"


@dataclass(frozen=True)
class StgPreflight:
    comfyui: str | None
    ltxvideo: str | None
    action: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"comfyui": self.comfyui, "ltxvideo": self.ltxvideo, "action": self.action, "message": self.message}


def check_stg(comfyui: str | None, ltxvideo: str | None) -> StgPreflight:
    """Decide from two version strings. Pure; no disk access."""
    comfy_new = meets_floor(comfyui, COMFYUI_MIN_VERSION)
    ltx_fixed = meets_floor(ltxvideo, LTXVIDEO_MIN_VERSION)
    shown_comfy = comfyui or "unknown"
    shown_ltx = ltxvideo or "unknown"
    if comfy_new is False:
        return StgPreflight(comfyui, ltxvideo, KEEP, f"ComfyUI {comfyui} predates {COMFYUI_MIN_VERSION}; STG stays on.")
    if ltx_fixed is True:
        return StgPreflight(
            comfyui, ltxvideo, KEEP, f"{LTXVIDEO_NODE} {ltxvideo} has the STG fix; STG stays on."
        )
    if comfy_new is True and ltx_fixed is False:
        return StgPreflight(
            comfyui,
            ltxvideo,
            DISABLE,
            f"ComfyUI {comfyui} with {LTXVIDEO_NODE} {ltxvideo} breaks STG ({ISSUE}). "
            f"STG is off (stg=0) for this run. Update {LTXVIDEO_NODE} to >={LTXVIDEO_MIN_VERSION}: {FIX}",
        )
    return StgPreflight(
        comfyui,
        ltxvideo,
        UNKNOWN,
        f"Could not read the ComfyUI ({shown_comfy}) and {LTXVIDEO_NODE} ({shown_ltx}) versions; "
        f"STG left as authored. ComfyUI >={COMFYUI_MIN_VERSION} needs {LTXVIDEO_NODE} "
        f">={LTXVIDEO_MIN_VERSION} for STG ({ISSUE}).",
    )


def _candidate_roots(workspace: Path | None) -> list[Path]:
    from master_agent.comfy.layout import code_dir_for
    from master_agent.comfy.root import managed_comfy_root

    roots: list[Path] = []
    try:
        roots.append(managed_comfy_root(workspace))
    except Exception:
        pass
    if workspace is None:
        try:
            from master_agent import config

            roots.append(code_dir_for(config.COMFYUI_ROOT))
        except Exception:
            pass
    return roots


def detect_versions(workspace: Path | None = None) -> tuple[str | None, str | None]:
    """``(comfyui, ltxvideo)`` from the first ComfyUI folder with a ``main.py``."""
    for root in _candidate_roots(workspace):
        if (root / "main.py").is_file():
            return read_comfyui_version(root), read_node_version(root, LTXVIDEO_NODE)
    return None, None


def stg_preflight(workspace: Path | None = None) -> StgPreflight:
    return check_stg(*detect_versions(workspace))


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def stg_nodes(workflow: dict[str, Any]) -> list[str]:
    """``GuiderParameters`` node ids whose literal ``stg`` is above 0."""
    out = []
    for nid, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "GuiderParameters":
            continue
        stg = (node.get("inputs") or {}).get("stg")
        if _numeric(stg) and float(stg) > 0:
            out.append(str(nid))
    return out


def apply_stg_preflight(
    workflow: dict[str, Any], *, result: StgPreflight | None = None, workspace: Path | None = None
) -> dict[str, Any] | None:
    """Run the preflight on a patched graph. None when the graph has no STG to guard."""
    nodes = stg_nodes(workflow)
    if not nodes:
        return None
    result = result if result is not None else stg_preflight(workspace)
    report = result.as_dict()
    if result.action == DISABLE:
        for nid in nodes:
            workflow[nid]["inputs"]["stg"] = 0.0
        report["disabled_nodes"] = nodes
        log.warning(result.message)
    elif result.action == UNKNOWN:
        log.warning(result.message)
    return report
