"""Remotion assemble: Sequence clips on the beat plan + full-track audio."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from master_agent.music.plan import BeatPlan, MV_FPS

REMOTION_PROPS_SCHEMA = "buddy.mv.remotion_props/v1"
DEFAULT_OUT_NAME = "MV-FIXED.mp4"
COMP_WIDTH = 1920
COMP_HEIGHT = 1080
COMP_ID = "MusicVideo"


def remotion_project_dir() -> Path:
    """In-repo Remotion project (``video_buddy/remotion-mv``)."""
    return Path(__file__).resolve().parents[2] / "remotion-mv"


def remotion_props(
    plan: BeatPlan | dict[str, Any],
    clips: dict[int, str | Path],
    audio: str | Path,
    *,
    width: int = COMP_WIDTH,
    height: int = COMP_HEIGHT,
) -> dict[str, Any]:
    data = plan.to_dict() if isinstance(plan, BeatPlan) else plan
    fps = int(data.get("fps") or MV_FPS)
    windows = []
    for win in data.get("windows") or []:
        idx = int(win["index"])
        clip = clips.get(idx)
        if clip is None:
            raise ValueError(f"no clip for window {idx}")
        windows.append(
            {
                "index": idx,
                "start_s": win["start_s"],
                "end_s": win["end_s"],
                "start_frame": int(win["start_frame"]),
                "end_frame": int(win["end_frame"]),
                "clip": str(Path(clip).resolve()),
                "energy": win.get("energy"),
                "label": win.get("label"),
            }
        )
    return {
        "schema": REMOTION_PROPS_SCHEMA,
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "durationInFrames": int(data["duration_frames"]),
        "audio": str(Path(audio).resolve()),
        "windows": windows,
    }


def remotion_command(
    props_path: str | Path,
    out_path: str | Path,
    *,
    remotion_dir: str | Path | None = None,
    scale: float | None = None,
) -> list[str]:
    """``npx remotion render`` argv (documented 1080p path).

    Optional ``scale=2/3`` notes a 720p render (1280×720) without a second
    composition. Production default is 1080p (``scale`` omitted).
    """
    cmd = [
        "npx",
        "--yes",
        "remotion",
        "render",
        "src/index.ts",
        COMP_ID,
        str(Path(out_path).resolve()),
        f"--props={Path(props_path).resolve()}",
    ]
    if scale is not None:
        cmd.append(f"--scale={scale}")
    _ = remotion_dir  # cwd is applied by the runner
    return cmd


def write_remotion_wiring(
    plan: BeatPlan | dict[str, Any],
    clips: dict[int, str | Path],
    audio: str | Path,
    *,
    out_path: str | Path,
    work_dir: str | Path,
    width: int = COMP_WIDTH,
    height: int = COMP_HEIGHT,
) -> dict[str, Any]:
    """Write props JSON + the exact render command. No Node required."""
    dest = Path(work_dir)
    dest.mkdir(parents=True, exist_ok=True)
    props = remotion_props(plan, clips, audio, width=width, height=height)
    props_path = dest / "remotion-props.json"
    props_path.write_text(json.dumps(props, indent=1) + "\n", encoding="utf-8")
    cmd = remotion_command(props_path, out_path, remotion_dir=remotion_project_dir())
    cmd_path = dest / "remotion-command.txt"
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    note = dest / "REMOTION.md"
    note.write_text(
        "1080p (default):\n"
        f"  cd {remotion_project_dir()}\n"
        "  npm install\n"
        f"  {' '.join(cmd)}\n\n"
        "720p note: add --scale=0.666667 (1280×720) to the same command.\n"
        f"Output: {Path(out_path).resolve()}\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "props_path": str(props_path),
        "command": cmd,
        "command_path": str(cmd_path),
        "remotion_dir": str(remotion_project_dir()),
        "out": str(Path(out_path)),
        "composition": COMP_ID,
        "width": width,
        "height": height,
        "fps": props["fps"],
        "windows": len(props["windows"]),
    }


def assemble_remotion(
    plan: BeatPlan | dict[str, Any],
    clips: dict[int, str | Path],
    audio: str | Path,
    *,
    out_path: str | Path,
    work_dir: str | Path,
    dry_run: bool = False,
    width: int = COMP_WIDTH,
    height: int = COMP_HEIGHT,
    runner: Optional[Any] = None,
    log=print,
) -> dict[str, Any]:
    """Sequence clips on the beat JSON + full audio.

    Dry-run writes Remotion props + the ``npx remotion render`` command and
    does not require Node. Live mode runs that command from the in-repo
    project unless ``runner`` is injected (tests).
    """
    wiring = write_remotion_wiring(
        plan,
        clips,
        audio,
        out_path=out_path,
        work_dir=work_dir,
        width=width,
        height=height,
    )
    if dry_run:
        log(f"remotion dry-run: props {wiring['props_path']}")
        log(f"remotion command: {' '.join(wiring['command'])}")
        wiring["dry_run"] = True
        return wiring

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    project = remotion_project_dir()
    cmd = list(wiring["command"])
    if runner is not None:
        runner(cmd, cwd=str(project), out=str(dest))
        wiring["dry_run"] = False
        wiring["rendered"] = dest.is_file()
        return wiring

    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError(
            "npx not found — install Node, then: "
            f"cd {project} && npm install && {' '.join(cmd)}"
        )
    env = os.environ.copy()
    log(f"remotion render: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(f"remotion render failed: {err}")
    wiring["dry_run"] = False
    wiring["rendered"] = dest.is_file()
    return wiring
