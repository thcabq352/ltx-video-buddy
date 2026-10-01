"""Stale-pack report and opt-in Comfy / node updates.

Startup (doctor, ``comfy start``, ``comfy status``) only prints what is stale
versus the comfy-cli pin in ``requirements.txt`` and ``pack_pins.json``.
Nothing here runs ``comfy install`` or downloads weights.

``comfy update`` changes ComfyUI or custom nodes only after an explicit
``--yes`` or a y/n answer, and only after ``comfy node save-snapshot``.
That snapshot is nodes and Python dependencies. Weight keep/wipe stays on
``models select --keep`` / ``--wipe``.

A commit pin is reported when HEAD differs. A bare ``--yes`` does not run
``comfy node update`` for that pin, because node update moves the checkout
forward and does not land on the pinned commit. ``--nodes --yes`` is the
explicit forward update, still after a snapshot.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SNAPSHOT_NOTE = (
    "Node snapshot records custom nodes and Python dependencies. "
    "It does not roll back files under MODELS_DIR. "
    "Weight version switches stay on models select --keep or --wipe."
)

_SEMVER = re.compile(r"v?\d+\.\d+(?:\.\d+)?$")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,40}$")
_REQ = Path(__file__).resolve().parents[2] / "requirements.txt"
_PINS = Path(__file__).resolve().with_name("pack_pins.json")


class UpdateError(RuntimeError):
    """Opt-in Comfy update failed or was refused. Nothing was partially applied
    after a failed snapshot (the snapshot file, when written, is left in place).
    """


@dataclass(frozen=True)
class NodePin:
    name: str
    commit: str
    optional: bool = False


def read_comfy_cli_pin() -> str:
    """Exact ``comfy-cli==`` pin from ``requirements.txt``."""
    if not _REQ.is_file():
        raise UpdateError(f"requirements.txt missing at {_REQ}")
    for line in _REQ.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw.startswith("comfy-cli=="):
            pin = raw.split("==", 1)[1].strip()
            if pin:
                return pin
    raise UpdateError("requirements.txt has no comfy-cli== pin")


def _load_pin_file() -> dict[str, Any]:
    try:
        data = json.loads(_PINS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateError(f"pack pins unreadable ({exc})") from exc
    if not isinstance(data, dict):
        raise UpdateError("pack pins must be a JSON object")
    return data


def load_comfyui_pin() -> str | None:
    raw = _load_pin_file().get("comfyui")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def load_node_pins() -> list[NodePin]:
    nodes = _load_pin_file().get("nodes") or []
    if not isinstance(nodes, list):
        raise UpdateError("pack pins nodes must be a list")
    out: list[NodePin] = []
    for item in nodes:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        commit = str(item.get("commit") or item.get("version") or "").strip()
        if not name or not commit:
            continue
        out.append(NodePin(name=name, commit=commit, optional=bool(item.get("optional"))))
    return out


def pin_style(expected: str | None) -> str:
    text = (expected or "").strip()
    if not text:
        return "none"
    if _SEMVER.fullmatch(text):
        return "semver"
    if _COMMIT.fullmatch(text):
        return "commit"
    return "text"


def installed_comfy_cli_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version("comfy-cli")
    except PackageNotFoundError:
        return None


def read_git_head(repo: Path) -> str | None:
    """HEAD commit for ``repo``, or None when it is not a git checkout."""
    git = repo / ".git"
    if not git.exists():
        return None
    if git.is_file():
        raw = git.read_text(encoding="utf-8", errors="replace")
        gitdir = None
        for line in raw.splitlines():
            if line.startswith("gitdir:"):
                gitdir = line.split(":", 1)[1].strip()
                break
        if not gitdir:
            return None
        resolved = Path(gitdir)
        if not resolved.is_absolute():
            resolved = (repo / resolved).resolve()
        git = resolved
    head_file = git / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8", errors="replace").strip()
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        ref_file = git / ref
        if ref_file.is_file():
            sha = ref_file.read_text(encoding="utf-8", errors="replace").strip()
            return sha or None
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == ref:
                    return parts[0]
        return None
    if _COMMIT.fullmatch(head):
        return head
    return None


def read_comfyui_version(workspace: Path) -> str | None:
    """``[project].version`` from ``pyproject.toml`` when the line is a simple string."""
    path = workspace / "pyproject.toml"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not match:
        return None
    return match.group(1).strip() or None


def _workspace(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    from master_agent.comfy import tower as tower_mod

    st = tower_mod.load_state()
    raw = (st.workspace or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return tower_mod._default_workspace()


def _matches(installed: str | None, expected: str | None, style: str) -> bool:
    if not expected or not installed:
        return False
    if style == "commit":
        return installed.lower().startswith(expected.lower())
    if style == "semver":
        return installed.lstrip("v") == expected.lstrip("v")
    return installed == expected


def _item(
    *,
    kind: str,
    name: str,
    installed: str | None,
    expected: str | None,
    optional: bool,
    detail: str,
) -> dict[str, Any]:
    style = pin_style(expected)
    stale = False
    if expected and style != "none":
        if not installed:
            stale = not optional
        else:
            stale = not _matches(installed, expected, style)
    return {
        "kind": kind,
        "name": name,
        "installed": installed,
        "expected": expected,
        "optional": optional,
        "stale": stale,
        "pin_style": style,
        "detail": detail,
    }


def _scan_cli() -> dict[str, Any]:
    pin = read_comfy_cli_pin()
    installed = installed_comfy_cli_version()
    if installed is None:
        detail = f"comfy-cli is not installed; pin {pin}"
    elif installed == pin:
        detail = f"comfy-cli {installed} matches pin {pin}"
    else:
        detail = f"comfy-cli {installed}; pin {pin}"
    return _item(
        kind="cli",
        name="comfy-cli",
        installed=installed,
        expected=pin,
        optional=False,
        detail=detail,
    )


def _scan_core(workspace: Path) -> dict[str, Any]:
    expected = load_comfyui_pin()
    style = pin_style(expected)
    present = workspace.is_dir()
    head = read_git_head(workspace) if present else None
    version = read_comfyui_version(workspace) if present else None
    if style == "commit":
        installed = head
    elif style == "semver":
        installed = version
    else:
        installed = head or version
        if present and not installed:
            installed = "present"
    if not present:
        detail = "ComfyUI workspace is missing; no core pin. No install ran."
        installed = None
    elif not expected:
        shown = installed or "present"
        detail = f"ComfyUI {shown} has no core pin"
    elif _matches(installed, expected, style):
        detail = f"ComfyUI {installed} matches pin {expected}"
    else:
        detail = f"ComfyUI {installed or 'missing'}; pin {expected}"
    return _item(
        kind="comfyui",
        name="ComfyUI",
        installed=installed,
        expected=expected,
        optional=False,
        detail=detail,
    )


def _scan_nodes(workspace: Path) -> tuple[list[dict[str, Any]], list[str]]:
    pins = load_node_pins()
    pinned_names = {pin.name for pin in pins}
    nodes_dir = workspace / "custom_nodes"
    installed_names: list[str] = []
    if nodes_dir.is_dir():
        for child in sorted(nodes_dir.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and child.name != "__pycache__":
                installed_names.append(child.name)
    items: list[dict[str, Any]] = []
    for pin in pins:
        folder = nodes_dir / pin.name
        head = read_git_head(folder) if folder.is_dir() else None
        if folder.is_dir() and not head:
            installed: str | None = "present"
        else:
            installed = head
        style = pin_style(pin.commit)
        if not folder.is_dir():
            detail = f"{pin.name} is not installed"
            if pin.optional:
                detail += " (optional)"
        elif _matches(installed, pin.commit, style):
            detail = f"{pin.name} {installed} matches pin {pin.commit}"
        else:
            detail = f"{pin.name} {installed or 'missing'}; pin {pin.commit}"
        items.append(
            _item(
                kind="node",
                name=pin.name,
                installed=None if not folder.is_dir() else installed,
                expected=pin.commit,
                optional=pin.optional,
                detail=detail,
            )
        )
    unpinned = [name for name in installed_names if name not in pinned_names]
    return items, unpinned


def scan_packs(workspace: Path | None = None) -> dict[str, Any]:
    """Compare the local install with known pins. Does not call comfy-cli."""
    ws = _workspace(workspace)
    items = [_scan_cli(), _scan_core(ws)]
    node_items, unpinned = _scan_nodes(ws)
    items.extend(node_items)
    stale = [item for item in items if item["stale"]]
    return {
        "workspace": str(ws),
        "items": items,
        "stale": stale,
        "unpinned_nodes": unpinned,
        "updated": False,
    }


def format_report_lines(report: dict[str, Any]) -> list[str]:
    lines = ["Stale pack check (no update yet):"]
    for item in report.get("items") or []:
        mark = "STALE" if item.get("stale") else "OK"
        lines.append(f"  {mark:<5}  {item.get('detail')}")
    unpinned = report.get("unpinned_nodes") or []
    if unpinned:
        lines.append(f"  OK     {len(unpinned)} custom node(s) have no pin")
    if not report.get("stale"):
        lines.append("Nothing is stale versus known pins.")
    return lines


def plan_actions(
    report: dict[str, Any],
    *,
    core: bool = False,
    nodes: bool = False,
    cli: bool = False,
) -> dict[str, Any]:
    """Actions an opt-in would run. Commit-pinned nodes wait for ``--nodes``."""
    items = list(report.get("items") or [])
    cli_item = next(item for item in items if item["kind"] == "cli")
    core_item = next(item for item in items if item["kind"] == "comfyui")
    node_items = [item for item in items if item["kind"] == "node"]
    explicit = bool(core or nodes or cli)
    actions: list[dict[str, Any]] = []
    notes: list[str] = []

    def cli_action() -> dict[str, Any] | None:
        pin = str(cli_item.get("expected") or "")
        if cli_item.get("installed") and cli_item.get("installed") == pin and not cli_item.get("stale"):
            return None
        if not explicit and not cli_item.get("stale"):
            return None
        if explicit and not cli:
            return None
        return {
            "kind": "cli",
            "pin": pin,
            "argv": ["pip", "install", f"comfy-cli=={pin}"],
            "label": f"pip install comfy-cli=={pin}",
        }

    def core_action() -> dict[str, Any]:
        argv = ["update", "comfy"]
        label = "comfy update comfy"
        if core_item.get("pin_style") == "semver" and core_item.get("expected"):
            version = str(core_item["expected"]).lstrip("v")
            argv.extend(["--version", version])
            label = f"comfy update comfy --version {version}"
        return {"kind": "core", "argv": argv, "label": label}

    if explicit:
        if core:
            actions.append(core_action())
            if core_item.get("pin_style") == "commit":
                notes.append(
                    "comfy update comfy follows the workspace branch. "
                    f"It does not check out commit pin {core_item.get('expected')}."
                )
            elif not core_item.get("expected"):
                notes.append("ComfyUI has no core pin. comfy update comfy follows the workspace branch.")
        if nodes:
            actions.append(
                {
                    "kind": "nodes",
                    "argv": ["node", "update", "all"],
                    "label": "comfy node update all",
                }
            )
            notes.append(
                "comfy node update moves custom nodes forward. "
                "It does not check out commit pins. "
                "The pre-update snapshot is the rollback for nodes and Python dependencies."
            )
        chosen = cli_action() if cli else None
        if chosen:
            actions.append(chosen)
    else:
        if core_item.get("stale") and core_item.get("pin_style") == "semver":
            actions.append(core_action())
        elif core_item.get("stale") and core_item.get("pin_style") == "commit":
            notes.append(
                f"ComfyUI {core_item.get('installed') or 'missing'} does not match "
                f"commit pin {core_item.get('expected')}. "
                "Pass --core --yes to run comfy update comfy after a snapshot. "
                "That follows the workspace branch and does not check out the pin."
            )
        semver_nodes = [
            item for item in node_items if item.get("stale") and item.get("pin_style") == "semver"
        ]
        commit_nodes = [
            item for item in node_items if item.get("stale") and item.get("pin_style") == "commit"
        ]
        if semver_nodes:
            names = [str(item["name"]) for item in semver_nodes]
            actions.append(
                {
                    "kind": "nodes",
                    "argv": ["node", "update", *names],
                    "label": "comfy node update " + " ".join(names),
                }
            )
        if commit_nodes:
            listed = ", ".join(
                f"{item['name']} ({item.get('installed') or 'missing'} vs {item.get('expected')})"
                for item in commit_nodes
            )
            notes.append(
                f"Commit-pinned nodes are stale: {listed}. "
                "Pass --nodes --yes to move custom nodes forward after a snapshot. "
                "That does not check out the pinned commit."
            )
        chosen = cli_action()
        if chosen:
            actions.append(chosen)
    return {"actions": actions, "notes": notes}


def workspace_looks_installed(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in ("main.py", "comfy", ".git"))


def _snapshot_dir() -> Path:
    from master_agent.comfy import tower as tower_mod

    return Path(tower_mod.STATE_DIR) / "comfy_snapshots"


def new_snapshot_path() -> Path:
    directory = _snapshot_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = directory / f"pre-update-{stamp}.json"
    suffix = 2
    while dest.exists():
        dest = directory / f"pre-update-{stamp}-{suffix}.json"
        suffix += 1
    return dest


def take_snapshot(workspace: Path) -> Path:
    """``comfy node save-snapshot`` into Buddy state. Raises when the file is missing."""
    from master_agent.comfy import tower as tower_mod

    dest = new_snapshot_path()
    try:
        tower_mod._run_comfy(
            ["node", "save-snapshot", "--output", str(dest)],
            workspace=workspace,
            timeout=180.0,
            check=True,
        )
    except tower_mod.TowerError as exc:
        raise UpdateError(f"{exc} Update aborted. No Comfy or node update ran.") from exc
    if not dest.is_file():
        raise UpdateError(
            f"comfy node save-snapshot did not write {dest}. "
            "Update aborted. No Comfy or node update ran."
        )
    return dest


def _pip_install_pinned_cli(pin: str) -> list[str]:
    expected = read_comfy_cli_pin()
    if pin != expected:
        raise UpdateError(f"refusing to install comfy-cli=={pin}; requirements pin is {expected}")
    cmd = [sys.executable, "-m", "pip", "install", f"comfy-cli=={pin}"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateError(f"pip install comfy-cli=={pin} failed: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise UpdateError(f"pip install comfy-cli=={pin} failed: {err[:800]}")
    return cmd


def _refuse_external_tree_edit() -> None:
    from master_agent.comfy.tower import TowerError, effective_mode

    try:
        mode = effective_mode()
    except TowerError as exc:
        raise UpdateError(str(exc)) from exc
    if mode == "external":
        raise UpdateError(
            "comfy_mode=external: refusing to update a user-owned Comfy. "
            "Buddy will not snapshot, update ComfyUI, or update nodes in that tree."
        )


def execute_plan(plan: dict[str, Any], *, workspace: Path | None = None) -> dict[str, Any]:
    """Run an already-confirmed plan. Snapshot first when Comfy or nodes change."""
    actions = list(plan.get("actions") or [])
    ws = _workspace(workspace)
    commands: list[list[str]] = []
    snapshot = ""
    tree = [action for action in actions if action.get("kind") in {"core", "nodes"}]
    if tree:
        _refuse_external_tree_edit()
        if not workspace_looks_installed(ws):
            raise UpdateError(
                "Managed Comfy workspace is not installed. This command does not run comfy install."
            )
        snapshot_path = take_snapshot(ws)
        snapshot = str(snapshot_path)
        commands.append(["node", "save-snapshot", "--output", snapshot])
    from master_agent.comfy import tower as tower_mod

    for action in actions:
        kind = action.get("kind")
        if kind == "cli":
            pip_cmd = _pip_install_pinned_cli(str(action.get("pin") or ""))
            commands.append(pip_cmd)
            continue
        argv = list(action.get("argv") or [])
        if argv[:2] == ["update", "cli"]:
            raise UpdateError("refusing comfy update cli; comfy-cli stays on the requirements pin")
        try:
            tower_mod._run_comfy(argv, workspace=ws, timeout=600.0, check=True)
        except tower_mod.TowerError as exc:
            kept = f" Snapshot kept at {snapshot}." if snapshot else ""
            raise UpdateError(f"{exc}{kept}") from exc
        commands.append(argv)
    return {"snapshot": snapshot, "commands": commands, "workspace": str(ws)}


def _default_confirm(prompt: str) -> bool:
    from master_agent.setup import confirm_prompt

    return confirm_prompt(prompt)


def apply_update(
    *,
    yes: bool = False,
    core: bool = False,
    nodes: bool = False,
    cli: bool = False,
    workspace: Path | None = None,
    confirm: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Report, then update only when ``yes`` or ``confirm`` accepts."""
    report = scan_packs(workspace)
    plan = plan_actions(report, core=core, nodes=nodes, cli=cli)
    lines = format_report_lines(report)
    for note in plan["notes"]:
        lines.append(f"NOTE  {note}")
    base = {
        "updated": False,
        "report": report,
        "plan": plan,
        "lines": lines,
        "commands": [],
        "snapshot": "",
        "workspace": report["workspace"],
    }
    if not plan["actions"]:
        base["reason"] = "nothing-to-do"
        lines.append("No update ran.")
        return base
    allowed = bool(yes)
    if not allowed:
        ask = confirm or _default_confirm
        allowed = bool(ask("Update the stale Comfy packs listed above? [y/N] "))
    if not allowed:
        base["reason"] = "no-opt-in"
        lines.append("No update ran.")
        return base
    executed = execute_plan(plan, workspace=workspace or Path(report["workspace"]))
    lines.append(SNAPSHOT_NOTE)
    if executed.get("snapshot"):
        lines.append(f"Snapshot: {executed['snapshot']}")
    for action in plan["actions"]:
        lines.append(f"RAN   {action.get('label')}")
    lines.append("Update finished.")
    return {
        **base,
        "updated": True,
        "reason": "opt-in",
        "lines": lines,
        "commands": executed["commands"],
        "snapshot": executed.get("snapshot") or "",
    }


def startup_report(workspace: Path | None = None) -> dict[str, Any]:
    """Hardware sentence plus stale pins. ``updated`` is always false."""
    from master_agent.comfy.hardware import scan_hardware

    try:
        hardware = scan_hardware()
    except Exception as exc:
        hardware = {
            "ok": True,
            "blocks_install": False,
            "band": "unknown",
            "sentence": f"Hardware scan failed ({exc}). This scan does not block install.",
        }
    try:
        packs = scan_packs(workspace)
    except Exception as exc:
        packs = {"stale": [], "items": [], "unpinned_nodes": [], "error": str(exc)}
    lines = [f"Hardware: {hardware.get('sentence')}"]
    stale = packs.get("stale") or []
    if packs.get("error"):
        lines.append(f"Packs: scan failed ({packs['error']}). No update ran.")
    elif stale:
        names = ", ".join(str(item.get("name")) for item in stale)
        lines.append(f"Packs: stale {names}. No update ran.")
        lines.append("Opt in with: python -m master_agent comfy update --yes")
    else:
        lines.append("Packs: nothing stale vs known pins. No update ran.")
    return {
        "updated": False,
        "blocks_install": False,
        "hardware": hardware,
        "packs": packs,
        "lines": lines,
    }


def cmd_update(args: Any) -> int:
    """CLI handler for ``python -m master_agent comfy update``."""
    workspace = getattr(args, "workspace", None)
    ws = Path(workspace).expanduser().resolve() if workspace else None
    try:
        result = apply_update(
            yes=bool(getattr(args, "yes", False)),
            core=bool(getattr(args, "update_core", False)),
            nodes=bool(getattr(args, "update_nodes", False)),
            cli=bool(getattr(args, "update_cli", False)),
            workspace=ws,
        )
    except UpdateError as exc:
        print(f"FAIL  {exc}")
        return 1
    if bool(getattr(args, "as_json", False)):
        public = {
            "updated": result["updated"],
            "reason": result.get("reason"),
            "snapshot": result.get("snapshot") or "",
            "lines": result.get("lines") or [],
            "commands": result.get("commands") or [],
            "stale": [item.get("name") for item in (result.get("report") or {}).get("stale") or []],
            "notes": (result.get("plan") or {}).get("notes") or [],
        }
        print(json.dumps(public, indent=2))
        return 0
    for line in result.get("lines") or []:
        print(line)
    return 0
