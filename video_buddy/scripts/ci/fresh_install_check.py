#!/usr/bin/env python3
"""Checks the fresh-install CI job runs against a real ComfyUI.

Each subcommand exits 0 on success and 1 with a plain reason on failure, so a
workflow step can call it directly. Run from the repo root with Buddy's venv:

    python video_buddy/scripts/ci/fresh_install_check.py scan-log video_buddy/state/logs/comfyui_8188.log
    python video_buddy/scripts/ci/fresh_install_check.py wait-ready --out system_stats.json
    python video_buddy/scripts/ci/fresh_install_check.py install-packs --workspace video_buddy/ComfyUI
    python video_buddy/scripts/ci/fresh_install_check.py validate-workflow
    python video_buddy/scripts/ci/fresh_install_check.py pins --python <comfy python> --stress
    python video_buddy/scripts/ci/fresh_install_check.py manager-channel --comfy video_buddy/ComfyUI
    python video_buddy/scripts/ci/fresh_install_check.py manager-live
    python video_buddy/scripts/ci/fresh_install_check.py delete-manager-config --comfy video_buddy/ComfyUI
"""

from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

VIDEO_BUDDY = Path(__file__).resolve().parents[2]
if str(VIDEO_BUDDY) not in sys.path:
    sys.path.insert(0, str(VIDEO_BUDDY))

DEFAULT_URL = "http://127.0.0.1:8188"
COMFY_REQUIREMENTS = VIDEO_BUDDY / "comfy-requirements.txt"
KLEIN_WORKFLOW = VIDEO_BUDDY / "workflows" / "260615_VIDEO-BUDDY_FLUX_KLEIN_9B_V01_api.json"

# Packs the Klein stills graph needs beyond what automatic-install puts in
# (ComfyUI-LTXVideo, ComfyUI-GGUF). Comfy registry ids. RES4LYF registers the
# res_2s sampler and beta57 scheduler the graph's KSampler uses.
# comfyui-mickmumpitz-nodes 1.5.9 is flagged in the registry and will not install.
KLEIN_PACKS = (
    "rgthree-comfy",
    "comfyui-kjnodes",
    "comfyui-easy-use",
    "comfyui-mickmumpitz-nodes@1.5.8",
    "RES4LYF",
)

# Neighbours of the pinned packages: gguf's own dependencies plus pip tooling.
UNRELATED_UPGRADES = ("numpy", "pyyaml", "requests", "tqdm", "setuptools")
TORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})

# Errors a runner without weights or the operator's input images always has.
INPUT_LOADERS = frozenset({"LoadImage", "LoadImageMask", "LoadVideo", "LoadAudio"})

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
_REQ_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!-]+)$")


class CheckFailed(Exception):
    pass


# --- ComfyUI log ---------------------------------------------------------------


def scan_log(text: str, *, allow_missing_modules: Iterable[str] = ()) -> list[str]:
    """Lines that fail the job: InvalidChannel, ModuleNotFoundError, IMPORT FAILED."""
    allowed = {m.split(".")[0] for m in allow_missing_modules}
    problems: list[str] = []
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).strip()
        if "InvalidChannel" in line or "An invalid channel was used" in line:
            problems.append(line)
        elif "IMPORT FAILED" in line:
            problems.append(line)
        else:
            m = _MISSING_MODULE.search(line)
            if m and m.group(1).split(".")[0] not in allowed:
                problems.append(line)
    return problems


# --- HTTP ----------------------------------------------------------------------


def get_json(url: str, *, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(url: str, *, timeout_s: float, interval_s: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = "not tried"
    while True:
        try:
            return get_json(f"{url}/system_stats", timeout=10)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last = str(exc)
        if time.monotonic() >= deadline:
            raise CheckFailed(f"{url}/system_stats did not answer within {timeout_s:.0f}s ({last})")
        time.sleep(interval_s)


# --- workflow vs live /object_info ---------------------------------------------


def workflow_classes(workflow: dict[str, Any]) -> set[str]:
    return {str(n["class_type"]) for n in workflow.values() if isinstance(n, dict) and n.get("class_type")}


def missing_classes(workflow: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    return sorted(c for c in workflow_classes(workflow) if c not in object_info)


def is_environment_only(issue: Any, workflow: dict[str, Any]) -> bool:
    """A weight file or operator input image the runner does not have; never a graph defect."""
    message = str(issue.message)
    if message.startswith("model file '"):
        return True
    node = workflow.get(str(issue.node_id))
    class_type = node.get("class_type") if isinstance(node, dict) else None
    return class_type in INPUT_LOADERS and "not in combo choices" in message


def split_errors(report: Any, workflow: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    blocking, env_only = [], []
    for issue in report.errors:
        (env_only if is_environment_only(issue, workflow) else blocking).append(issue)
    return blocking, env_only


def check_workflow(path: Path, object_info: dict[str, Any]) -> list[str]:
    """Report lines; raises CheckFailed when the graph does not fit this ComfyUI."""
    from master_agent.comfy.validator import format_report, validate_workflow, validate_workflow_file

    workflow = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"{path.name}: {len(workflow_classes(workflow))} node classes, {len(object_info)} in live /object_info"]
    missing = missing_classes(workflow, object_info)
    if missing:
        raise CheckFailed("node classes missing from live /object_info: " + ", ".join(missing))
    live = validate_workflow(copy.deepcopy(workflow), object_info, file_label=str(path), object_info_source="live")
    blocking, env_only = split_errors(live, workflow)
    lines += [f"  not on this runner (ok): {i}" for i in env_only]
    lines += [f"  {i}" for i in live.warnings]
    if blocking:
        raise CheckFailed("live validation failed:\n" + "\n".join(f"  {i}" for i in blocking))
    offline = validate_workflow_file(path, prefer_live=False)
    lines.append("offline: " + format_report(offline).replace("\n", "\n  "))
    if not offline.ok:
        raise CheckFailed("offline validation (cached object_info) failed:\n" + format_report(offline))
    return lines


# --- node packs ----------------------------------------------------------------


def pack_folder(spec: str) -> str:
    return spec.split("@", 1)[0]


def missing_pack_folders(workspace: Path, packs: Iterable[str]) -> list[str]:
    folder = workspace / "custom_nodes"
    have = {p.name.lower() for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
    return [pack_folder(p) for p in packs if pack_folder(p).lower() not in have]


def install_packs(workspace: Path, packs: Iterable[str]) -> str:
    from master_agent.automatic_install import comfy_argv
    from master_agent.comfy.tower import _child_env

    packs = list(packs)
    argv = comfy_argv(workspace, "node", "install", *packs)
    print("$ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, env=_child_env(workspace), capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    still = missing_pack_folders(workspace, packs)
    if still:
        raise CheckFailed(f"not installed: {', '.join(still)}\n{out[-2000:]}")
    return out


# --- pinned Comfy packages -----------------------------------------------------


def read_pins(path: Path = COMFY_REQUIREMENTS) -> dict[str, str]:
    """``{name: version}``. Every requirement line must be an exact ``name==version`` pin."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_PIN.match(line)
        if not m:
            raise CheckFailed(f"{path.name}: {line!r} is not an exact name==version pin")
        pins[m.group(1).lower()] = m.group(2)
    if not pins:
        raise CheckFailed(f"{path.name} has no requirements")
    return pins


def installed_versions(python: str, names: Iterable[str]) -> dict[str, str | None]:
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out = {}\n"
        "for n in sys.argv[1:]:\n"
        "    try:\n"
        "        out[n] = m.version(n)\n"
        "    except m.PackageNotFoundError:\n"
        "        out[n] = None\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run([python, "-c", code, *names], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise CheckFailed(f"{python} could not report versions: {proc.stderr.strip()[-500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def pin_mismatches(pins: dict[str, str], installed: dict[str, str | None]) -> list[str]:
    out = []
    for name, want in sorted(pins.items()):
        have = installed.get(name)
        if have != want:
            out.append(f"{name}: pinned {want}, installed {have or 'nothing'}")
    return out


def without_torch(requirements_text: str) -> str:
    """ComfyUI's requirements minus torch*: ``--upgrade`` would swap the CPU wheels for multi-GB CUDA ones."""
    keep = []
    for raw in requirements_text.splitlines():
        name = re.split(r"[<>=!~\[;\s]", raw.strip(), maxsplit=1)[0].lower()
        if name in TORCH_PACKAGES:
            continue
        keep.append(raw)
    return "\n".join(keep) + "\n"


def _pip(python: str, *args: str) -> None:
    cmd = [python, "-m", "pip", "install", *args]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise CheckFailed(f"pip failed: {' '.join(cmd)}\n{(proc.stdout + proc.stderr)[-2000:]}")


def check_pins(python: str, requirements: Path, *, stress: bool, comfy_requirements: Path | None) -> list[str]:
    pins = read_pins(requirements)
    lines = ["pins: " + ", ".join(f"{k}=={v}" for k, v in sorted(pins.items()))]

    def assert_pinned(when: str) -> None:
        bad = pin_mismatches(pins, installed_versions(python, pins))
        if bad:
            raise CheckFailed(f"{when}: " + "; ".join(bad))
        lines.append(f"ok {when}")

    assert_pinned("after install")
    if not stress:
        return lines
    _pip(python, "--upgrade", *UNRELATED_UPGRADES)
    assert_pinned("after pip install --upgrade " + " ".join(UNRELATED_UPGRADES))
    if comfy_requirements is not None:
        with tempfile.TemporaryDirectory() as tmp:
            filtered = Path(tmp) / "comfy-reqs-no-torch.txt"
            filtered.write_text(without_torch(comfy_requirements.read_text(encoding="utf-8")), encoding="utf-8")
            _pip(python, "--upgrade", "-r", str(filtered))
        assert_pinned(f"after pip install --upgrade -r {comfy_requirements.name} (torch excluded)")
    _pip(python, "--upgrade", "-r", str(requirements))
    assert_pinned(f"after automatic-install's pip install --upgrade -r {requirements.name}")
    return lines


# --- ComfyUI-Manager channel ---------------------------------------------------


def manager_files(code_dir: Path) -> list[Path]:
    from master_agent.comfy.manager_config import manager_dirs

    _primary, dirs = manager_dirs(code_dir)
    return [d / name for d in dirs for name in ("config.ini", "channels.list") if (d / name).is_file()]


def channel_problems(code_dir: Path) -> list[str]:
    """Empty when Manager will read channel_url = Comfy-Org and accept it."""
    from master_agent.comfy.manager_config import BROKEN_CHANNEL_URLS, CHANNEL_URL, manager_dirs

    primary, dirs = manager_dirs(code_dir)
    problems: list[str] = []
    for d in dirs:
        cfg = d / "config.ini"
        if cfg.is_file() and any(b in _config_channel(cfg) for b in BROKEN_CHANNEL_URLS):
            problems.append(f"{cfg}: channel_url is {_config_channel(cfg)}")
        listed = d / "channels.list"
        if listed.is_file():
            problems += [f"{listed}: {line.strip()}" for line in listed.read_text(encoding="utf-8").splitlines()
                         if any(b in line for b in BROKEN_CHANNEL_URLS)]
    cfg = primary / "config.ini"
    if not cfg.is_file():
        return problems + [f"{cfg} does not exist"]
    url = _config_channel(cfg).rstrip("/")
    if url != CHANNEL_URL:
        problems.append(f"{cfg}: channel_url is {url or 'missing'}, want {CHANNEL_URL}")
    listed = primary / "channels.list"
    urls = []
    if listed.is_file():
        urls = [line.split("::", 1)[1].strip().rstrip("/") for line in listed.read_text(encoding="utf-8").splitlines()
                if line.count("::") == 1]
    if CHANNEL_URL not in urls:
        problems.append(f"{listed}: {CHANNEL_URL} is not a listed channel, so Manager raises InvalidChannel")
    return list(dict.fromkeys(problems))


def _config_channel(path: Path) -> str:
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(path, encoding="utf-8")
    return cp.get("default", "channel_url", fallback="").strip() if cp.has_section("default") else ""


def snapshot(code_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(code_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in manager_files(code_dir)
    }


def live_channel_problems(payload: dict[str, Any]) -> list[str]:
    """``GET /v2/manager/channel_url_list``: Manager's own view of channel_url vs channels.list."""
    from master_agent.comfy.manager_config import CHANNEL_URL

    selected = payload.get("selected")
    entries = dict(item.split("::", 1) for item in payload.get("list") or [] if item.count("::") == 1)
    if selected == "custom" or selected not in entries:
        return [f"Manager reports channel {selected!r}: channel_url is not in channels.list (InvalidChannel)"]
    if entries[selected].rstrip("/") != CHANNEL_URL:
        return [f"Manager's selected channel {selected!r} is {entries[selected]}, want {CHANNEL_URL}"]
    return []


def delete_manager_config(code_dir: Path) -> Path:
    from master_agent.comfy.manager_config import manager_dirs

    primary, _ = manager_dirs(code_dir)
    cfg = primary / "config.ini"
    if not cfg.is_file():
        raise CheckFailed(f"{cfg} does not exist; nothing to reset")
    cfg.unlink()
    return cfg


# --- CLI -----------------------------------------------------------------------


def _cmd_scan_log(a: argparse.Namespace) -> list[str]:
    path = Path(a.log)
    if not path.is_file():
        raise CheckFailed(f"{path} does not exist")
    problems = scan_log(path.read_text(encoding="utf-8", errors="replace"), allow_missing_modules=a.allow_missing_module)
    if problems:
        raise CheckFailed(f"{path}:\n" + "\n".join(f"  {p}" for p in problems))
    return [f"{path}: no InvalidChannel, ModuleNotFoundError or IMPORT FAILED"]


def _cmd_wait_ready(a: argparse.Namespace) -> list[str]:
    stats = wait_ready(a.url, timeout_s=a.timeout)
    if a.out:
        Path(a.out).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    system = stats.get("system") or {}
    devices = ", ".join(str(d.get("type")) for d in stats.get("devices") or [])
    return [f"{a.url}/system_stats ok: ComfyUI {system.get('comfyui_version')}, devices: {devices or 'none'}"]


def _cmd_validate(a: argparse.Namespace) -> list[str]:
    object_info = get_json(f"{a.url}/object_info", timeout=120)
    if a.object_info_out:
        Path(a.object_info_out).write_text(json.dumps(object_info), encoding="utf-8")
    return check_workflow(Path(a.workflow), object_info)


def _cmd_install_packs(a: argparse.Namespace) -> list[str]:
    install_packs(Path(a.workspace), KLEIN_PACKS)
    return ["installed " + ", ".join(KLEIN_PACKS)]


def _cmd_pins(a: argparse.Namespace) -> list[str]:
    comfy_reqs = Path(a.comfy_requirements) if a.comfy_requirements else None
    return check_pins(a.python, Path(a.requirements), stress=a.stress, comfy_requirements=comfy_reqs)


def _cmd_manager_channel(a: argparse.Namespace) -> list[str]:
    code = Path(a.comfy)
    problems = channel_problems(code)
    if problems:
        raise CheckFailed("\n".join(problems))
    lines = [f"{p}: ok" for p in manager_files(code)]
    snap = snapshot(code)
    if a.compare:
        before = json.loads(Path(a.compare).read_text(encoding="utf-8"))
        if before != snap:
            raise CheckFailed(f"Manager files changed since {a.compare}: {before} -> {snap}")
        lines.append(f"unchanged since {a.compare}")
    if a.save:
        Path(a.save).write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return lines


def _cmd_manager_live(a: argparse.Namespace) -> list[str]:
    for route in ("/v2/manager/channel_url_list", "/manager/channel_url_list"):
        try:
            payload = get_json(a.url + route)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    else:
        raise CheckFailed("ComfyUI-Manager is not loaded (no channel_url_list route); launch with --enable-manager")
    problems = live_channel_problems(payload)
    if problems:
        raise CheckFailed("\n".join(problems))
    return [f"Manager channel: {payload.get('selected')} ({route})"]


def _cmd_delete(a: argparse.Namespace) -> list[str]:
    return [f"deleted {delete_manager_config(Path(a.comfy))}"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan-log", help="fail on InvalidChannel, ModuleNotFoundError, IMPORT FAILED")
    s.add_argument("log")
    s.add_argument("--allow-missing-module", action="append", default=[],
                   help="a module the job deliberately skips (repeatable)")
    s.set_defaults(func=_cmd_scan_log)

    s = sub.add_parser("wait-ready", help="poll /system_stats")
    s.add_argument("--url", default=DEFAULT_URL)
    s.add_argument("--timeout", type=float, default=180.0)
    s.add_argument("--out")
    s.set_defaults(func=_cmd_wait_ready)

    s = sub.add_parser("validate-workflow", help="graph vs live /object_info, then offline validation")
    s.add_argument("--url", default=DEFAULT_URL)
    s.add_argument("--workflow", default=str(KLEIN_WORKFLOW))
    s.add_argument("--object-info-out")
    s.set_defaults(func=_cmd_validate)

    s = sub.add_parser("install-packs", help="comfy node install the Klein stills packs")
    s.add_argument("--workspace", required=True)
    s.set_defaults(func=_cmd_install_packs)

    s = sub.add_parser("pins", help="assert comfy-requirements.txt pins are installed")
    s.add_argument("--python", default=sys.executable)
    s.add_argument("--requirements", default=str(COMFY_REQUIREMENTS))
    s.add_argument("--stress", action="store_true", help="upgrade unrelated packages, then re-check")
    s.add_argument("--comfy-requirements", help="ComfyUI's requirements.txt to reinstall with --upgrade")
    s.set_defaults(func=_cmd_pins)

    s = sub.add_parser("manager-channel", help="config.ini / channels.list on Comfy-Org, never ltdrdata")
    s.add_argument("--comfy", required=True, help="ComfyUI code folder")
    s.add_argument("--save", help="write a hash snapshot of the Manager files")
    s.add_argument("--compare", help="fail unless the files match this snapshot")
    s.set_defaults(func=_cmd_manager_channel)

    s = sub.add_parser("manager-live", help="ask the running Manager which channel it selected")
    s.add_argument("--url", default=DEFAULT_URL)
    s.set_defaults(func=_cmd_manager_live)

    s = sub.add_parser("delete-manager-config", help="delete Manager's config.ini")
    s.add_argument("--comfy", required=True)
    s.set_defaults(func=_cmd_delete)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines = args.func(args)
    except CheckFailed as exc:
        print(f"FAIL  {args.command}: {exc}", flush=True)
        return 1
    for line in lines:
        print(line)
    print(f"PASS  {args.command}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
