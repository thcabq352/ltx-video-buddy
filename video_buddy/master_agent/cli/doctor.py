"""Doctor, inventory, health, and model scan commands."""

from __future__ import annotations

import argparse
import json

from master_agent.cli.runtime import comfy_client
from master_agent.comfy.client import ComfyClientError
from master_agent.models.inventory import format_summary, scan_inventory
from master_agent.cli.common import _weight_mode

def cmd_setup(args: argparse.Namespace) -> int:
    from master_agent.setup import cmd_setup as run_setup

    return run_setup(
        do_fix=bool(args.fix),
        fix_models=bool(getattr(args, "fix_models", False)),
        mode=_weight_mode(args),
        yes=bool(getattr(args, "yes", False)),
    )


def cmd_inventory(args: argparse.Namespace) -> int:
    """List discovered weights (paths + roles) before doctor suggests downloads."""
    from master_agent.config import COMFYUI_ROOT, MODELS_DIR
    from master_agent.models.inventory import format_listing, scan_inventory

    inv = scan_inventory(MODELS_DIR, COMFYUI_ROOT, write=False)
    if getattr(args, "json", False):
        print(json.dumps(inv.to_dict(), indent=1))
        return 0
    print(format_listing(inv))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from master_agent.llm import format_local_llm_health

    client = comfy_client()
    rc = 0
    try:
        stats = client.health()
    except ComfyClientError as e:
        print(f"FAIL  {e}")
        rc = 1
        print(format_local_llm_health(), end="")
        return rc
    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    print(f"OK    ComfyUI at {client.base_url}")
    print(f"      os={system.get('os')} python={system.get('python_version')}")
    for dev in devices:
        name = dev.get("name") or "?"
        vram_total = (dev.get("vram_total") or 0) / 1e9
        vram_free = (dev.get("vram_free") or 0) / 1e9
        print(f"      gpu={name} vram={vram_free:.1f}G free / {vram_total:.1f}G")
    print(format_local_llm_health(), end="")
    return rc


def cmd_scan_models(args: argparse.Namespace) -> int:
    inv = scan_inventory()
    print(format_summary(inv))
    bad = [v for v, i in inv.bundles.items() if not i.get("runnable")]
    return 1 if bad and args.strict else 0

