"""GPU / VRAM / ROCm scan for doctor and managed-Comfy startup.

The sentence comes from ``vram_policy.hardware_route``. This module only
detects the card. It does not install ROCm, Comfy, or weights, and it does
not fail setup when the card is small or missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.models.vram_policy import hardware_route

_CARD_DIR = re.compile(r"card\d+")


@dataclass(frozen=True)
class GpuProbe:
    """One detected device. ``vendor`` is ``nvidia``, ``amd``, or ``none``."""

    vendor: str
    name: str = ""
    vram_gb: float | None = None
    vram_source: str = "none"
    rocm: bool = False


def _run_capture(cmd: list[str], timeout: float = 3.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def parse_nvidia_smi(text: str) -> tuple[str, float] | None:
    """First ``name, memory.total`` row from ``nvidia-smi`` CSV output."""
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or "," not in raw:
            continue
        if raw.lower().startswith("name"):
            continue
        name, mem = raw.split(",", 1)
        token = mem.strip().split()[0] if mem.strip() else ""
        try:
            mib = float(token)
        except ValueError:
            continue
        if mib <= 0:
            continue
        return name.strip(), round(mib / 1024.0, 1)
    return None


def parse_rocm_name(text: str) -> str:
    for key in ("card series", "marketing name", "card model"):
        for line in (text or "").splitlines():
            if key not in line.lower() or ":" not in line:
                continue
            value = line.split(":", 1)[-1].strip()
            if value and not value.lower().startswith("0x"):
                return value
    return ""


def parse_rocm_vram_gb(text: str) -> float | None:
    """VRAM total from ``rocm-smi --showmeminfo vram`` text."""
    for line in (text or "").splitlines():
        low = line.lower()
        if "vram" not in low or "total" not in low:
            continue
        nums = re.findall(r"(\d+)", line)
        if not nums:
            continue
        raw = float(nums[-1])
        if raw > 1024 * 1024:
            return round(raw / (1024 ** 3), 1)
        if raw > 256:
            return round(raw / 1024.0, 1)
        if raw > 0:
            return raw
    return None


def rocm_present() -> bool:
    """True when the ROCm tools or ``ROCM_PATH`` are on this machine."""
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return True
    base = Path((os.environ.get("ROCM_PATH") or "/opt/rocm").strip() or "/opt/rocm")
    return (base / "bin" / "rocminfo").is_file() or (base / "bin" / "rocm-smi").is_file()


def amd_sysfs_vram_gb(root: Path | None = None) -> tuple[bool, float | None, str]:
    """``(is_amd, vram_gb, product_name)`` from sysfs. Missing sysfs is not AMD."""
    drm = Path(root) if root is not None else Path("/sys/class/drm")
    if not drm.is_dir():
        return False, None, ""
    for card in sorted(drm.glob("card*")):
        if not _CARD_DIR.fullmatch(card.name):
            continue
        vendor = card / "device" / "vendor"
        if not vendor.is_file():
            continue
        code = vendor.read_text(encoding="utf-8", errors="replace").strip().lower()
        if code not in {"0x1002", "1002"}:
            continue
        vram = None
        mem = card / "device" / "mem_info_vram_total"
        if mem.is_file():
            try:
                raw = float(mem.read_text(encoding="utf-8", errors="replace").strip())
            except ValueError:
                raw = 0.0
            if raw > 0:
                vram = round(raw / (1024 ** 3), 1)
        name = ""
        product = card / "device" / "product_name"
        if product.is_file():
            name = product.read_text(encoding="utf-8", errors="replace").strip()
        return True, vram, name
    return False, None, ""


def _nvidia_probe() -> GpuProbe | None:
    if not shutil.which("nvidia-smi"):
        return None
    code, out = _run_capture(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if code != 0:
        return None
    parsed = parse_nvidia_smi(out)
    if parsed is None:
        return None
    name, gb = parsed
    return GpuProbe(vendor="nvidia", name=name, vram_gb=gb, vram_source="nvidia-smi", rocm=False)


def _amd_probe() -> GpuProbe:
    rocm = rocm_present()
    name = ""
    vram: float | None = None
    source = "rocm" if rocm else "sysfs"
    if shutil.which("rocm-smi"):
        code, out = _run_capture(["rocm-smi", "--showproductname"])
        if code == 0:
            name = parse_rocm_name(out)
        code, mem_out = _run_capture(["rocm-smi", "--showmeminfo", "vram"])
        if code == 0:
            vram = parse_rocm_vram_gb(mem_out)
            if vram is not None:
                source = "rocm-smi"
    found, sys_vram, sys_name = amd_sysfs_vram_gb()
    if not name and sys_name:
        name = sys_name
    if vram is None and sys_vram is not None:
        vram = sys_vram
        source = "sysfs"
    if not found and not rocm:
        return GpuProbe(vendor="none", name="", vram_gb=None, vram_source="none", rocm=False)
    return GpuProbe(vendor="amd", name=name, vram_gb=vram, vram_source=source, rocm=rocm)


def detect_gpu() -> GpuProbe:
    """NVIDIA when ``nvidia-smi`` answers, else AMD from ROCm or sysfs, else none."""
    nvidia = _nvidia_probe()
    if nvidia is not None:
        return nvidia
    if rocm_present():
        return _amd_probe()
    found, _, _ = amd_sysfs_vram_gb()
    if found:
        return _amd_probe()
    return GpuProbe(vendor="none", name="", vram_gb=None, vram_source="none", rocm=False)


def _env_vram() -> tuple[float | None, str | None]:
    raw = (os.getenv("VRAM_GB") or "").strip()
    if not raw:
        return None, None
    try:
        return float(raw), "env"
    except ValueError:
        return None, None


def scan_hardware(probe: GpuProbe | None = None) -> dict[str, Any]:
    """Detect the card and return one routing sentence. Install is never blocked."""
    try:
        found = probe if probe is not None else detect_gpu()
    except Exception as exc:
        sentence = f"Hardware scan failed ({exc}). This scan does not block install."
        return {
            "ok": True,
            "blocks_install": False,
            "vendor": "unknown",
            "name": "",
            "vram_gb": None,
            "vram_source": "error",
            "rocm": False,
            "band": "unknown",
            "sentence": sentence,
        }
    vram = found.vram_gb
    source = found.vram_source
    override, override_source = _env_vram()
    if override_source:
        vram = override
        source = override_source
    route = hardware_route(
        vendor=found.vendor,
        vram_gb=vram,
        rocm=bool(found.rocm),
        gpu_name=found.name,
    )
    return {
        "ok": True,
        "blocks_install": False,
        "vendor": route.vendor,
        "name": found.name,
        "vram_gb": vram,
        "vram_source": source,
        "rocm": bool(found.rocm),
        "band": route.band,
        "sentence": route.sentence,
    }
