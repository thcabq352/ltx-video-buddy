"""Doctor hardware routing. No GPU required. Install is never blocked.

Run: python -m pytest tests/test_hardware_scan.py -q
"""

from __future__ import annotations

from pathlib import Path

from master_agent.comfy.hardware import (
    GpuProbe,
    amd_sysfs_vram_gb,
    detect_gpu,
    parse_nvidia_smi,
    parse_rocm_name,
    parse_rocm_vram_gb,
    scan_hardware,
)
from master_agent.models.vram_policy import FULL_CHECKPOINT_MIN_GB, LOW_VRAM_MAX_GB, hardware_route, vram_band
from master_agent.setup import check_hardware, print_report, snapshot


def test_band_edges():
    assert FULL_CHECKPOINT_MIN_GB == 12.0
    assert LOW_VRAM_MAX_GB == 8.0
    assert vram_band(12) == "full"
    assert vram_band(16) == "full"
    assert vram_band(11.9) == "gguf"
    assert vram_band(8.1) == "gguf"
    assert vram_band(8) == "low"
    assert vram_band(4) == "low"
    assert vram_band(None) == "unknown"


def test_nvidia_routes_one_sentence_and_never_blocks():
    full = hardware_route(vendor="nvidia", vram_gb=12, gpu_name="RTX 3060")
    assert full.blocks_install is False
    assert full.sentence == "NVIDIA RTX 3060 12GB: full checkpoints are in range."

    mid = hardware_route(vendor="nvidia", vram_gb=10, gpu_name="NVIDIA GeForce RTX 3080")
    assert mid.band == "gguf"
    assert mid.blocks_install is False
    assert mid.sentence == (
        "NVIDIA GeForce RTX 3080 10GB: use GGUF. Sulphur GGUF can run on less."
    )

    low = hardware_route(vendor="nvidia", vram_gb=8, gpu_name="RTX 3070")
    assert low.band == "low"
    assert low.blocks_install is False
    assert low.sentence == (
        "NVIDIA RTX 3070 8GB: use GGUF or CPU, or upgrade. Sulphur GGUF can run on less."
    )


def test_amd_rocm_and_missing_rocm():
    rocm = hardware_route(vendor="amd", vram_gb=16, rocm=True, gpu_name="Radeon RX 6800")
    assert rocm.blocks_install is False
    assert rocm.sentence == (
        "AMD Radeon RX 6800 16GB with ROCm: slower, and it works; full checkpoints are in range."
    )
    small = hardware_route(vendor="amd", vram_gb=6, rocm=True, gpu_name="")
    assert "slower, and it works" in small.sentence
    assert "use GGUF or CPU, or upgrade." in small.sentence
    assert "Sulphur GGUF can run on less." in small.sentence

    bare = hardware_route(vendor="amd", vram_gb=16, rocm=False, gpu_name="Radeon")
    assert bare.blocks_install is False
    assert bare.sentence == "AMD Radeon 16GB without ROCm: install ROCm first, or use GGUF."
    tight = hardware_route(vendor="amd", vram_gb=8, rocm=False)
    assert "install ROCm first, or use GGUF." in tight.sentence
    assert "Sulphur GGUF can run on less." in tight.sentence


def test_missing_gpu_still_routes():
    missing = hardware_route(vendor="none", vram_gb=None, rocm=False)
    assert missing.blocks_install is False
    assert "No GPU detected" in missing.sentence
    assert "This scan does not block install." in missing.sentence
    assert "Sulphur GGUF can run on less." in missing.sentence
    numbered = hardware_route(vendor="unknown", vram_gb=16)
    assert numbered.band == "full"
    assert numbered.sentence.startswith("16GB reported with no identified GPU:")
    assert numbered.blocks_install is False


def test_nvidia_smi_and_rocm_parsers():
    parsed = parse_nvidia_smi("NVIDIA GeForce RTX 5060 Ti, 16384\n")
    assert parsed == ("NVIDIA GeForce RTX 5060 Ti", 16.0)
    assert parse_nvidia_smi("") is None
    name = parse_rocm_name("GPU[0] : Card series: AMD Radeon RX 7900 XTX\n")
    assert "7900" in name
    vram = parse_rocm_vram_gb("GPU[0] : VRAM Total Memory (B): 17179869184\n")
    assert vram == 16.0


def test_sysfs_amd_without_rocm(tmp_path: Path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n", encoding="utf-8")
    (card / "mem_info_vram_total").write_text(str(8 * 1024 ** 3), encoding="utf-8")
    (card / "product_name").write_text("Radeon Graphics\n", encoding="utf-8")
    found, gb, name = amd_sysfs_vram_gb(tmp_path)
    assert found is True
    assert gb == 8.0
    assert "Radeon" in name
    assert amd_sysfs_vram_gb(tmp_path / "missing") == (False, None, "")


def test_detect_prefers_nvidia_then_amd(monkeypatch):
    monkeypatch.setattr(
        "master_agent.comfy.hardware.shutil.which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )

    def fake_run(cmd, timeout=3.0):
        assert cmd[0] == "nvidia-smi"
        return 0, "NVIDIA RTX 4000 Ada, 12288"

    monkeypatch.setattr("master_agent.comfy.hardware._run_capture", fake_run)
    probe = detect_gpu()
    assert probe.vendor == "nvidia"
    assert probe.vram_gb == 12.0
    assert probe.rocm is False

    monkeypatch.setattr("master_agent.comfy.hardware.shutil.which", lambda _name: None)
    monkeypatch.setattr("master_agent.comfy.hardware.rocm_present", lambda: True)
    monkeypatch.setattr("master_agent.comfy.hardware._amd_probe", lambda: GpuProbe("amd", "Radeon", 8.0, "rocm-smi", True))
    assert detect_gpu().vendor == "amd"


def test_env_vram_overrides_probe_and_doctor_stays_ok(monkeypatch, capsys):
    monkeypatch.setenv("VRAM_GB", "10")
    scan = scan_hardware(GpuProbe("nvidia", "RTX 4090", 24.0, "nvidia-smi", False))
    assert scan["blocks_install"] is False
    assert scan["vram_source"] == "env"
    assert scan["band"] == "gguf"
    assert "10GB" in scan["sentence"]
    assert "use GGUF" in scan["sentence"]

    monkeypatch.delenv("VRAM_GB", raising=False)
    monkeypatch.setattr(
        "master_agent.comfy.hardware.detect_gpu",
        lambda: GpuProbe("nvidia", "RTX 3050", 6.0, "nvidia-smi", False),
    )
    row = check_hardware()
    assert row["name"] == "hardware"
    assert row["ok"] is True
    assert row["blocks_install"] is False
    assert row["band"] == "low"
    assert "upgrade" in row["detail"]
    names = {item["name"] for item in snapshot()}
    assert "hardware" in names
    assert "pack-pins" in names
    assert "vram-policy" in names
    assert print_report([row]) == 0
    text = capsys.readouterr().out
    assert "NEED" not in text
    assert "hardware" in text
