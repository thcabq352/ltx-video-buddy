"""Doctor-time CUDA free-memory safety net. No GPU: every probe is mocked.

Run: python -m pytest tests/test_vram_probe.py -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from master_agent.models import vram_policy
from master_agent.models.vram_policy import (
    NVFP4_MIN_VRAM_GB,
    CudaMemProbe,
    doctor_vram_gb,
    format_doctor_line,
    preference_order,
    probe_cuda_mem,
)
from master_agent.models.weights import TRANSFORMER_PREFERENCE

BF16 = TRANSFORMER_PREFERENCE[-1]


def _fake_torch(*, available: bool = True, mem=None, raises: Exception | None = None):
    def mem_get_info():
        if raises is not None:
            raise raises
        return mem

    cuda = types.SimpleNamespace(is_available=lambda: available, mem_get_info=mem_get_info)
    return types.SimpleNamespace(cuda=cuda)


@pytest.fixture
def declared_16(monkeypatch):
    monkeypatch.setattr("master_agent.config.VRAM_GB", 16.0)
    monkeypatch.setattr("master_agent.config.VRAM_SOURCE", "nvidia-smi")
    monkeypatch.setattr("master_agent.config.FORCE_LOADER", "")


def test_probe_none_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert probe_cuda_mem() is None


def test_probe_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert probe_cuda_mem() is None


def test_probe_none_when_mem_get_info_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(raises=RuntimeError("CUDA error")))
    assert probe_cuda_mem() is None


def test_probe_reports_gib(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mem=(9 * 2**30, 16 * 2**30)))
    assert probe_cuda_mem() == CudaMemProbe(free_gb=9.0, total_gb=16.0)


def test_failed_probe_keeps_declared_threshold():
    usable, mem = doctor_vram_gb(16.0, probe=lambda: None)
    assert (usable, mem) == (16.0, None)


def test_low_free_memory_caps_doctor_vram():
    usable, mem = doctor_vram_gb(16.0, probe=lambda: CudaMemProbe(free_gb=9.5, total_gb=16.0))
    assert usable == 9.5
    assert mem is not None and mem.total_gb == 16.0


def test_probe_never_raises_doctor_vram_above_declared():
    usable, _mem = doctor_vram_gb(12.0, probe=lambda: CudaMemProbe(free_gb=23.0, total_gb=24.0))
    assert usable == 12.0


def test_doctor_line_holds_heavy_back_on_shared_card(declared_16):
    line = format_doctor_line(probe=lambda: CudaMemProbe(free_gb=6.0, total_gb=16.0))
    assert "CUDA probe: 6.0GB free of 16.0GB" in line
    assert "holds NVFP4/bf16 back" in line
    assert f"not suggested below {NVFP4_MIN_VRAM_GB:.0f}GB" in line


def test_doctor_line_unchanged_when_probe_fails(declared_16):
    line = format_doctor_line(probe=lambda: None)
    assert "CUDA probe" not in line
    assert f"else NVFP4 (≥{NVFP4_MIN_VRAM_GB:.0f}GB)" in line


def test_doctor_line_allows_heavy_when_free_memory_fits(declared_16):
    line = format_doctor_line(probe=lambda: CudaMemProbe(free_gb=15.2, total_gb=16.0))
    assert "CUDA probe: 15.2GB free" in line
    assert "holds" not in line
    assert "else NVFP4" in line


def test_policy_order_ignores_probe(monkeypatch):
    def boom():
        raise AssertionError("preference_order must not probe CUDA")

    monkeypatch.setattr(vram_policy, "probe_cuda_mem", boom)
    order = preference_order(TRANSFORMER_PREFERENCE, vram_gb=16.0, force_loader="")
    assert BF16 in order


def test_ltx25_weights_row_flags_bf16_when_free_memory_is_low(tmp_path: Path, monkeypatch, declared_16):
    from master_agent.setup import check_ltx25_weights

    root = tmp_path / "models"
    bf16 = root / "diffusion_models" / BF16
    bf16.parent.mkdir(parents=True)
    bf16.write_bytes(b"bf16")
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [root])

    monkeypatch.setattr(vram_policy, "probe_cuda_mem", lambda: None)
    assert "not suggested" not in check_ltx25_weights()["detail"]

    monkeypatch.setattr(
        vram_policy, "probe_cuda_mem", lambda: CudaMemProbe(free_gb=7.0, total_gb=16.0)
    )
    assert "not suggested for this VRAM" in check_ltx25_weights()["detail"]
