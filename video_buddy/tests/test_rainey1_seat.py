"""Seat rule, Klein 4B stills, VRAM profile, first gate, detached handoff.

No GPU and no network. nvidia-smi, llama-server, and Hub downloads are injected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from master_agent.gpu_seat import (
    SeatRefused,
    ServerView,
    acquire_for_submit,
    layers_use_gpu,
    release_after_render,
    reset_state,
)
from master_agent.models.klein import (
    DIFFUSION_FILE,
    LICENSE_LINE,
    KleinRefused,
    accept_hub_model,
    assert_allowed_name,
    download_klein_4b,
)
from master_agent.orchestrator.flux_ltx import plan, uses_klein_4b
from master_agent.process_detach import detached_popen_kwargs
from master_agent.vram_env import apply_vram_env, plan_vram_env


@pytest.fixture(autouse=True)
def _seat(monkeypatch):
    reset_state()
    monkeypatch.delenv("SEAT_REQUIRE_PROBE", raising=False)
    monkeypatch.setenv("SEAT_MIN_FREE_GB", "6")
    yield
    reset_state()
    monkeypatch.setenv("SEAT_MIN_FREE_GB", "0")


def test_gpu_llm_is_stopped_and_low_vram_refuses():
    stopped = []
    view = ServerView(running=True, gpu_layers=32, owned=True, argv=["llama-server", "-ngl", "32"])
    with pytest.raises(SeatRefused) as exc:
        acquire_for_submit(
            purpose="still",
            view=view,
            free_gb=1.5,
            stop=lambda current: stopped.append(current.gpu_layers),
        )
    assert stopped == [32]
    assert "SEAT RULE:" in str(exc.value)
    assert "Nothing was queued" in str(exc.value)


def test_missing_nvidia_smi_still_unloads_and_allows():
    stopped = []
    view = ServerView(running=True, gpu_layers=None, owned=True)
    line = acquire_for_submit(
        purpose="clip",
        view=view,
        free_gb=None,
        free_probe=lambda: None,
        stop=lambda current: stopped.append(current),
    )
    assert stopped and "nvidia-smi did not report" in line


def test_restart_is_cpu_when_card_is_under_20gb(monkeypatch):
    started = []
    monkeypatch.delenv("LLAMACPP_NGL", raising=False)
    monkeypatch.setattr("master_agent.config.VRAM_GB", 12)
    view = ServerView(running=True, gpu_layers=24, owned=True)
    acquire_for_submit(view=view, free_gb=10, stop=lambda _v: None)
    note = release_after_render(start=lambda ngl: started.append(ngl), free_probe=lambda: 10)
    assert started == [0]
    assert "CPU" in (note or "")


def test_restart_keeps_gpu_layers_on_a_20gb_card(monkeypatch):
    started = []
    monkeypatch.delenv("LLAMACPP_NGL", raising=False)
    monkeypatch.setattr("master_agent.config.VRAM_GB", 24)
    view = ServerView(running=True, gpu_layers=16, owned=True)
    acquire_for_submit(view=view, free_gb=20, stop=lambda _v: None)
    release_after_render(start=lambda ngl: started.append(ngl), free_probe=lambda: 12)
    assert started == [16]
    assert layers_use_gpu(16) and not layers_use_gpu(0)


def test_klein_refuses_gated_9b_and_flux1():
    with pytest.raises(KleinRefused, match="Gated"):
        accept_hub_model({"id": "black-forest-labs/FLUX.2-klein-4B", "gated": True, "cardData": {"license": "apache-2.0"}})
    with pytest.raises(KleinRefused):
        assert_allowed_name("black-forest-labs/FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors")
    with pytest.raises(KleinRefused):
        assert_allowed_name("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors")
    with pytest.raises(KleinRefused):
        assert_allowed_name("black-forest-labs/FLUX.3-klein", "flux3.safetensors")
    accept_hub_model(
        {
            "id": "black-forest-labs/FLUX.2-klein-4b-fp8",
            "gated": False,
            "cardData": {"license": "apache-2.0"},
        }
    )


def test_klein_download_skips_a_full_file(tmp_path: Path):
    fetched = []

    def fetch(*, repo_id, repo_filename, dest, progress):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        fetched.append(repo_filename)
        return dest

    from master_agent.models import klein as klein_mod

    spec = klein_mod.FILES[0]
    dest = tmp_path / spec.dest_folder / spec.filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"\0" * spec.size_bytes)
    download_klein_4b(lambda _m: None, root=tmp_path, fetch=fetch)
    assert spec.repo_filename not in fetched
    assert len(fetched) == 2


def test_vram_profile_12_16_24_does_not_overwrite(tmp_path: Path):
    twelve = plan_vram_env(12, {})
    assert twelve.updates["LLAMACPP_NGL"] == "0"
    assert twelve.updates["VRAM_GB"] == "12"
    assert twelve.updates["FORCE_LOADER"] == "gguf"
    sixteen = plan_vram_env(16, {})
    assert sixteen.updates == {"LLAMACPP_NGL": "0"}
    assert any("bf16" in note for note in sixteen.notes)
    big = plan_vram_env(24, {})
    assert "LLAMACPP_NGL" not in big.updates
    kept = plan_vram_env(12, {"VRAM_GB": "16"})
    assert "VRAM_GB" not in kept.updates
    assert any("left your VRAM_GB=16" in note for note in kept.notes)
    path = tmp_path / ".env"
    path.write_text("VRAM_GB=16\n", encoding="utf-8")
    apply_vram_env(12, path, apply=True)
    text = path.read_text(encoding="utf-8")
    assert "VRAM_GB=16" in text
    assert "LLAMACPP_NGL=0" in text


def test_first_gate_blocks_until_judge_and_clip(tmp_path, monkeypatch):
    from master_agent import first_gate

    monkeypatch.delenv("FIRST_GATE", raising=False)
    monkeypatch.setattr(first_gate, "state_path", lambda: tmp_path / "first_gate.json")
    with pytest.raises(first_gate.FirstGateBlocked, match="GPU is not free"):
        first_gate.run_first_gate(
            still_runner=lambda: {"judge_fired": True},
            clip_runner=lambda: {"ok": True},
            judge_probe=lambda: {"fired": True},
            vram_free_gb=1.0,
        )
    with pytest.raises(first_gate.FirstGateBlocked, match="did not fire"):
        first_gate.run_first_gate(
            still_runner=lambda: {"judge_fired": True},
            clip_runner=lambda: {"ok": True},
            judge_probe=lambda: {"fired": False, "reason": "off"},
            vram_free_gb=12,
        )
    with pytest.raises(first_gate.FirstGateBlocked):
        first_gate.assert_submit_allowed({"1": {"class_type": "KSampler"}})
    result = first_gate.run_first_gate(
        still_runner=lambda: {"judge_fired": True},
        clip_runner=lambda: {"ok": True},
        judge_probe=lambda: {"fired": True, "provider": "llamacpp"},
        vram_free_gb=12,
    )
    assert result["status"] == "passed"
    assert first_gate.is_passed()


def test_detached_flags_are_not_a_thread():
    flags = detached_popen_kwargs()
    if os.name == "nt":
        import subprocess

        assert flags["creationflags"] & subprocess.DETACHED_PROCESS
        assert flags["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert flags["start_new_session"] is True
    assert "ThreadPoolExecutor" not in Path("/workspace/video_buddy/streamlit_app.py").read_text(encoding="utf-8")


def test_look_check_uses_llamacpp_and_records_a_fire(monkeypatch):
    from master_agent.judge import local_look

    monkeypatch.setattr("master_agent.config.JUDGE_ENABLED", True)
    monkeypatch.setattr("master_agent.config.VISION_ENABLED", True)
    monkeypatch.setattr("master_agent.config.VISION_MODEL", "qwen3-vl-heretic")
    seen = []
    result = local_look.fire_look_check(kind="still", ping=lambda kind: seen.append(kind) or {"fired": True})
    assert result["fired"] is True and result["provider"] == "llamacpp"
    assert seen == ["still"]
    assert local_look.look_provider_spec().startswith("llamacpp:")


def test_license_line_is_in_the_install_banner():
    from master_agent.install_report import light_install_banner

    text = light_install_banner(where="start")
    assert "Apache 2.0" in text
    assert "only Flux" in text
    assert LICENSE_LINE.startswith("FLUX.2 [klein] 4B")


def test_flux_ltx_i2v_uses_klein_4b():
    spec = plan("a cup")
    assert spec["slug"] == "flux_ltx_i2v"
    assert spec["still_generator"] == DIFFUSION_FILE
    assert spec["video_variant"] == "ltx25_t2v_i2v"
    assert uses_klein_4b(spec)
    assert "flux1" not in spec["still_generator"]


def test_sdxl_workflows_still_resolve():
    root = Path("/workspace/video_buddy/workflows")
    names = [
        "Video-Buddy_AI-RENDERING_SDXL_ADV_v04.json",
        "Video-Buddy_AI-RENDERING_SDXL_ADV_v06.json",
        "Video-Buddy_AI-RENDERING_SDXL_ADV_v07.json",
    ]
    for name in names:
        text = (root / name).read_text(encoding="utf-8")
        assert "SDXL" in text or "sdxl" in text.lower()


def test_z_image_turbo_download_and_policy_still_resolve():
    from master_agent.models.vram_policy import _OTHER
    import importlib.util

    path = Path("/workspace/video_buddy/state/download_models.py")
    spec = importlib.util.spec_from_file_location("vb_download_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blob = "\n".join(" ".join(row) for row in mod.DOWNLOADS)
    assert "Comfy-Org/z_image_turbo" in blob
    assert "z_image_turbo_bf16.safetensors" in blob
    assert "Z-Image-Turbo-Fun-Controlnet-Union-2.1" in blob
    assert "z_image_turbo_bf16.safetensors" in _OTHER
    assert "flux1" not in blob
    assert "klein-9" not in blob.lower()


def test_flux1_is_not_a_download_target():
    from master_agent.config import FLUX_UNET, MODEL_FILES
    from master_agent.models.weights import WEIGHT_FILES

    assert FLUX_UNET == DIFFUSION_FILE
    assert MODEL_FILES["flux"]["diffusion"] == DIFFUSION_FILE
    assert "flux1" not in WEIGHT_FILES["flux"].filename
    assert "flux1" not in WEIGHT_FILES["flux"].repo_id
    assert WEIGHT_FILES["flux"].gated is False
