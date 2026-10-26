"""Opt-in Klein Qwen encoder and 360 ERP LoRA. No GPU and no network."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from master_agent.models.klein import (
    DIFFUSION_BYTES,
    DIFFUSION_FILE,
    DIFFUSION_REPO,
    DIFFUSION_SHA256,
    FILES,
    TEXT_BYTES,
    TEXT_FILE,
    TEXT_REPO_FILE,
    TEXT_SHA256,
    VAE_BYTES,
    VAE_FILE,
    VAE_SHA256,
)
from master_agent.models.llm_gguf import LLM_SLOTS
from master_agent.models.weights import (
    BUNDLES,
    HF_KLEIN_360_LORA,
    KLEIN_360_LORA_BYTES,
    KLEIN_360_LORA_SHA256,
    WEIGHT_FILES,
    download_verified_file,
    files_for_bundle,
    hub_resolve_url,
    weight_destination,
)

_DEFAULT_BUNDLES = (
    "ltx23_core",
    "ltx25_core",
    "ltx25_all",
    "ltx25_flux_i2v",
    "flux",
)
_LORA = "flux-2-klein-4B-360-erp-outpaint-lora_V1.safetensors"


def _default_automatic_filenames() -> set[str]:
    names = {spec.filename for spec in FILES}
    names.update(weight.filename for weight in files_for_bundle("ltx23_core"))
    for slot in LLM_SLOTS.values():
        names.update(spec.filename for spec in slot.files)
    return names


def test_fp8_unet_comes_from_black_forest_and_vae_size_is_pinned():
    fp8 = next(spec for spec in FILES if spec.filename == DIFFUSION_FILE)
    vae = next(spec for spec in FILES if spec.filename == VAE_FILE)
    assert fp8.repo_id == "black-forest-labs/FLUX.2-klein-4b-fp8"
    assert fp8.repo_id == DIFFUSION_REPO
    assert fp8.size_bytes == 4_070_624_520 == DIFFUSION_BYTES
    assert fp8.repo_filename == DIFFUSION_FILE
    assert DIFFUSION_SHA256.startswith("97ed34fe")
    assert "Comfy-Org" not in fp8.repo_id
    assert vae.repo_id == "Comfy-Org/Flux-2-Klein-4B"
    assert vae.size_bytes == 336_211_292 == VAE_BYTES
    assert VAE_SHA256.startswith("868fe7b3")
    flux = WEIGHT_FILES["flux"]
    assert flux.repo_id == DIFFUSION_REPO
    assert flux.filename == DIFFUSION_FILE
    assert flux.size_bytes == DIFFUSION_BYTES
    assert flux.dest_folder == "diffusion_models"


def test_qwen_encoder_stays_in_the_default_klein_pack_and_has_a_standalone_row():
    packed = next(spec for spec in FILES if spec.filename == TEXT_FILE)
    assert packed.repo_id == "Comfy-Org/Flux-2-Klein-4B"
    assert packed.repo_filename == TEXT_REPO_FILE
    assert packed.dest_folder == "text_encoders"
    assert packed.size_bytes == 3_848_213_998 == TEXT_BYTES
    assert TEXT_SHA256 == "3eab03a77adb0ee5304a4e677d5c10ac22f9049c1d7c894adca4f8bb39206ca8"
    row = WEIGHT_FILES["klein_qwen_encoder"]
    assert row.mandatory is False
    assert row.filename == TEXT_FILE
    assert row.repo_id == packed.repo_id
    assert row.repo_filename == TEXT_REPO_FILE
    assert row.size_bytes == TEXT_BYTES
    assert row.sha256 == TEXT_SHA256
    assert row.dest_folder == "text_encoders"
    assert row.gated is False
    for bundle in _DEFAULT_BUNDLES:
        assert "klein_qwen_encoder" not in BUNDLES[bundle]
    assert TEXT_FILE in _default_automatic_filenames()


def test_360_lora_is_opt_in_and_outside_the_default_set():
    row = WEIGHT_FILES["klein_360_lora"]
    assert row.mandatory is False
    assert row.filename == _LORA
    assert row.dest_folder == "loras"
    assert row.repo_id == HF_KLEIN_360_LORA == "nomadoor/flux-2-klein-4B-360-erp-outpaint-lora"
    assert row.repo_filename == _LORA
    assert row.size_bytes == 46_224_184 == KLEIN_360_LORA_BYTES
    assert row.sha256 == KLEIN_360_LORA_SHA256
    assert row.sha256.startswith("567cc22d") and row.sha256.endswith("de166")
    assert row.gated is False
    assert _LORA not in {spec.filename for spec in FILES}
    assert _LORA not in _default_automatic_filenames()
    for bundle in _DEFAULT_BUNDLES:
        assert "klein_360_lora" not in BUNDLES[bundle]
        assert _LORA not in {weight.filename for weight in files_for_bundle(bundle)}


def test_legacy_bulk_downloader_keeps_the_lora_out_and_the_fp8_repo():
    path = Path(__file__).resolve().parents[1] / "state" / "download_models.py"
    spec = importlib.util.spec_from_file_location("vb_state_download_models", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    repo_files = [(row[0], row[1]) for row in mod.DOWNLOADS]
    assert not any("360-erp-outpaint" in name for _repo, name in repo_files)
    fp8 = [row for row in repo_files if row[1].endswith("flux-2-klein-4b-fp8.safetensors")]
    assert fp8 == [("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors")]
    encoder = [row for row in repo_files if row[1].endswith(TEXT_FILE)]
    assert encoder == [("Comfy-Org/Flux-2-Klein-4B", TEXT_REPO_FILE)]
    vae = [row for row in repo_files if row[1].endswith("flux2-vae.safetensors")]
    assert vae == [("Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors")]


def test_opt_in_rows_resolve_to_text_encoders_and_loras():
    root = Path("/models")
    encoder = WEIGHT_FILES["klein_qwen_encoder"]
    lora = WEIGHT_FILES["klein_360_lora"]
    assert weight_destination(encoder, root) == root / "text_encoders" / TEXT_FILE
    assert weight_destination(lora, root) == root / "loras" / _LORA
    assert hub_resolve_url(encoder) == (
        "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/"
        "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors"
    )
    assert hub_resolve_url(lora) == (
        "https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora/resolve/main/"
        + _LORA
    )


def test_verified_download_skips_a_full_file_and_refuses_without_consent(tmp_path, monkeypatch):
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [tmp_path])
    small = replace(WEIGHT_FILES["klein_360_lora"], size_bytes=4)
    dest = weight_destination(small, tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"abcd")

    def boom(*_a, **_k):
        raise AssertionError("download_resolve must not run")

    monkeypatch.setattr("master_agent.models.llm_gguf.download_resolve", boom)
    got = download_verified_file(small, dest_root=tmp_path, yes=True, progress=lambda _m: None)
    assert got == dest

    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)
    assert download_verified_file(
        WEIGHT_FILES["klein_qwen_encoder"], dest_root=tmp_path, yes=False, progress=lambda _m: None
    ) is None
    assert not (tmp_path / "text_encoders" / TEXT_FILE).exists()


def test_verified_download_passes_size_sha_and_destination(tmp_path, monkeypatch):
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)
    seen: dict = {}

    def fake(url, dest, size, *, sha256="", progress=print, **_kw):
        seen.update(url=url, dest=Path(dest), size=size, sha256=sha256)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
        return dest

    monkeypatch.setattr("master_agent.models.llm_gguf.download_resolve", fake)
    lora = WEIGHT_FILES["klein_360_lora"]
    got = download_verified_file(lora, dest_root=tmp_path, yes=True, progress=lambda _m: None)
    assert seen["url"] == hub_resolve_url(lora)
    assert seen["dest"] == tmp_path / "loras" / _LORA
    assert seen["size"] == KLEIN_360_LORA_BYTES
    assert seen["sha256"] == KLEIN_360_LORA_SHA256
    assert got == seen["dest"]

    encoder = WEIGHT_FILES["klein_qwen_encoder"]
    download_verified_file(encoder, dest_root=tmp_path, yes=True, progress=lambda _m: None)
    assert seen["dest"] == tmp_path / "text_encoders" / TEXT_FILE
    assert seen["size"] == TEXT_BYTES
    assert seen["sha256"] == TEXT_SHA256
    assert "split_files/text_encoders/" in seen["url"]


def test_download_models_flags_do_not_select_the_default_bundle(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("default bundle scan must not run")

    monkeypatch.setattr("master_agent.models.weights.scan_bundle", boom)
    monkeypatch.setattr("master_agent.models.weights.resolve_weight", lambda *_a, **_k: None)
    monkeypatch.setattr("master_agent.setup.print_inventory_preamble", lambda: None)
    from master_agent.cli.models import cmd_download_models

    rc = cmd_download_models(
        Namespace(
            bundle=None,
            wan=False,
            vace=False,
            krea=False,
            qwen=False,
            flux_pack=False,
            ltx23=False,
            h3=False,
            ltx25=True,
            heartmula=False,
            selector=False,
            json=False,
            yes=False,
            optional=False,
            scan_only=False,
            use_existing=False,
            download=False,
            with_360_lora=True,
            with_qwen_encoder=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert _LORA in out
    assert TEXT_FILE in out
    assert "models/loras/" in out
    assert "models/text_encoders/" in out
    assert KLEIN_360_LORA_SHA256 in out
    assert "Nothing downloaded" in out


def test_automatic_install_plan_omits_opt_in_unless_the_flag_is_set(monkeypatch, capsys):
    from master_agent import automatic_install as ai

    monkeypatch.setattr(ai, "run_preflight", lambda **_k: ai.Preflight(checks=[], gpu={"vendor": "cpu"}))
    monkeypatch.setattr(ai, "print_preflight", lambda _p: None)
    monkeypatch.setattr(ai, "managed_workspace", lambda: Path("/tmp/vb-comfy-unused"))
    for name in (
        "step_env_file",
        "step_vram_profile",
        "step_playwright",
        "step_ffmpeg",
        "step_comfy_cli",
        "step_comfyui",
        "step_model_folders",
        "step_model_paths",
        "step_nodes",
        "step_node_requirements",
        "step_comfy_packages",
        "step_manager_config",
        "step_accel",
        "step_weights",
        "step_klein_4b",
        "step_llamacpp",
        "step_llm_models",
    ):
        monkeypatch.setattr(ai, name, lambda *a, _n=name, **k: ai.StepResult(_n, ai.SKIPPED, "stub"))

    base = dict(
        yes=False,
        preflight_only=False,
        dry_run=True,
        gpu="cpu",
        skip_weights=True,
        skip_sage=True,
        skip_llm=True,
    )
    assert ai.cmd_automatic_install(Namespace(**base)) == 0
    plain = capsys.readouterr().out
    assert "360-erp-outpaint" not in plain
    assert "Klein 360 LoRA" not in plain
    assert "Qwen Klein encoder" not in plain

    assert ai.cmd_automatic_install(Namespace(**base, with_360_lora=True, with_qwen_encoder=True)) == 0
    flagged = capsys.readouterr().out
    assert "Klein 360 LoRA" in flagged
    assert _LORA in flagged
    assert "loras" in flagged
    assert "Qwen Klein encoder" in flagged
    assert TEXT_FILE in flagged
    assert "text_encoders" in flagged


def test_install_py_forwards_opt_in_flags(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "vb_install_opt_in", Path(__file__).resolve().parents[1] / "install.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("")
    (venv / "bin" / "python").write_text("")
    monkeypatch.setattr(mod, "VENV", venv)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "_venv_python", lambda: venv / "bin" / "python")
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["install.py", "--automatic-install", "--yes", "--with-qwen-encoder", "--with-360-lora"],
    )
    assert mod.main() == 0
    forwarded = calls[-1]
    assert forwarded[1:] == [
        "-m",
        "master_agent",
        "automatic-install",
        "--yes",
        "--with-qwen-encoder",
        "--with-360-lora",
    ]
