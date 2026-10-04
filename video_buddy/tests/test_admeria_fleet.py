"""Admeria / rainey1 fleet: 12GB loader, no surprise pulls, scan-only.

No GPU required. Run: python -m pytest tests/test_admeria_fleet.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from master_agent.models.inventory import check_bundles, scan_inventory
from master_agent.models.vram_policy import format_doctor_line, pack_kind
from master_agent.models.weights import (
    WEIGHT_FILES,
    WeightStatus,
    download_files,
    find_ltx23_compatible,
    resolve_weight,
    transformer_preference_order,
)
from master_agent.setup import install_ollama_models, ollama_has_model


def _write(path: Path, blob: bytes = b"weight") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _download_ns(**overrides) -> Namespace:
    base = dict(
        bundle=None,
        wan=False,
        vace=False,
        krea=False,
        qwen=False,
        flux_pack=False,
        h3=False,
        ltx25=True,
        json=False,
        yes=False,
        optional=False,
        scan_only=False,
        use_existing=False,
        download=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_ollama_has_model_matches_tags():
    listed = "NAME ID SIZE MODIFIED\nqwen3-vl-heretic:latest abc 4.1 GB\nnomic-embed-text:latest def 274 MB\n"
    assert ollama_has_model(listed, "qwen3-vl-heretic")
    assert ollama_has_model(listed, "nomic-embed-text")
    assert not ollama_has_model(listed, "missing-model")


def test_install_ollama_skips_models_already_listed(monkeypatch, capsys):
    calls: list[list[str]] = []

    monkeypatch.setattr("master_agent.setup._which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)

    def fake_run(cmd, timeout=120):
        calls.append(list(cmd))
        if cmd[:2] == ["ollama", "list"]:
            return 0, (
                "NAME ID SIZE\n"
                "qwen3-vl-heretic:latest abc 4 GB\n"
                "nomic-embed-text:latest def 274 MB\n"
            )
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr("master_agent.setup._run", fake_run)
    install_ollama_models(consent=True)
    out = capsys.readouterr().out
    assert "already in ollama list" in out
    assert all(cmd[:2] != ["ollama", "pull"] for cmd in calls)


def test_install_ollama_missing_model_needs_confirmation(monkeypatch, capsys):
    calls: list[list[str]] = []

    monkeypatch.setattr("master_agent.setup._which", lambda name: "/usr/bin/ollama")

    def fake_run(cmd, timeout=120):
        calls.append(list(cmd))
        if cmd[:2] == ["ollama", "list"]:
            return 0, "NAME ID SIZE\n"
        if cmd[:2] == ["ollama", "pull"]:
            raise AssertionError("pull must not run without confirmation")
        return 0, ""

    monkeypatch.setattr("master_agent.setup._run", fake_run)
    monkeypatch.setattr("master_agent.setup.confirm_prompt", lambda _prompt: False)
    install_ollama_models(consent=False)
    out = capsys.readouterr().out
    assert "no confirmation" in out
    assert all(cmd[:2] != ["ollama", "pull"] for cmd in calls)


def test_vram_12_and_force_gguf_hide_nvfp4_and_bf16():
    forced = transformer_preference_order(vram_gb=12, force_loader="gguf")
    assert forced
    assert forced[0].endswith(".gguf")
    assert all(pack_kind(name).startswith("gguf") for name in forced)
    plain = transformer_preference_order(vram_gb=12, force_loader="")
    assert plain[0].endswith(".gguf")
    # Q4 GGUF filenames may contain "bf16" (source dtype). Reject heavy packs, not that substring.
    assert all(pack_kind(name) not in {"nvfp4", "bf16", "fp16"} for name in plain)
    assert all("nvfp4" not in name.lower() for name in plain)
    assert pack_kind("sulphur_dev-Q3_K_S.gguf") == "gguf_q4"


def test_doctor_line_reports_12gb(monkeypatch):
    monkeypatch.setattr("master_agent.config.VRAM_GB", 12.0)
    monkeypatch.setattr("master_agent.config.VRAM_SOURCE", "env")
    monkeypatch.setattr("master_agent.config.FORCE_LOADER", "gguf")
    line = format_doctor_line()
    assert "12" in line
    assert "GGUF" in line
    assert "not suggested" in line.lower()
    assert "FORCE_LOADER=gguf" in line


def test_sulphur_q3_counts_for_ltx23_not_ltx25(tmp_path: Path):
    dest = _write(tmp_path / "unet" / "sulphur_dev-Q3_K_S.gguf", b"sulphur")
    assert find_ltx23_compatible([tmp_path]) == dest
    assert resolve_weight(WEIGHT_FILES["transformer"], [tmp_path]) is None
    bundles = check_bundles(
        scan_inventory(tmp_path, tmp_path / "no-comfy", write=False).entries
    )
    assert "diffusion" in bundles["base"]["present"]


def test_heretic_encoder_variant_skips_download(tmp_path: Path, monkeypatch):
    te = _write(tmp_path / "text_encoders" / "gemma-3-12b-it-heretic.safetensors", b"te")
    logs: list[str] = []

    def boom(*_a, **_k):
        raise AssertionError("hub download must not start when a heretic encoder exists")

    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [tmp_path])
    monkeypatch.setattr("master_agent.models.download.download_hub_file", boom)
    found = resolve_weight(WEIGHT_FILES["text_encoder"], [tmp_path])
    assert found == te
    paths = download_files(
        [WEIGHT_FILES["text_encoder"]],
        dest_root=tmp_path / "dest",
        progress=logs.append,
        yes=True,
    )
    assert paths == [te]
    assert "local file found" in "\n".join(logs).lower()


def test_scan_only_does_not_fetch(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("fetch must not run under --scan-only")

    missing = WeightStatus(
        bundle="ltx25_all",
        missing_mandatory=[WEIGHT_FILES["transformer"]],
    )
    monkeypatch.setattr("master_agent.models.weights.scan_bundle", lambda *_a, **_k: missing)
    monkeypatch.setattr("master_agent.models.weights.download_missing_bundle", boom)
    monkeypatch.setattr("master_agent.models.weights.download_named_file", boom)
    monkeypatch.setattr(
        "master_agent.models.weights.ic_ingredients_placement",
        lambda *_a, **_k: {"ok": True, "detail": "skipped", "fix": "", "path": ""},
    )
    from master_agent.__main__ import cmd_download_models

    rc = cmd_download_models(_download_ns(scan_only=True, yes=True))
    out = capsys.readouterr().out
    assert rc == 2
    assert "scan-only" in out
    assert "nothing downloaded" in out.lower()


def test_use_existing_does_not_fetch(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("fetch must not run under --use-existing")

    missing = WeightStatus(
        bundle="ltx25_all",
        missing_mandatory=[WEIGHT_FILES["audio_vae"]],
    )
    monkeypatch.setattr("master_agent.models.weights.scan_bundle", lambda *_a, **_k: missing)
    monkeypatch.setattr("master_agent.models.weights.download_missing_bundle", boom)
    monkeypatch.setattr("master_agent.models.weights.download_named_file", boom)
    monkeypatch.setattr(
        "master_agent.models.weights.ic_ingredients_placement",
        lambda *_a, **_k: {"ok": False, "detail": "missing ic", "fix": "", "path": ""},
    )
    from master_agent.__main__ import cmd_download_models

    rc = cmd_download_models(_download_ns(use_existing=True, yes=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "use-existing" in out


def test_download_without_yes_does_not_fetch(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("fetch must not run when the prompt is declined")

    missing = WeightStatus(
        bundle="ltx25_all",
        missing_mandatory=[WEIGHT_FILES["video_vae"]],
    )
    monkeypatch.setattr("master_agent.models.weights.scan_bundle", lambda *_a, **_k: missing)
    monkeypatch.setattr("master_agent.models.weights.download_missing_bundle", boom)
    monkeypatch.setattr("master_agent.models.weights.download_named_file", boom)
    monkeypatch.setattr("master_agent.setup.confirm_prompt", lambda _prompt: False)
    monkeypatch.setattr(
        "master_agent.models.weights.ic_ingredients_placement",
        lambda *_a, **_k: {"ok": True, "detail": "ok", "fix": "", "path": "x"},
    )
    from master_agent.__main__ import cmd_download_models

    rc = cmd_download_models(_download_ns(download=True, yes=False))
    assert rc == 2
    assert "nothing downloaded" in capsys.readouterr().out.lower()


def test_inventory_command_lists_role_and_path(tmp_path: Path, monkeypatch, capsys):
    dest = _write(tmp_path / "models" / "unet" / "sulphur_dev-Q3_K_S.gguf", b"sulphur")
    monkeypatch.setattr("master_agent.config.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("master_agent.config.COMFYUI_ROOT", tmp_path / "comfy")
    from master_agent.__main__ import cmd_inventory

    rc = cmd_inventory(Namespace(json=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "sulphur_dev-Q3_K_S.gguf" in out
    assert "diffusion" in out
    assert str(dest) in out
    assert "nothing is downloaded" in out.lower()

    rc_json = cmd_inventory(Namespace(json=True))
    blob = capsys.readouterr().out
    assert rc_json == 0
    assert "sulphur_dev-Q3_K_S.gguf" in blob
    assert '"role": "diffusion"' in blob
