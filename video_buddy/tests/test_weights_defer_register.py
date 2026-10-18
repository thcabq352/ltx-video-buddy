"""Fix 3: Xet/CAS retry, --skip-weights deferral (INFO, not FAIL), ``models register``.

No network, no real weights: pack sizes are shrunk and files live under tmp_path.

Run: python -m pytest tests/test_weights_defer_register.py -q
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent import automatic_install as ai
from master_agent import config
from master_agent import setup as setup_mod
from master_agent.comfy import tower
from master_agent.models import deferred, download, register, weights
from master_agent.install_report import FAIL, INFO

LTX23_KEYS = (
    "ltx23_checkpoint",
    "ltx23_gguf",
    "ltx23_text_encoder",
    "ltx23_distilled_lora",
    "ltx23_video_vae",
    "ltx23_tiny_vae",
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = tmp_path / "state"
    state.mkdir()
    models = tmp_path / "models"
    comfy = tmp_path / "ComfyUI"
    monkeypatch.setattr(tower, "STATE_DIR", state)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(ai, "models_dir", lambda: models)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(comfy))
    monkeypatch.setattr(weights, "model_search_roots", lambda: [models, comfy / "models"])
    sizes = {}
    for i, key in enumerate(LTX23_KEYS):
        small = dataclasses.replace(weights.WEIGHT_FILES[key], size_bytes=100 + i)
        monkeypatch.setitem(weights.WEIGHT_FILES, key, small)
        sizes[key] = small
    return SimpleNamespace(tmp=tmp_path, state=state, models=models, comfy=comfy, files=sizes)


def _make(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _make_pack(folder: Path, files) -> list[Path]:
    return [_make(folder / w.filename, w.size_bytes) for w in files.values()]


# --- Xet / CAS retry -----------------------------------------------------------


def test_is_xet_error_matches_message_and_cause_chain():
    assert download.is_xet_error(RuntimeError("CAS service error: reconstruction failed"))
    assert download.is_xet_error(OSError("hf_xet download failed"))
    try:
        try:
            raise ValueError("Xet storage unreachable")
        except ValueError as inner:
            raise RuntimeError("download failed") from inner
    except RuntimeError as outer:
        assert download.is_xet_error(outer)
    assert not download.is_xet_error(OSError("Connection reset by peer"))
    assert not download.is_xet_error(RuntimeError("404 Not Found"))


def _hub_patches(monkeypatch, tmp_path: Path, calls: list[bool], fail_with: list[BaseException]):
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.setattr(download, "_find_existing_local", lambda *a: None)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **k: None)
    monkeypatch.setattr(download, "_configured_model_roots", lambda: [tmp_path])
    monkeypatch.setattr(download, "assert_pack_download_destination", lambda dest: dest)
    blob = _make(tmp_path / "hub" / "blob", 3)

    def fake_hub(*, repo_id, filename):
        calls.append(constants.HF_HUB_DISABLE_XET)
        if fail_with:
            raise fail_with.pop(0)
        return str(blob)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub)


def test_download_retries_once_with_xet_disabled(tmp_path, monkeypatch):
    import os

    calls: list[bool] = []
    _hub_patches(monkeypatch, tmp_path, calls, [RuntimeError("CAS service error: xet reconstruction")])
    log: list[str] = []
    dest = tmp_path / "models" / "vae" / "x.safetensors"
    got = download.download_hub_file(repo_id="r/x", repo_filename="x.safetensors", dest=dest, progress=log.append)
    assert got == dest and dest.read_bytes() == b"\0\0\0"
    assert calls == [False, True]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert any("HF_HUB_DISABLE_XET=1" in line for line in log)


def test_download_does_not_retry_other_errors(tmp_path, monkeypatch):
    calls: list[bool] = []
    _hub_patches(monkeypatch, tmp_path, calls, [OSError("Connection reset by peer")])
    with pytest.raises(OSError, match="Connection reset"):
        download.download_hub_file(
            repo_id="r/x", repo_filename="x.safetensors", dest=tmp_path / "m" / "x.safetensors", progress=lambda s: None
        )
    assert calls == [False]


def test_download_does_not_loop_when_xet_already_disabled(tmp_path, monkeypatch):
    calls: list[bool] = []
    _hub_patches(monkeypatch, tmp_path, calls, [RuntimeError("xet failed")])
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    with pytest.raises(RuntimeError, match="xet failed"):
        download.download_hub_file(
            repo_id="r/x", repo_filename="x.safetensors", dest=tmp_path / "m" / "x.safetensors", progress=lambda s: None
        )
    assert len(calls) == 1


# --- --skip-weights deferral ---------------------------------------------------


def test_skip_weights_apply_saves_marker_and_reports_info(env):
    res = ai.step_weights(True, skip=True)
    assert res.status == ai.WEIGHTS_DEFERRED
    assert not res.failed_required
    assert ai.classify_step(res) == INFO
    assert any("models register" in line for line in res.lines)
    marker = json.loads((env.state / deferred.FILENAME).read_text())
    assert marker["bundle"] == "ltx23_core" and marker["reason"] == "--skip-weights"
    assert len(marker["missing"]) == 6


def test_skip_weights_dry_run_writes_nothing(env):
    res = ai.step_weights(False, skip=True)
    assert res.status == ai.WEIGHTS_DEFERRED
    assert not (env.state / deferred.FILENAME).exists()


def test_weights_present_clears_marker(env):
    deferred.mark_deferred(["x"])
    for w in env.files.values():
        _make(env.models / w.dest_folder / w.filename, w.size_bytes)
    res = ai.step_weights(True, skip=True)
    assert res.status == ai.SKIPPED
    assert not deferred.weights_deferred()


def test_without_marker_missing_ltx23_row_still_fails(env):
    row = setup_mod.check_ltx23_weights()
    assert not row["ok"] and not row.get("deferred")
    assert setup_mod.classify_row(row) == FAIL


def test_doctor_row_is_info_when_deferred(env, capsys):
    deferred.mark_deferred(["LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf"])
    row = setup_mod.check_ltx23_weights()
    assert row["deferred"] is True and not row["ok"]
    assert "deferred (--skip-weights" in row["detail"]
    assert "models register" in row["fix"]
    assert setup_mod.classify_row(row) == INFO
    rc = setup_mod.print_report([row])
    out = capsys.readouterr().out
    assert rc == 0
    assert "INFO  ltx23-weights" in out
    assert "safe to ignore because you deferred the LTX 2.3 weights" in out


def test_automatic_install_with_skip_weights_exits_zero(env, monkeypatch, capsys):
    from argparse import Namespace

    ok = lambda name: (lambda *a, **k: ai.StepResult(name, ai.OK, "ready"))  # noqa: E731
    for fn, name in (
        ("step_env_file", ".env"), ("step_playwright", "Playwright Chromium"), ("step_ffmpeg", "ffmpeg"),
        ("step_comfy_cli", "comfy-cli"), ("step_comfyui", "ComfyUI"), ("step_model_folders", "model folders"),
        ("step_model_paths", "model paths"), ("step_nodes", "custom nodes"),
        ("step_accel", "Triton + SageAttention"), ("step_llamacpp", "llama.cpp"), ("step_llm_models", "LLM models"),
    ):
        monkeypatch.setattr(ai, fn, ok(name))
    monkeypatch.setattr(ai, "run_preflight", lambda **k: SimpleNamespace(ok=True, gpu={}, llama=None))
    monkeypatch.setattr(ai, "print_preflight", lambda pf: None)
    monkeypatch.setattr(ai, "gpu_install_flag", lambda gpu, override: "--cpu")
    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")
    monkeypatch.setattr(
        ai, "verify_rows", lambda: [{"ok": True, "name": "ffmpeg", "detail": "ok"}, setup_mod.check_ltx23_weights()]
    )
    args = Namespace(yes=True, preflight_only=False, dry_run=False, gpu="auto", skip_weights=True, skip_sage=False, skip_llm=False)
    rc = ai.cmd_automatic_install(args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "DEFERRED" in out and "Still needs you" not in out
    assert "INFO  ltx23-weights" in out
    assert "Install OK" in out
    assert "python -m master_agent models register <file-or-folder>" in out


# --- models register -----------------------------------------------------------


def test_register_hardlinks_into_models_dir_subfolders(env):
    src = env.tmp / "Downloads"
    _make_pack(src, env.files)
    _make(src / "unrelated-flux.safetensors", 5)
    deferred.mark_deferred(["x"])
    report = register.register_weights([src])
    assert report.ok and report.complete and report.missing == []
    assert report.marker_cleared and not deferred.weights_deferred()
    assert {i.action for i in report.items} == {register.LINKED}
    assert not any("unrelated" in i.source for i in report.items)
    for w in env.files.values():
        dest = env.models / w.dest_folder / w.filename
        assert dest.is_file() and dest.samefile(src / w.filename)


def test_register_rejects_wrong_size_unless_forced(env):
    w = env.files["ltx23_gguf"]
    bad = _make(env.tmp / "dl" / w.filename, w.size_bytes - 1)
    report = register.register_weights([bad])
    assert not report.ok
    item = report.items[0]
    assert item.action == register.REJECTED and "expected" in item.detail and "--force" in item.detail
    assert not (env.models / "diffusion_models" / w.filename).exists()
    forced = register.register_weights([bad], force=True)
    assert forced.ok and (env.models / "diffusion_models" / w.filename).is_file()


def test_register_rejects_unknown_name_and_missing_path(env):
    other = _make(env.tmp / "dl" / "some-other-model.safetensors", 10)
    report = register.register_weights([other, env.tmp / "nope"])
    assert [i.action for i in report.items] == [register.REJECTED, register.REJECTED]
    assert "no such file" in report.items[0].detail
    assert "Expected one of" in report.items[1].detail


def test_register_accepts_scan_alternate_without_exact_size(env):
    alt = _make(env.tmp / "dl" / "sulphur_dev-Q3_K_S.gguf", 7)
    report = register.register_weights([alt])
    assert report.ok and report.items[0].key == "ltx23_gguf"
    assert (env.models / "diffusion_models" / "sulphur_dev-Q3_K_S.gguf").is_file()


def test_register_leaves_files_already_in_managed_comfy_models(env):
    w = env.files["ltx23_tiny_vae"]
    inside = _make(env.comfy / "models" / "vae" / w.filename, w.size_bytes)
    report = register.register_weights([inside])
    assert report.items[0].action == register.IN_PLACE
    assert not (env.models / "vae" / w.filename).exists()


def test_register_dry_run_and_copy_mode(env):
    w = env.files["ltx23_video_vae"]
    src = _make(env.tmp / "dl" / w.filename, w.size_bytes)
    plan = register.register_weights([src], dry_run=True)
    assert plan.items[0].action == register.WOULD
    assert not (env.models / "vae" / w.filename).exists()
    done = register.register_weights([src], mode="copy")
    dest = env.models / "vae" / w.filename
    assert done.items[0].action == register.COPIED and dest.is_file() and not dest.samefile(src)
    again = register.register_weights([src], mode="copy")
    assert again.items[0].action == register.IN_PLACE


def test_register_refuses_models_dir_inside_managed_comfy(env, monkeypatch):
    """The write guard follows the resolved MANAGED_COMFY_ROOT, same as pack downloads."""
    monkeypatch.setattr(config, "MODELS_DIR", env.comfy / "models")
    w = env.files["ltx23_tiny_vae"]
    src = _make(env.tmp / "dl" / w.filename, w.size_bytes)
    report = register.register_weights([src])
    assert report.items[0].action == register.REJECTED
    assert "Refusing to write" in report.items[0].detail


def test_cli_models_register_json(env, capsys):
    from master_agent.cli.parser import main

    w = env.files["ltx23_tiny_vae"]
    src = _make(env.tmp / "dl" / w.filename, w.size_bytes)
    rc = main(["models", "register", str(src), "--json"])
    out = capsys.readouterr().out
    data = json.loads(out[out.index("{\n"):])
    assert rc == 0 and data["ok"] is True
    assert data["items"][0]["action"] == register.LINKED
    assert data["complete"] is False and len(data["missing"]) == 5
