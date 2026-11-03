"""New-user install: model folder layout, the light-install banner, install-line classification.

No GPU, no network, no downloads. Only tmp paths are written.

Run: python -m pytest tests/test_new_user_install.py -q
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent import automatic_install as ai
from master_agent import config
from master_agent import install_report as ir
from master_agent import setup as setup_mod
from master_agent.comfy import model_paths
from master_agent.models import layout, weights

ROOT = Path(__file__).resolve().parent.parent


# --- model folders ---------------------------------------------------------------


def test_ensure_model_folders_creates_every_subfolder(tmp_path):
    root = tmp_path / "models"
    actions = layout.ensure_model_folders(root)
    assert [a.path for a in actions] == [root / n for n in layout.SUBFOLDER_NAMES]
    assert all(a.action == layout.CREATED for a in actions)
    assert all((root / n).is_dir() for n in layout.SUBFOLDER_NAMES)
    assert layout.missing_model_folders(root) == []


def test_ensure_model_folders_is_idempotent_and_keeps_files(tmp_path):
    root = tmp_path / "models"
    layout.ensure_model_folders(root)
    keep = root / "loras" / "mine.safetensors"
    keep.write_bytes(b"x")
    again = layout.ensure_model_folders(root)
    assert all(a.action == layout.EXISTS for a in again)
    assert keep.read_bytes() == b"x"


def test_ensure_model_folders_fills_gaps_only(tmp_path):
    root = tmp_path / "models"
    (root / "vae").mkdir(parents=True)
    actions = {a.path.name: a.action for a in layout.ensure_model_folders(root)}
    assert actions["vae"] == layout.EXISTS
    assert actions["loras"] == layout.CREATED


def test_ensure_model_folders_dry_run_creates_nothing(tmp_path):
    root = tmp_path / "models"
    actions = layout.ensure_model_folders(root, dry_run=True)
    assert all(a.action == layout.WOULD_CREATE for a in actions)
    assert not root.exists()


def test_layout_covers_every_download_folder_and_is_mapped_for_comfy():
    names = set(layout.SUBFOLDER_NAMES)
    assert {w.dest_folder for w in weights.WEIGHT_FILES.values()} <= names
    mapped = {folder for _key, folders in model_paths._FOLDERS for folder in folders}
    assert names <= mapped
    for required in ("checkpoints", "loras", "latent_upscale_models", "upscale_models", "text_encoders",
                     "vae", "diffusion_models", "clip_vision", "unet"):
        assert required in names


def test_model_folder_defaults_to_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "m")
    assert layout.model_folder("loras") == tmp_path / "m" / "loras"


# --- automatic install step ------------------------------------------------------------


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    monkeypatch.setattr(ai, "models_dir", lambda: root)
    return root


def test_step_model_folders_plan_prints_mkdirs_and_creates_nothing(models):
    res = ai.step_model_folders(False)
    assert res.status == ai.PLANNED
    assert res.lines == [f"mkdir {models / n}" for n in layout.SUBFOLDER_NAMES]
    assert not models.exists()


def test_step_model_folders_apply_then_skip(models):
    res = ai.step_model_folders(True)
    assert res.status == ai.OK
    assert all((models / n).is_dir() for n in layout.SUBFOLDER_NAMES)
    assert ai.step_model_folders(True).status == ai.SKIPPED


def test_automatic_install_makes_folders_before_weights(models, monkeypatch, capsys):
    seen: list[str] = []
    pf = ai.Preflight(checks=[], gpu={"vendor": "cpu"})
    monkeypatch.setattr(ai, "run_preflight", lambda **k: pf)
    monkeypatch.setattr(ai, "print_preflight", lambda p: None)
    monkeypatch.setattr(ai, "managed_workspace", lambda: models.parent / "ComfyUI")

    def stub(name):
        def make(*_a, **_k):
            seen.append(name)
            return ai.StepResult(name, ai.PLANNED, "stub")
        return make

    for fn, name in (
        ("step_env_file", ".env"), ("step_playwright", "Playwright Chromium"), ("step_ffmpeg", "ffmpeg"),
        ("step_comfy_cli", "comfy-cli"), ("step_windows_defender_exclusion", "Windows Defender exclusion"),
        ("step_comfyui", "ComfyUI"), ("step_model_paths", "model paths"),
        ("step_nodes", "custom nodes"), ("step_accel", "Triton + SageAttention"),
        ("step_weights", "LTX 2.3 weights"), ("step_vram_profile", "VRAM profile"),
        ("step_klein_4b", "Klein 4B stills"), ("step_llamacpp", "llama.cpp"), ("step_llm_models", "LLM models"),
    ):
        monkeypatch.setattr(ai, fn, stub(name))
    real = ai.step_model_folders
    monkeypatch.setattr(ai, "step_model_folders", lambda apply: seen.append("model folders") or real(apply))
    args = Namespace(yes=False, preflight_only=False, dry_run=True, gpu="auto",
                     skip_weights=False, skip_sage=False, skip_llm=False)
    assert ai.cmd_automatic_install(args) == 0
    assert seen.index("model folders") < seen.index("LTX 2.3 weights")
    out = capsys.readouterr().out
    assert f"mkdir {models / 'checkpoints'}" in out
    assert not models.exists()


# --- doctor / setup --fix ----------------------------------------------------------------


def test_doctor_row_names_missing_folders_and_fix_creates_them(models, capsys):
    row = setup_mod.check_model_folders()
    assert row["ok"] is False
    assert str(models) in row["detail"] and "latent_upscale_models" in row["detail"]
    assert "setup --fix" in row["fix"]
    assert not models.exists()
    setup_mod.install_model_folders()
    assert setup_mod.check_model_folders()["ok"] is True
    assert "created 12 model folder(s)" in capsys.readouterr().out


def test_doctor_scan_does_not_create_folders(models):
    setup_mod.print_report([setup_mod.check_model_folders()])
    assert not models.exists()


def test_need_lines_name_the_exact_models_dir_folder(models):
    layout.ensure_model_folders(models)
    up = weights.ltx23_latent_upscaler_placement([models])
    assert up["ok"] is False
    assert str(models / "latent_upscale_models") in up["detail"]
    assert str(models / "latent_upscale_models") in up["fix"]
    lora = weights.ltx23_inoutpaint_lora_placement([models])
    assert str(models / "loras") in lora["fix"]
    assert (models / "loras").is_dir()
    assert "ComfyUI/models" not in up["fix"] + lora["fix"]


# --- light-install banner ------------------------------------------------------------------


@pytest.mark.parametrize("where", ["start", "end"])
def test_light_banner_is_loud_ascii_and_names_the_full_command(where):
    text = ir.light_install_banner(where=where)
    text.encode("ascii")
    assert "LIGHT INSTALL" in text
    assert "ComfyUI" in text and "LTX 2.3" in text
    assert "python install.py --automatic-install --yes" in text
    assert "install.bat --automatic-install --yes" in text
    assert text.splitlines()[0].startswith("#" * 40)


def _installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("vb_install_banner", ROOT / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    venv = tmp_path / ".venv"
    monkeypatch.setattr(mod, "VENV", venv)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    py = mod._venv_python()
    py.parent.mkdir(parents=True)
    py.write_text("")
    (venv / "pyvenv.cfg").write_text("")
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd, *a, **k: calls.append([str(c) for c in cmd]) or 0)
    return SimpleNamespace(mod=mod, calls=calls, venv=venv)


def test_plain_install_prints_banner_at_start_and_end(tmp_path, monkeypatch, capsys):
    inst = _installer(tmp_path, monkeypatch)
    monkeypatch.setattr(inst.mod.sys, "argv", ["install.py"])
    assert inst.mod.main() == 0
    out = capsys.readouterr().out
    assert out.count("THIS IS THE LIGHT INSTALL") == 2
    assert "(starting)" in out and "(finished)" in out
    tail = out.rstrip().splitlines()[-12:]
    assert any("did NOT install ComfyUI" in line for line in tail)
    assert any("python install.py --automatic-install --yes" in line for line in tail)
    assert inst.calls[-1][-3:] == ["master_agent", "setup", "--fix"]


def test_automatic_install_path_has_no_light_banner(tmp_path, monkeypatch, capsys):
    inst = _installer(tmp_path, monkeypatch)
    monkeypatch.setattr(inst.mod.sys, "argv", ["install.py", "--automatic-install", "--dry-run"])
    assert inst.mod.main() == 0
    assert "LIGHT INSTALL" not in capsys.readouterr().out
    assert inst.calls[-1][-2:] == ["automatic-install", "--dry-run"]


def test_venv_without_pip_fails_loud_with_the_fix(tmp_path, monkeypatch, capsys):
    inst = _installer(tmp_path, monkeypatch)
    monkeypatch.setattr(inst.mod.subprocess, "call", lambda cmd, *a, **k: 1)
    monkeypatch.setattr(inst.mod.sys, "argv", ["install.py"])
    with pytest.raises(SystemExit) as exc:
        inst.mod.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "has no pip" in out and "Delete" in out


def test_pip_self_upgrade_failure_is_nonfatal_when_pip_exists(tmp_path, monkeypatch, capsys):
    inst = _installer(tmp_path, monkeypatch)
    (inst.venv / "lib" / "python3.12" / "site-packages" / "pip").mkdir(parents=True)
    monkeypatch.setattr(
        inst.mod.subprocess, "call", lambda cmd, *a, **k: 1 if "--upgrade" in cmd else 0
    )
    monkeypatch.setattr(inst.mod.sys, "argv", ["install.py"])
    assert inst.mod.main() == 0
    out = capsys.readouterr().out
    assert "FAILED-NONFATAL  pip self-upgrade" in out
    assert "safe to ignore because" in out


# --- classification ---------------------------------------------------------------------------


def _row(name, ok, detail="d", fix=""):
    return {"name": name, "ok": ok, "detail": detail, "fix": fix}


def test_light_report_downgrades_what_the_light_install_never_provides(capsys):
    rows = [
        _row("python", True), _row("ffmpeg", True),
        _row("comfyui", False, "not reachable at http://127.0.0.1:8188"),
        _row("ltx23-weights", False), _row("h3-weights", False), _row("llm-models", False),
    ]
    assert setup_mod.print_report(rows, light=True) == 0
    out = capsys.readouterr().out
    assert "NEED" not in out
    assert "INFO  comfyui" in out
    assert "WARN  llm-models" in out
    assert out.count("safe to ignore because") == 3
    assert out.strip().splitlines()[-1] == "Install OK: 2 ok, 4 non-fatal note(s)."


def test_light_report_keeps_real_failures_loud_and_nonzero(capsys):
    rows = [_row("python", True), _row("ffmpeg", False, "not on PATH", "apt install ffmpeg"), _row("comfyui", False)]
    assert setup_mod.print_report(rows, light=True) == 1
    out = capsys.readouterr().out
    assert "NEED  ffmpeg" in out and "-> apt install ffmpeg" in out
    assert out.strip().splitlines()[-1].startswith("Install FAILED: 1 problem(s) to fix (ffmpeg)")


def test_plain_doctor_still_counts_comfy_and_weights():
    assert setup_mod.classify_row(_row("comfyui", False)) == ir.FAIL
    assert setup_mod.classify_row(_row("comfyui", False), light=True) == ir.INFO
    assert setup_mod.classify_row(_row("ltx23-weights", False), weight_optional=True) == ir.INFO
    assert setup_mod.classify_row(_row("venv", False), light=True) == ir.FAIL
    assert setup_mod.classify_row(_row("llm-models", False)) == ir.WARN


def test_plain_doctor_footer_does_not_send_comfy_to_setup_fix(capsys):
    assert setup_mod.print_report([_row("comfyui", False), _row("env", False)]) == 1
    out = capsys.readouterr().out
    assert "setup --fix handles: env" in out
    assert "setup --fix does not handle: comfyui" in out
    assert "python install.py --automatic-install --yes" in out


def test_comfy_row_tells_a_new_user_to_run_the_full_install(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(setup_mod, "_managed_comfy_state", lambda: ("absent", tmp_path / "ComfyUI"))
    row = setup_mod.check_comfy()
    assert row["ok"] is False
    assert "not installed yet" in row["fix"] and "--automatic-install --yes" in row["fix"]
    monkeypatch.setattr(setup_mod, "_managed_comfy_state", lambda: ("installed", tmp_path / "ComfyUI"))
    assert "comfy start" in setup_mod.check_comfy()["fix"]


def test_playwright_failure_is_nonfatal_with_reason(monkeypatch, capsys):
    monkeypatch.setattr(setup_mod, "_run", lambda cmd, **k: (1, "Host system is missing dependencies"))
    setup_mod.install_playwright()
    out = capsys.readouterr().out
    assert out.count("FAILED-NONFATAL") == 1 and "WARN" not in out
    assert "safe to ignore because only the web scraper uses Chromium" in out


def test_ffmpeg_failure_stays_loud(monkeypatch, capsys):
    monkeypatch.setattr(setup_mod, "_which", lambda name: None)
    setup_mod.install_ffmpeg()
    out = capsys.readouterr().out
    assert out.startswith("installing ffmpeg") and "FAIL  could not auto-install ffmpeg" in out


@pytest.mark.parametrize(
    "result,level",
    [
        (ai.StepResult("ComfyUI", ai.OK, "installed"), ir.OK),
        (ai.StepResult("comfy-cli", ai.SKIPPED, "pinned"), ir.OK),
        (ai.StepResult("ffmpeg", ai.NEEDS_YOU, "sudo"), ir.FAIL),
        (ai.StepResult("LTX 2.3 weights", ai.FAILED, "x"), ir.FAIL),
        (ai.StepResult("Triton + SageAttention", ai.NONFATAL, "x", required=False), ir.NONFATAL),
        (ai.StepResult("llama.cpp", ai.NEEDS_YOU, "cmake", required=False), ir.NONFATAL),
        (ai.StepResult("LLM models", ai.FAILED, "MISSING: qwen3-vl-heretic", required=True), ir.FAIL),
        (ai.StepResult("LLM models", ai.SKIPPED, "present: qwen3-vl-heretic", required=True), ir.OK),
    ],
)
def test_classify_step(result, level):
    assert ai.classify_step(result) == level


def test_optional_step_prints_reason(capsys):
    ai._print_steps([ai.StepResult("Triton + SageAttention", ai.NONFATAL, "no wheel", required=False)])
    assert "safe to ignore because ComfyUI runs without them, just slower" in capsys.readouterr().out


def test_summary_line():
    assert ir.summary_line(ok=9, notes=2, failed=[]) == "Install OK: 9 ok, 2 non-fatal note(s)."
    assert ir.summary_line(ok=8, notes=1, failed=["ffmpeg"]).startswith("Install FAILED: 1 problem(s) to fix (ffmpeg)")
