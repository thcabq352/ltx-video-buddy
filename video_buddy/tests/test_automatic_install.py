"""Automatic install: pre-flight, plan, LTX 2.3-only weights, Comfy steps.

Subprocess, network, disk and GPU are mocked. Only tmp paths are written.

Run: python -m pytest tests/test_automatic_install.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent import automatic_install as ai
from master_agent import llamacpp_install
from master_agent import setup as setup_mod
from master_agent.comfy import comfy_venv, tower
from master_agent.models import weights

GB = ai.GB
FAKE_TOKEN = "hf_SECRETtokenVALUE1234567890"


def _args(**kw) -> Namespace:
    base = dict(yes=False, preflight_only=False, dry_run=False, gpu="auto", skip_weights=False, skip_sage=False)
    base.update(kw)
    return Namespace(**base)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A clean, empty machine: no Comfy, no llama.cpp, no build tools, no weights, plenty of disk, NVIDIA 24 GB."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    models = tmp_path / "models"
    workspace = tmp_path / "ComfyUI"
    monkeypatch.setattr(tower, "STATE_DIR", state_dir)
    monkeypatch.setattr(tower, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(workspace))
    for var in ("COMFY_MODE", "COMFY_CLI", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(setup_mod, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(setup_mod, "ENV_EXAMPLE", tmp_path / ".env.example")
    from master_agent import config

    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(ai, "models_dir", lambda: models)
    llama_root = tmp_path / "llama.cpp"
    monkeypatch.setattr(config, "LLAMACPP_ROOT", llama_root)
    monkeypatch.setattr(config, "LLAMACPP_BIN", "")
    monkeypatch.setattr(weights, "model_search_roots", lambda: [models])

    calls: list[list[str]] = []
    which: dict[str, str | None] = {}

    def fake_run(cmd, *, timeout=600.0, env=None):
        calls.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(ai, "_run", fake_run)
    monkeypatch.setattr(ai, "_which", lambda name: which.get(name))
    monkeypatch.setattr(ai, "disk_free", lambda path: 500 * GB)
    monkeypatch.setattr(ai, "_volume_id", lambda path: 1)
    monkeypatch.setattr(ai, "_is_root", lambda: False)
    monkeypatch.setattr(ai, "hf_whoami", lambda token: pytest.fail("token check must not run without a token"))
    monkeypatch.setattr(ai, "hf_repo_access", lambda repo, token: (True, "access ok"))
    gpu = {"vendor": "nvidia", "name": "RTX 4090", "vram_gb": 24.0, "sentence": "Routing: NVFP4 tier."}
    monkeypatch.setattr(ai, "detect_gpu", lambda: dict(gpu))
    monkeypatch.setattr(tower, "resolve_comfy_cli", lambda: ["comfy"])
    monkeypatch.setattr(
        ai,
        "plan_llamacpp",
        lambda gpu: llamacpp_install.plan_llamacpp(
            gpu_vendor=str(gpu.get("vendor") or ""),
            root=llama_root,
            ref="b11389",
            existing=ai.existing_llama_server(),
            which=lambda name: which.get(name),
            system="Linux",
            machine="x86_64",
        ),
    )
    monkeypatch.setattr(
        weights, "download_files", lambda *a, **k: pytest.fail("downloads must be mocked per test")
    )
    monkeypatch.setattr(
        llamacpp_install, "_download", lambda url, dest: pytest.fail("llama.cpp downloads must be mocked per test")
    )
    return SimpleNamespace(
        tmp=tmp_path, models=models, workspace=workspace, state=state_dir, calls=calls, which=which, gpu=gpu,
        llama_root=llama_root,
    )


# --- LTX 2.3 is the only video model ------------------------------------------


def test_ltx23_bundle_is_ltx23_files_only():
    files = weights.files_for_bundle(ai.LTX23_BUNDLE)
    keys = {w.key for w in files}
    assert keys == {
        "ltx23_checkpoint",
        "ltx23_gguf",
        "ltx23_text_encoder",
        "ltx23_distilled_lora",
        "ltx23_video_vae",
        "ltx23_tiny_vae",
    }
    for w in files:
        name = w.filename.lower()
        assert w.repo_id in {
            weights.HF_LTX23, weights.HF_LTX23_GGUF, weights.HF_LTX23_KIJAI, weights.HF_LTX2_COMFY
        }, w.repo_id
        for other in ("ltx-2.5", "ltx25", "h3", "wan", "flux", "qwen", "krea", "vace", "heartmula"):
            assert other not in name, (other, w.filename)
        assert w.gated is False
        assert w.mandatory is True
    assert sum(w.size_bytes for w in files) == pytest.approx(76.5 * GB, rel=0.01)


def test_plan_lists_only_ltx23_and_apply_downloads_only_ltx23(env, monkeypatch):
    got: list[tuple[str, bool]] = []

    def fake_download(bundle, *, yes=False, **kw):
        got.append((bundle, yes))
        for w in weights.files_for_bundle(bundle):
            dest = env.models / w.dest_folder / w.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
        return weights.scan_bundle(bundle), []

    monkeypatch.setattr(weights, "download_missing_bundle", fake_download)
    plan = ai.step_weights(False, skip=False)
    assert plan.status == ai.PLANNED
    assert len(plan.lines) == 6
    joined = "\n".join(plan.lines).lower()
    for other in ("ltx-2.5", "h3", "wan", "flux"):
        assert other not in joined
    assert got == []

    done = ai.step_weights(True, skip=False)
    assert got == [("ltx23_core", True)]
    assert done.status == ai.OK


def test_cmd_download_models_ltx23_flag_and_default_unchanged(monkeypatch):
    from master_agent.cli import models as cli_models

    class Picked(Exception):
        pass

    def fake_scan(bundle, **kw):
        raise Picked(bundle)

    monkeypatch.setattr(weights, "scan_bundle", fake_scan)
    monkeypatch.setattr(cli_models, "print_inventory_preamble", lambda *a, **k: None, raising=False)
    base = dict(bundle=None, h3=False, ltx25=True, heartmula=False, json=False, yes=False)

    def pick(**extra) -> str:
        with pytest.raises(Picked) as exc:
            cli_models.cmd_download_models(Namespace(**{**base, **extra}))
        return str(exc.value)

    assert pick(ltx23=True) == "ltx23_core"
    assert pick() == "ltx25_all"
    assert pick(h3=True) == "h3_all"


# --- pre-flight ---------------------------------------------------------------


def test_preflight_order_and_happy_path(env):
    pf = ai.run_preflight()
    assert [c.name for c in pf.checks] == [
        "Disk space",
        "Hugging Face access",
        "GPU / VRAM",
        "llama.cpp",
        "ComfyUI folder",
    ]
    assert pf.ok
    by = {c.name: c for c in pf.checks}
    assert by["Disk space"].status == ai.PASS
    assert by["GPU / VRAM"].status == ai.INFO and not by["GPU / VRAM"].blocking
    llama = by["llama.cpp"]
    assert llama.status == ai.INFO and not llama.blocking
    text = " ".join(llama.lines)
    assert "official prebuilt release b11389" in text and "the CUDA toolkit (nvcc)" in text
    assert "qwen3-vl-heretic" in text and "no public GGUF source is defined" in text
    assert pf.llama.method == "prebuilt"


def test_preflight_low_disk_fails_closed_with_free_and_needed(env, monkeypatch):
    monkeypatch.setattr(ai, "disk_free", lambda path: 40 * GB)
    pf = ai.run_preflight()
    bad = pf.first_failure
    assert bad is not None and bad.name == "Disk space"
    text = "\n".join(bad.lines)
    assert "40.0 GB free" in text
    # 76.5 weights + 15 Comfy + 2.5 llama.cpp CUDA prebuilt + 5 headroom
    assert "99.0 GB needed" in text
    assert "llama.cpp 2.5 GB" in text
    assert "MODELS_DIR=" in text and "Free up at least" in text


def test_preflight_disk_grouped_per_volume(env, monkeypatch):
    llama_dir = env.llama_root
    for d in (env.models, env.workspace, llama_dir):
        d.mkdir()
    volumes = {env.models: 1, env.workspace: 2, llama_dir: 3}

    def vol(path):
        return volumes[Path(path)]

    monkeypatch.setattr(ai, "_volume_id", vol)
    free = {1: 100 * GB, 2: 10 * GB, 3: 100 * GB}
    monkeypatch.setattr(ai, "disk_free", lambda path: free[vol(path)])
    check = ai.check_disk(
        weights_bytes=70 * GB,
        comfy_bytes=15 * GB,
        llama_bytes=2 * GB,
        models=env.models,
        workspace=env.workspace,
        llama_dir=llama_dir,
    )
    assert check.status == ai.FAIL
    text = "\n".join(check.lines)
    assert text.count("space on the drive holding") == 3
    assert "NOT enough" in text and "MANAGED_COMFY_ROOT=" in text


def test_preflight_missing_token_not_needed_passes(env):
    check = ai.check_hf(weights.scan_bundle("ltx23_core").missing_mandatory, env_file=env.tmp / ".env")
    assert check.status == ai.PASS
    assert "public" in " ".join(check.lines)


def test_preflight_invalid_token_fails_and_never_prints_it(env, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(ai, "hf_whoami", lambda token: (False, "Hugging Face rejected the token (401 Unauthorized)"))
    rc = ai.cmd_automatic_install(_args(preflight_only=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "STOPPED at 'Hugging Face access'" in out
    assert "huggingface.co/settings/tokens" in out
    assert FAKE_TOKEN not in out


def test_preflight_valid_token_passes_without_printing(env, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(ai, "hf_whoami", lambda token: (True, "scott"))
    rc = ai.cmd_automatic_install(_args(preflight_only=True))
    out = capsys.readouterr().out
    assert rc == 0 and "valid" in out and FAKE_TOKEN not in out


def test_preflight_gated_file_without_token_fails(env):
    gated = [SimpleNamespace(gated=True, repo_id="Lightricks/gated")]
    check = ai.check_hf(gated, env_file=env.tmp / ".env")
    assert check.failed
    assert "HF_TOKEN=" in "\n".join(check.lines)


def test_preflight_gated_license_not_accepted_fails(env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(ai, "hf_whoami", lambda token: (True, "scott"))
    monkeypatch.setattr(ai, "hf_repo_access", lambda repo, token: (False, "access denied (license not accepted on the repo page)"))
    check = ai.check_hf([SimpleNamespace(gated=True, repo_id="Lightricks/gated")], env_file=env.tmp / ".env")
    assert check.failed
    assert "https://huggingface.co/Lightricks/gated" in "\n".join(check.lines)


def test_preflight_no_gpu_is_informational_cpu_slow(env, monkeypatch):
    monkeypatch.setattr(ai, "detect_gpu", lambda: {"vendor": "unknown", "vram_gb": None, "sentence": ""})
    pf = ai.run_preflight()
    gpu = next(c for c in pf.checks if c.name == "GPU / VRAM")
    assert gpu.status == ai.INFO and pf.ok
    assert "very slow" in " ".join(gpu.lines)
    assert ai.gpu_install_flag(pf.gpu) == "--cpu"


def test_preflight_small_vram_routes_gguf_q4(env):
    check = ai.check_gpu({"vendor": "nvidia", "name": "RTX 4060", "vram_gb": 8.0, "sentence": "Routing: GGUF."})
    text = " ".join(check.lines)
    assert "GGUF Q4" in text and "Routing: GGUF." in text


def test_preflight_external_mode_refuses(env, monkeypatch, capsys):
    monkeypatch.setenv("COMFY_MODE", "external")
    rc = ai.cmd_automatic_install(_args(yes=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "STOPPED at 'ComfyUI folder'" in out and "COMFY_MODE" in out
    assert env.calls == []


def test_preflight_foreign_workspace_refuses(env):
    env.workspace.mkdir()
    (env.workspace / "notes.txt").write_text("mine")
    pf = ai.run_preflight()
    assert pf.first_failure is not None and pf.first_failure.name == "ComfyUI folder"


def test_preflight_unwritable_workspace_refuses(env, monkeypatch):
    monkeypatch.setattr(ai.os, "access", lambda path, mode: False)
    check = ai.check_workspace(env.workspace, state="absent", mode="managed")
    assert check.failed and "cannot write" in " ".join(check.lines)


def test_preflight_prints_first_and_changes_nothing(env, capsys):
    rc = ai.cmd_automatic_install(_args(preflight_only=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("AUTOMATIC INSTALL — pre-flight check")
    assert "Pre-flight passed." in out
    assert env.calls == []
    assert not env.workspace.exists() and not env.models.exists()
    assert not (env.tmp / ".env").exists()


def test_failed_preflight_stops_before_any_step(env, monkeypatch, capsys):
    monkeypatch.setattr(ai, "disk_free", lambda path: 1 * GB)
    rc = ai.cmd_automatic_install(_args(yes=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "STOPPED at 'Disk space'" in out and "Nothing was downloaded or installed." in out
    assert "AUTOMATIC INSTALL — installing" not in out
    assert env.calls == []


def test_dry_run_prints_plan_and_changes_nothing(env, capsys):
    rc = ai.cmd_automatic_install(_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert out.index("pre-flight check") < out.index("plan (dry run")
    assert "Video model: LTX 2.3 only." in out
    assert "--nvidia" in out and "automatic-install --yes" in out
    assert "PLANNED          llama.cpp (optional)" in out
    assert "ollama" not in out.lower()
    assert env.calls == []
    assert not env.workspace.exists() and not env.models.exists() and not env.llama_root.exists()


def test_parser_install_all_alias_and_help(monkeypatch, capsys):
    from master_agent.cli import parser as cli_parser

    seen: list[Namespace] = []
    monkeypatch.setattr(cli_parser, "cmd_automatic_install", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(cli_parser, "ensure_dirs", lambda: None, raising=False)
    assert cli_parser.main(["install-all", "--preflight-only", "--gpu", "cpu"]) == 0
    assert seen[0].command == "automatic-install" and seen[0].preflight_only and seen[0].gpu == "cpu"
    with pytest.raises(SystemExit):
        cli_parser.main(["--help"])
    out = capsys.readouterr().out
    assert "automatic-install" in out and "install-all" not in out
    with pytest.raises(SystemExit):
        cli_parser.main(["automatic-install", "--help"])
    out = capsys.readouterr().out
    assert "--skip-llm" in out and "llama.cpp" in out
    assert "ollama" not in out.lower()


# --- ComfyUI / nodes / ffmpeg steps -------------------------------------------


def _fake_comfy(ws: Path, *, venv: bool = True) -> None:
    (ws / "comfy").mkdir(parents=True, exist_ok=True)
    (ws / "main.py").write_text("")
    if venv:
        py = ws / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("")


def test_comfy_install_argv_uses_workspace_gpu_flag_and_strips_outer_venv(env, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/buddy/.venv")
    seen: dict = {}

    def fake_run(cmd, *, timeout=600.0, env=None):
        seen["cmd"], seen["env"] = cmd, env
        return 0, ""

    monkeypatch.setattr(ai, "_run", fake_run)
    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")
    monkeypatch.setattr(comfy_venv, "probe_comfy_env", lambda ws, **k: comfy_venv.ComfyEnv(python_path="/x/python"))
    res = ai.step_comfyui(True, workspace=env.workspace, state="absent", gpu_flag="--nvidia")
    assert res.status == ai.OK
    assert seen["cmd"] == [
        "comfy", f"--workspace={env.workspace}", "--where", "local", "--skip-prompt", "install", "--nvidia",
    ]
    assert "VIRTUAL_ENV" not in seen["env"]


def test_comfy_install_partial_restores_and_installed_skips(env):
    plan = ai.step_comfyui(False, workspace=env.workspace, state="partial", gpu_flag="--cpu")
    assert plan.status == ai.PLANNED and plan.detail.startswith("repair:")
    assert plan.detail.endswith("install --cpu --restore")
    skip = ai.step_comfyui(True, workspace=env.workspace, state="installed", gpu_flag="--cpu")
    assert skip.status == ai.SKIPPED
    assert env.calls == []


def test_comfy_install_state(env, monkeypatch):
    assert ai.comfy_install_state(env.workspace) == "absent"
    _fake_comfy(env.workspace, venv=False)
    assert ai.comfy_install_state(env.workspace) == "partial"
    _fake_comfy(env.workspace)
    monkeypatch.setattr(comfy_venv, "module_importable", lambda py, mod, **k: mod == "torch")
    assert ai.comfy_install_state(env.workspace) == "installed"


@pytest.mark.parametrize(
    "gpu,platform_name,machine,want",
    [
        ({"vendor": "nvidia"}, "linux", "x86_64", "--nvidia"),
        ({"vendor": "amd"}, "linux", "x86_64", "--amd"),
        ({"vendor": "unknown"}, "win32", "AMD64", "--cpu"),
        ({"vendor": "apple"}, "darwin", "arm64", "--m-series"),
    ],
)
def test_gpu_install_flag(monkeypatch, gpu, platform_name, machine, want):
    monkeypatch.setattr(ai.sys, "platform", platform_name)
    monkeypatch.setattr(ai.platform, "machine", lambda: machine)
    assert ai.gpu_install_flag(gpu) == want
    assert ai.gpu_install_flag(gpu, "cpu") == "--cpu"


def test_nodes_install_only_ltx23_packs(env, monkeypatch):
    _fake_comfy(env.workspace)
    (env.workspace / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True)
    plan = ai.step_nodes(False, workspace=env.workspace)
    assert plan.detail.endswith("node install ComfyUI-LTXVideo")
    assert set(ai.REQUIRED_NODES) == {"ComfyUI-LTXVideo", "ComfyUI-GGUF"}

    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")

    def fake_run(cmd, *, timeout=600.0, env=None):
        (Path(cmd[1].split("=", 1)[1]) / "custom_nodes" / "ComfyUI-LTXVideo").mkdir()
        return 0, ""

    monkeypatch.setattr(ai, "_run", fake_run)
    assert ai.step_nodes(True, workspace=env.workspace).status == ai.OK
    assert ai.step_nodes(True, workspace=env.workspace).status == ai.SKIPPED


def test_ffmpeg_linux_never_sudo_silently(env, monkeypatch):
    monkeypatch.setattr(ai.sys, "platform", "linux")
    env.which["apt-get"] = "/usr/bin/apt-get"
    res = ai.step_ffmpeg(True)
    assert res.status == ai.NEEDS_YOU
    assert "sudo apt-get install -y ffmpeg" in res.detail
    assert env.calls == []


def test_model_paths_written_outside_comfy_tree(env):
    res = ai.step_model_paths(True)
    assert res.status == ai.OK
    yaml_path = env.state / "extra_model_paths.yaml"
    assert yaml_path.is_file()
    assert str(env.models) in yaml_path.read_text()
    assert not env.workspace.exists()


def test_apply_summary_reports_needs_you(env, monkeypatch, capsys):
    """Full --yes run with every step mocked: required gaps exit 1 with plain next steps."""
    monkeypatch.setattr(ai, "step_playwright", lambda apply: ai.StepResult("Playwright Chromium", ai.OK, "ready", required=False))
    monkeypatch.setattr(ai, "step_comfy_cli", lambda apply: ai.StepResult("comfy-cli", ai.SKIPPED, "pinned"))
    monkeypatch.setattr(ai, "step_comfyui", lambda apply, **k: ai.StepResult("ComfyUI", ai.OK, "installed"))
    monkeypatch.setattr(ai, "step_nodes", lambda apply, **k: ai.StepResult("custom nodes", ai.OK, "installed"))
    monkeypatch.setattr(ai, "step_accel", lambda apply, **k: ai.StepResult("Triton + SageAttention", ai.NONFATAL, "x", required=False))
    monkeypatch.setattr(ai, "step_weights", lambda apply, **k: ai.StepResult("LTX 2.3 weights", ai.OK, "ready"))
    monkeypatch.setattr(ai, "verify_rows", lambda: [{"ok": False, "name": "ffmpeg", "detail": "missing"}])

    def offline(url, dest):
        raise OSError("offline")

    monkeypatch.setattr(llamacpp_install, "_download", offline)
    rc = ai.cmd_automatic_install(_args(yes=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "AUTOMATIC INSTALL — summary" in out
    assert "Not ready for generate yet. Still needs you:" in out
    needs = out.split("Still needs you:")[1]
    assert "ffmpeg" in needs
    assert "Triton + SageAttention" not in needs and "llama.cpp" not in needs and "LLM models" not in needs
    assert "No local LLM runtime yet" in out


def test_install_py_forwards_automatic_install_flags(monkeypatch, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("vb_install", Path(__file__).resolve().parents[1] / "install.py")
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
    monkeypatch.setattr(mod.sys, "argv", ["install.py", "--automatic-install", "--yes", "--gpu", "cpu", "--skip-sage"])
    assert mod.main() == 0
    assert calls[-1][1:] == ["-m", "master_agent", "automatic-install", "--yes", "--skip-sage", "--gpu", "cpu"]
    assert not any("setup" in c for c in calls)


# --- Ollama is gone; llama.cpp + LLM models ---------------------------------------


def test_full_yes_run_never_invokes_ollama_and_skipped_models_do_not_block(env, monkeypatch, capsys):
    """Every subprocess / download is recorded. Nothing mentions Ollama; the run still ends ready."""
    import subprocess as sp
    import tarfile

    recorded: list[str] = []

    def fake_sp_run(cmd, *a, **k):
        recorded.append(" ".join(map(str, cmd)))
        return sp.CompletedProcess(cmd, 0, "version: 0.5.0-dev (build 11389)", "")

    def fake_download(url, dest):
        recorded.append(url)
        stage = env.tmp / "stage" / "llama-b11389"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "llama-server").write_text("#!/bin/sh\n")
        with tarfile.open(dest, "w:gz") as tf:
            tf.add(stage, arcname="llama-b11389")

    def fake_run(cmd, *, timeout=600.0, env=None):
        recorded.append(" ".join(cmd))
        return 0, ""

    monkeypatch.setattr(sp, "run", fake_sp_run)
    monkeypatch.setattr(llamacpp_install, "_download", fake_download)
    monkeypatch.setattr(ai, "_run", fake_run)
    env.which["ffmpeg"] = "/usr/bin/ffmpeg"
    monkeypatch.setattr(ai, "step_playwright", lambda apply: ai.StepResult("Playwright Chromium", ai.OK, "ready", required=False))
    monkeypatch.setattr(ai, "step_comfy_cli", lambda apply: ai.StepResult("comfy-cli", ai.SKIPPED, "pinned"))
    monkeypatch.setattr(ai, "step_comfyui", lambda apply, **k: ai.StepResult("ComfyUI", ai.OK, "installed"))
    monkeypatch.setattr(ai, "step_nodes", lambda apply, **k: ai.StepResult("custom nodes", ai.OK, "installed"))
    monkeypatch.setattr(ai, "step_accel", lambda apply, **k: ai.StepResult("Triton + SageAttention", ai.SKIPPED, "cpu", required=False))
    monkeypatch.setattr(ai, "step_weights", lambda apply, **k: ai.StepResult("LTX 2.3 weights", ai.OK, "ready"))
    monkeypatch.setattr(ai, "verify_rows", lambda: [{"ok": True, "name": "ffmpeg", "detail": "ok"}])

    rc = ai.cmd_automatic_install(_args(yes=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Ready for generate (LTX 2.3)" in out
    assert "Skipped qwen3-vl-heretic: no public GGUF source is defined; continuing." in out
    assert "Skipped nomic-embed-text: no public GGUF source is defined; continuing." in out
    assert "Skipped (no public GGUF source is defined): qwen3-vl-heretic, nomic-embed-text" in out
    assert "SKIPPED          LLM models (optional)" in out
    assert "Still needs you" not in out
    assert "OK               llama.cpp (optional): official prebuilt b11389 (cuda)" in out
    assert any("llama-b11389-bin-ubuntu-cuda-12.8-x64.tar.gz" in r for r in recorded)
    assert all("ollama" not in r.lower() for r in recorded), recorded
    assert "ollama" not in out.lower()
    assert (env.llama_root / "installed.json").is_file()


def test_llm_models_skip_and_continue_without_source(env, capsys):
    res = ai.step_llm_models(True, skip=False)
    out = capsys.readouterr().out
    assert res.status == ai.SKIPPED and not res.required and not res.failed_required
    assert out.splitlines() == [
        "Skipped qwen3-vl-heretic: no public GGUF source is defined; continuing.",
        "Skipped nomic-embed-text: no public GGUF source is defined; continuing.",
    ]
    assert "skipped (no public GGUF source): qwen3-vl-heretic, nomic-embed-text" == res.detail


def test_llm_models_already_in_models_dir_are_found(env, capsys):
    env.models.mkdir()
    (env.models / "qwen3-vl-heretic-Q4_K_M.gguf").write_bytes(b"x")
    (env.models / "nomic-embed-text").mkdir()
    (env.models / "nomic-embed-text" / "nomic-embed-text-v1.5.f16.gguf").write_bytes(b"x")
    res = ai.step_llm_models(True, skip=False)
    assert capsys.readouterr().out == ""
    assert res.detail == "present: qwen3-vl-heretic, nomic-embed-text"
    assert ai.check_llamacpp(ai.plan_llamacpp(env.gpu), skip=False).lines[-1].startswith("This is information only")


def test_skip_llm_leaves_llama_out_of_disk_and_steps(env, monkeypatch, capsys):
    monkeypatch.setattr(ai, "disk_free", lambda path: 97 * GB)
    pf = ai.run_preflight(skip_llm=True)
    assert pf.ok and pf.llama is None
    assert "llama.cpp" not in "\n".join(next(c for c in pf.checks if c.name == "Disk space").lines)
    assert "--skip-llm" in next(c for c in pf.checks if c.name == "llama.cpp").lines[0]
    assert ai.step_llamacpp(True, plan=None, skip=True).status == ai.SKIPPED
    assert ai.step_llm_models(True, skip=True).status == ai.SKIPPED
    assert capsys.readouterr().out == ""


def test_llamacpp_row_found_on_path(env):
    env.which["llama-server"] = "/usr/local/bin/llama-server"
    pf = ai.run_preflight()
    row = next(c for c in pf.checks if c.name == "llama.cpp")
    assert row.lines[0] == "llama.cpp found: /usr/local/bin/llama-server (LLAMACPP_BIN or PATH)."
    assert pf.llama.disk_bytes == 0


def test_llamacpp_row_will_build_with_toolchain(env, monkeypatch):
    env.gpu["vendor"] = "unknown"
    env.which.update({"git": "git", "cmake": "cmake", "c++": "c++"})
    pf = ai.run_preflight()
    row = next(c for c in pf.checks if c.name == "llama.cpp")
    assert "will be built from source" in row.lines[0] and "tag b11389, CPU backend" in row.lines[0]


def test_llamacpp_unavailable_needs_you_prints_command_never_runs_sudo(env, monkeypatch):
    monkeypatch.setattr(
        ai,
        "plan_llamacpp",
        lambda gpu: llamacpp_install.plan_llamacpp(
            gpu_vendor="unknown", root=env.llama_root, ref="b11389", existing=None,
            which=lambda n: None, system="Linux", machine="riscv64",
        ),
    )
    pf = ai.run_preflight()
    row = next(c for c in pf.checks if c.name == "llama.cpp")
    assert row.status == ai.INFO and pf.ok
    assert "unavailable" in row.lines[0]
    assert any("sudo apt-get install -y git cmake build-essential" in line for line in row.lines)
    res = ai.step_llamacpp(True, plan=pf.llama, skip=False)
    assert res.status == ai.NEEDS_YOU and not res.required and not res.failed_required
    assert "sudo apt-get install -y git cmake build-essential" in res.detail
    assert env.calls == []


def test_existing_llama_server_honors_llamacpp_bin(env, monkeypatch, tmp_path):
    from master_agent import config

    binary = tmp_path / "my-llama-server"
    binary.write_text("")
    monkeypatch.setattr(config, "LLAMACPP_BIN", str(binary))
    assert ai.existing_llama_server() == str(binary)
