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
    """A clean, empty machine: no Comfy, no Ollama, no weights, plenty of disk, NVIDIA 24 GB."""
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
    monkeypatch.setattr(ai, "ollama_models_dir", lambda: tmp_path / "ollama")
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
    monkeypatch.setattr(ai, "ollama_responds", lambda url=None: False)
    monkeypatch.setattr(ai, "hf_whoami", lambda token: pytest.fail("token check must not run without a token"))
    monkeypatch.setattr(ai, "hf_repo_access", lambda repo, token: (True, "access ok"))
    gpu = {"vendor": "nvidia", "name": "RTX 4090", "vram_gb": 24.0, "sentence": "Routing: NVFP4 tier."}
    monkeypatch.setattr(ai, "detect_gpu", lambda: dict(gpu))
    monkeypatch.setattr(tower, "resolve_comfy_cli", lambda: ["comfy"])
    monkeypatch.setattr(
        weights, "download_files", lambda *a, **k: pytest.fail("downloads must be mocked per test")
    )
    return SimpleNamespace(
        tmp=tmp_path, models=models, workspace=workspace, state=state_dir, calls=calls, which=which, gpu=gpu
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
        "Ollama",
        "ComfyUI folder",
    ]
    assert pf.ok
    by = {c.name: c for c in pf.checks}
    assert by["Disk space"].status == ai.PASS
    assert by["GPU / VRAM"].status == ai.INFO and not by["GPU / VRAM"].blocking
    assert by["Ollama"].status == ai.INFO and not by["Ollama"].blocking
    assert "automatic install will install it" in " ".join(by["Ollama"].lines)


def test_preflight_low_disk_fails_closed_with_free_and_needed(env, monkeypatch):
    monkeypatch.setattr(ai, "disk_free", lambda path: 40 * GB)
    pf = ai.run_preflight()
    bad = pf.first_failure
    assert bad is not None and bad.name == "Disk space"
    text = "\n".join(bad.lines)
    assert "40.0 GB free" in text
    # 76.5 weights + 15 Comfy + 6.4 Ollama + 5 headroom
    assert "102.9 GB needed" in text
    assert "MODELS_DIR=" in text and "Free up at least" in text


def test_preflight_disk_grouped_per_volume(env, monkeypatch):
    ollama_dir = env.tmp / "ollama"
    for d in (env.models, env.workspace, ollama_dir):
        d.mkdir()
    volumes = {env.models: 1, env.workspace: 2, ollama_dir: 3}

    def vol(path):
        return volumes[Path(path)]

    monkeypatch.setattr(ai, "_volume_id", vol)
    free = {1: 100 * GB, 2: 10 * GB, 3: 100 * GB}
    monkeypatch.setattr(ai, "disk_free", lambda path: free[vol(path)])
    check = ai.check_disk(
        weights_bytes=70 * GB,
        comfy_bytes=15 * GB,
        ollama_bytes=6 * GB,
        models=env.models,
        workspace=env.workspace,
        ollama_dir=ollama_dir,
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


def test_preflight_ollama_present_but_down_is_info(env):
    check = ai.check_ollama(True, False)
    assert check.status == ai.INFO and "ollama serve" in check.lines[0]


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
    assert env.calls == []
    assert not env.workspace.exists() and not env.models.exists()


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


# --- ComfyUI / nodes / Ollama steps -------------------------------------------


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


def test_ollama_linux_without_passwordless_sudo_needs_you(env, monkeypatch):
    monkeypatch.setattr(ai.sys, "platform", "linux")
    env.which.update({"curl": "/usr/bin/curl", "sudo": "/usr/bin/sudo"})

    def fake_run(cmd, *, timeout=600.0, env=None):
        env_calls.append(cmd)
        return (1, "a password is required") if cmd[:2] == ["sudo", "-n"] else (0, "")

    env_calls: list[list[str]] = []
    monkeypatch.setattr(ai, "_run", fake_run)
    res = ai.step_ollama(True)
    assert res.status == ai.NEEDS_YOU
    assert ai.OLLAMA_LINUX_SCRIPT in res.detail
    assert env_calls == [["sudo", "-n", "true"]]


def test_ollama_linux_with_passwordless_sudo_runs_official_script(env, monkeypatch, capsys):
    monkeypatch.setattr(ai.sys, "platform", "linux")
    env.which.update({"curl": "/usr/bin/curl", "sudo": "/usr/bin/sudo"})

    which = env.which

    def fake_run(cmd, **kw):
        if cmd[0] == "sh":
            which["ollama"] = "/usr/local/bin/ollama"
        return 0, ""

    monkeypatch.setattr(ai, "_run", fake_run)
    assert ai.step_ollama(True).status == ai.OK
    assert ai.OLLAMA_LINUX_SCRIPT in capsys.readouterr().out


def test_ollama_windows_uses_winget(env, monkeypatch):
    monkeypatch.setattr(ai.sys, "platform", "win32")
    env.which["winget"] = "winget"
    plan = ai.step_ollama(False)
    assert plan.status == ai.PLANNED and "Ollama.Ollama" in plan.detail


def test_ffmpeg_linux_never_sudo_silently(env, monkeypatch):
    monkeypatch.setattr(ai.sys, "platform", "linux")
    env.which["apt-get"] = "/usr/bin/apt-get"
    res = ai.step_ffmpeg(True)
    assert res.status == ai.NEEDS_YOU
    assert "sudo apt-get install -y ffmpeg" in res.detail
    assert env.calls == []


def test_ollama_model_pull_failure_is_nonfatal(env, monkeypatch):
    env.which["ollama"] = "/usr/bin/ollama"
    monkeypatch.setattr(setup_mod, "ollama_list_text", lambda: "nomic-embed-text:latest  abc  274 MB")
    monkeypatch.setattr(ai, "_run", lambda cmd, **k: (1, "pull model manifest: file does not exist"))
    res = ai.step_ollama_models(True)
    assert res.status == ai.NONFATAL and not res.required
    assert "qwen3-vl-heretic" in res.detail and "nomic" not in res.detail.split("could not pull")[1]


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
    rc = ai.cmd_automatic_install(_args(yes=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "AUTOMATIC INSTALL — summary" in out
    assert "Not ready for generate yet. Still needs you:" in out
    assert "Triton + SageAttention" not in out.split("Still needs you:")[1]


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
