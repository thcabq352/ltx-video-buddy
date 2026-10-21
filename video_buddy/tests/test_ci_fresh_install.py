"""scripts/ci/fresh_install_check.py: the checks the fresh-install CI job runs.

No ComfyUI, no network. Only tmp paths are written; pip is mocked.

Run: python -m pytest tests/test_ci_fresh_install.py -q
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent.comfy import manager_config as mc

VB = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fresh_install_check", VB / "scripts" / "ci" / "fresh_install_check.py")
fic = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fic
_spec.loader.exec_module(fic)

LTDRDATA = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main"
GOOD = mc.CHANNEL_URL


def _comfy(root: Path, *, system_user_api: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("")
    (root / "comfy").mkdir(exist_ok=True)
    api = "def get_system_user_directory(name):\n    pass\n" if system_user_api else ""
    (root / "folder_paths.py").write_text(api)
    return root


# --- log scan ------------------------------------------------------------------


def test_scan_log_flags_the_three_failures():
    log = "\n".join([
        "[INFO] Starting server",
        "\x1b[31m[ERROR]\x1b[0m [ComfyUI-Manager] An invalid channel was used: https://x",
        "comfyui_manager.glob.manager_core.InvalidChannel: https://x",
        "   0.0 seconds (IMPORT FAILED): /c/custom_nodes/ComfyUI-GGUF",
        "ModuleNotFoundError: No module named 'gguf'",
        "ImportError: No module named 'triton'",
    ])
    problems = fic.scan_log(log)
    assert len(problems) == 4
    assert problems[0].startswith("[ERROR] [ComfyUI-Manager] An invalid channel")
    assert any("IMPORT FAILED" in p for p in problems)
    assert any("'gguf'" in p for p in problems)


def test_scan_log_clean_start_passes():
    log = (
        "[INFO] Import times for custom nodes:\n"
        "[INFO]    0.0 seconds: /c/custom_nodes/ComfyUI-GGUF\n"
        "[INFO]    0.6 seconds: /c/custom_nodes/ComfyUI-LTXVideo\n"
        "[INFO] Found comfy_kitchen backend triton: {'unavailable_reason': \"ImportError: No module named 'triton'\"}\n"
    )
    assert fic.scan_log(log) == []


def test_scan_log_allows_only_named_modules():
    log = "ModuleNotFoundError: No module named 'triton.language'\nModuleNotFoundError: No module named 'colour'\n"
    assert fic.scan_log(log, allow_missing_modules=["triton"]) == ["ModuleNotFoundError: No module named 'colour'"]


def test_scan_log_cli(tmp_path, capsys):
    log = tmp_path / "comfyui_8188.log"
    log.write_text("[INFO] ok\n")
    assert fic.main(["scan-log", str(log)]) == 0
    log.write_text("ModuleNotFoundError: No module named 'gguf'\n")
    assert fic.main(["scan-log", str(log)]) == 1
    assert "FAIL  scan-log" in capsys.readouterr().out
    assert fic.main(["scan-log", str(tmp_path / "missing.log")]) == 1


# --- pins ----------------------------------------------------------------------


def test_repo_comfy_requirements_are_exact_pins():
    assert fic.read_pins() == {"gguf": "0.19.0", "colour": "0.1.5"}


@pytest.mark.parametrize("line", ["gguf", "gguf>=0.19", "gguf~=0.19.0", "colour==0.1.*"])
def test_read_pins_rejects_loose_requirements(tmp_path, line):
    req = tmp_path / "r.txt"
    req.write_text(f"# comment\n{line}\n")
    with pytest.raises(fic.CheckFailed, match="not an exact"):
        fic.read_pins(req)


def test_pin_mismatches():
    pins = {"gguf": "0.19.0", "colour": "0.1.5"}
    assert fic.pin_mismatches(pins, {"gguf": "0.19.0", "colour": "0.1.5"}) == []
    assert fic.pin_mismatches(pins, {"gguf": "0.20.0", "colour": None}) == [
        "colour: pinned 0.1.5, installed nothing",
        "gguf: pinned 0.19.0, installed 0.20.0",
    ]


def test_installed_versions_reads_the_target_interpreter():
    got = fic.installed_versions(sys.executable, ["pytest", "definitely-not-a-package-xyz"])
    assert got == {"pytest": importlib.metadata.version("pytest"), "definitely-not-a-package-xyz": None}


def test_without_torch_keeps_everything_else():
    text = "torch\ntorchsde\ntorchvision>=0.20\ntorchaudio ; sys_platform != 'darwin'\nnumpy>=1.25.0\n# torch note\n"
    assert fic.without_torch(text) == "torchsde\nnumpy>=1.25.0\n# torch note\n"


def test_check_pins_stress_reinstalls_then_rechecks(tmp_path, monkeypatch):
    req = tmp_path / "comfy-requirements.txt"
    req.write_text("gguf==0.19.0\ncolour==0.1.5\n")
    comfy_reqs = tmp_path / "requirements.txt"
    comfy_reqs.write_text("torch\nnumpy>=1.25.0\ngguf\n")
    pip_calls: list[tuple[str, ...]] = []
    filtered: list[str] = []

    def fake_pip(python, *args):
        pip_calls.append(args)
        if "-r" in args and Path(args[-1]).name.startswith("comfy-reqs-no-torch"):
            filtered.append(Path(args[-1]).read_text())

    monkeypatch.setattr(fic, "_pip", fake_pip)
    monkeypatch.setattr(fic, "installed_versions", lambda py, names: {"gguf": "0.19.0", "colour": "0.1.5"})
    lines = fic.check_pins("py", req, stress=True, comfy_requirements=comfy_reqs)
    assert pip_calls[0] == ("--upgrade", *fic.UNRELATED_UPGRADES)
    assert pip_calls[-1] == ("--upgrade", "-r", str(req))
    assert filtered == ["numpy>=1.25.0\ngguf\n"]
    assert sum(line.startswith("ok ") for line in lines) == 4


def test_check_pins_fails_when_an_upgrade_moves_a_pin(tmp_path, monkeypatch):
    req = tmp_path / "comfy-requirements.txt"
    req.write_text("gguf==0.19.0\n")
    versions = iter([{"gguf": "0.19.0"}, {"gguf": "0.20.0"}])
    monkeypatch.setattr(fic, "_pip", lambda *a: None)
    monkeypatch.setattr(fic, "installed_versions", lambda py, names: next(versions))
    with pytest.raises(fic.CheckFailed, match="pinned 0.19.0, installed 0.20.0"):
        fic.check_pins("py", req, stress=True, comfy_requirements=None)


# --- Klein stills graph --------------------------------------------------------


def _cached_object_info() -> dict:
    return json.loads((VB / "state" / "object_info.json").read_text(encoding="utf-8"))


def _klein() -> dict:
    return json.loads(fic.KLEIN_WORKFLOW.read_text(encoding="utf-8"))


def test_klein_packs_cover_every_custom_node_class():
    oi = _cached_object_info()
    packs = {fic.pack_folder(p).lower() for p in fic.KLEIN_PACKS}
    needed = set()
    for cls in fic.workflow_classes(_klein()):
        module = str(oi[cls].get("python_module") or "")
        if module.startswith("custom_nodes."):
            needed.add(module.split(".", 1)[1].lower())
    assert needed and needed <= packs


def test_klein_graph_passes_against_the_cached_object_info():
    lines = fic.check_workflow(fic.KLEIN_WORKFLOW, _cached_object_info())
    assert any(line.startswith("offline: PASS") for line in lines)


def test_klein_graph_missing_pack_class_fails():
    oi = _cached_object_info()
    oi.pop("ResolutionPicker")
    with pytest.raises(fic.CheckFailed, match="ResolutionPicker"):
        fic.check_workflow(fic.KLEIN_WORKFLOW, oi)


def test_klein_graph_without_res4lyf_samplers_fails():
    """Live object_info on a ComfyUI without RES4LYF has no res_2s / beta57."""
    oi = json.loads(json.dumps(_cached_object_info()))
    req = oi["KSampler"]["input"]["required"]
    req["sampler_name"] = [[s for s in req["sampler_name"][0] if s != "res_2s"]]
    with pytest.raises(fic.CheckFailed, match="res_2s"):
        fic.check_workflow(fic.KLEIN_WORKFLOW, oi)


def test_environment_only_errors_are_weights_and_input_images():
    wf = {"1": {"class_type": "LoadImage"}, "2": {"class_type": "KSampler"}, "3": {"class_type": "UNETLoader"}}
    issue = lambda nid, msg: SimpleNamespace(node_id=nid, input_name="x", message=msg)  # noqa: E731
    report = SimpleNamespace(errors=[
        issue("1", "value 'a.png' not in combo choices: 'example.png'"),
        issue("3", "model file 'flux.safetensors' not in server choices and not found in local model inventory"),
        issue("2", "value 'res_2s' not in combo choices: 'euler'"),
        issue("1", "required input missing (class LoadImage)"),
    ])
    blocking, env_only = fic.split_errors(report, wf)
    assert [i.node_id for i in env_only] == ["1", "3"]
    assert [i.message for i in blocking] == ["value 'res_2s' not in combo choices: 'euler'",
                                            "required input missing (class LoadImage)"]


def test_missing_pack_folders(tmp_path):
    (tmp_path / "custom_nodes" / "rgthree-comfy").mkdir(parents=True)
    (tmp_path / "custom_nodes" / "comfyui-mickmumpitz-nodes").mkdir()
    assert fic.missing_pack_folders(tmp_path, ["rgthree-comfy", "comfyui-mickmumpitz-nodes@1.5.8", "RES4LYF"]) == [
        "RES4LYF"
    ]


# --- Manager channel -----------------------------------------------------------


def test_channel_check_after_repair_and_snapshot_is_stable(tmp_path):
    code = _comfy(tmp_path / "ComfyUI")
    assert fic.channel_problems(code) == [f"{code / 'user' / '__manager' / 'config.ini'} does not exist"]
    mc.ensure_manager_config(code)
    assert fic.channel_problems(code) == []
    snap = fic.snapshot(code)
    assert set(snap) == {"user/__manager/config.ini", "user/__manager/channels.list"}
    mc.ensure_manager_config(code)
    assert fic.snapshot(code) == snap


def test_channel_check_flags_ltdrdata_and_unlisted_channel(tmp_path):
    code = _comfy(tmp_path / "ComfyUI")
    d = code / "user" / "__manager"
    d.mkdir(parents=True)
    (d / "config.ini").write_text(f"[default]\nchannel_url = {LTDRDATA}\n")
    (d / "channels.list").write_text(f"default::{LTDRDATA}\n")
    problems = fic.channel_problems(code)
    assert any("channel_url is " + LTDRDATA in p for p in problems)
    assert any("not a listed channel" in p for p in problems)


def test_channel_check_flags_ltdrdata_left_in_legacy_folder(tmp_path):
    code = _comfy(tmp_path / "ComfyUI")
    mc.ensure_manager_config(code)
    legacy = code.joinpath(*mc.LEGACY_DIR)
    legacy.mkdir(parents=True)
    (legacy / "config.ini").write_text(f"[default]\nchannel_url = {LTDRDATA}\n")
    assert fic.channel_problems(code) == [f"{legacy / 'config.ini'}: channel_url is {LTDRDATA}"]


def test_channel_check_flags_stale_secondary_channel(tmp_path):
    code = _comfy(tmp_path / "ComfyUI")
    mc.ensure_manager_config(code)
    listed = code / "user" / "__manager" / "channels.list"
    listed.write_text(listed.read_text() + f"recent::{LTDRDATA}/node_db/new\n")
    assert fic.channel_problems(code) == [f"{listed}: recent::{LTDRDATA}/node_db/new"]


def test_manager_reset_cli_delete_repair_unchanged(tmp_path):
    code = _comfy(tmp_path / "ComfyUI")
    mc.ensure_manager_config(code)
    snap = tmp_path / "snap.json"
    assert fic.main(["manager-channel", "--comfy", str(code), "--save", str(snap)]) == 0
    assert fic.main(["delete-manager-config", "--comfy", str(code)]) == 0
    assert fic.main(["manager-channel", "--comfy", str(code)]) == 1
    assert fic.main(["delete-manager-config", "--comfy", str(code)]) == 1
    mc.ensure_manager_config(code)
    assert fic.main(["manager-channel", "--comfy", str(code), "--compare", str(snap)]) == 0
    cfg = code / "user" / "__manager" / "config.ini"
    cfg.write_text(cfg.read_text() + "use_uv = True\n")
    assert fic.main(["manager-channel", "--comfy", str(code), "--compare", str(snap)]) == 1


def test_live_channel_payload():
    good = {"selected": "default", "list": [f"default::{GOOD}", f"recent::{GOOD}/node_db/new"]}
    assert fic.live_channel_problems(good) == []
    custom = {"selected": "custom", "list": [f"default::{GOOD}"]}
    assert "InvalidChannel" in fic.live_channel_problems(custom)[0]
    old = {"selected": "default", "list": [f"default::{LTDRDATA}"]}
    assert LTDRDATA in fic.live_channel_problems(old)[0]
