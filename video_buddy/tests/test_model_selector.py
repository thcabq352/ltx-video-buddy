"""Phase C model selector. No Hub fetch, no GPU, no Comfy process.

Run: PYTHONPATH=video_buddy python -m pytest tests/test_model_selector.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.config import DEFAULT_ALL_IN_ONE_CKPT
from master_agent.heartmula.doctor import HUB_USED_STORAGE_BYTES
from master_agent.models.selector import (
    CATALOG_PATH,
    REQUIRED_LABEL,
    SCHEMA,
    apply_selection,
    ask_keep_or_wipe,
    catalog_document,
    eta_seconds,
    format_eta,
    job_version_block,
    load_state,
    render_plan,
    save_state,
    version_for_variant,
    version_ready,
    version_wipe_filenames,
    video_models,
    wipe_version,
    assess,
)
from master_agent.models.weights import SULPHUR_Q3_GGUF, WEIGHT_FILES


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr("master_agent.config.MODELS_DIR", models)
    monkeypatch.setattr("master_agent.config.STATE_DIR", state)
    monkeypatch.setattr(
        "master_agent.models.weights.model_search_roots",
        lambda: [models],
    )
    monkeypatch.setattr(
        "master_agent.models.selector.probe_disk",
        lambda _path: 2_000_000_000_000,
    )
    return models


def _touch(root: Path, folder: str, name: str) -> Path:
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weight")
    return path


def _forbid_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _boom(*_a, **_k):
        calls.append("hf")
        raise AssertionError("Hub fetch is out of bounds in selector tests")

    monkeypatch.setattr("master_agent.models.weights.download_files", _boom)
    monkeypatch.setattr("master_agent.heartmula.doctor.download_missing_slots", _boom)
    return calls


def test_catalog_groups_by_capability_and_has_no_hub_urls():
    document = catalog_document()
    assert document["schema"] == SCHEMA
    assert document["group_by"] == "capability"
    ids = [cap["id"] for cap in document["capabilities"]]
    assert ids == ["video_generation", "soundtrack", "local_studio"]
    blob = json.dumps(document).lower()
    assert "https://" not in blob
    assert "http://" not in blob
    assert "vantagewithai" not in blob
    assert "civitai" not in blob
    soundtrack = document["capabilities"][1]
    assert soundtrack["toggle"]["label"] == "Enable Soundtrack Studio"
    assert "heartmula" in soundtrack["toggle"]["wraps"]
    local_rows = document["capabilities"][2]["models"]
    assert local_rows
    assert all(row["local_only"] and row["downloadable"] is False for row in local_rows)


def test_catalog_file_matches_builder():
    assert json.loads(CATALOG_PATH.read_text(encoding="utf-8")) == catalog_document()


def test_version_radio_swaps_video_rows():
    names_23 = {row.filename for row in video_models("2.3")}
    names_25 = {row.filename for row in video_models("2.5")}
    assert names_23.isdisjoint(names_25)
    required_25 = [row for row in video_models("2.5") if row.required]
    assert {row.weight_key for row in required_25} == {
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "spatial_upscaler",
    }
    assert all(row.to_dict()["required_label"] == REQUIRED_LABEL for row in required_25)
    assert all(row.to_dict()["locked"] is True for row in required_25)
    for row in video_models("2.5"):
        assert row.size_bytes == WEIGHT_FILES[row.weight_key].size_bytes
    ic = next(row for row in video_models("2.5") if row.weight_key == "ic_lora")
    assert ic.required is False
    assert ic.package == "ic_lora"
    assert ic.size_bytes == WEIGHT_FILES["ic_lora"].size_bytes


def test_running_total_eta_and_skip_present(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    base = assess("2.5", [], soundtrack=False, roots=[empty], mbps=50)
    with_ic = assess("2.5", ["ltx25.ic_lora"], soundtrack=False, roots=[empty], mbps=50)
    extra = WEIGHT_FILES["ic_lora"].size_bytes
    assert with_ic.download_bytes == base.download_bytes + extra
    assert with_ic.eta_seconds == pytest.approx(eta_seconds(with_ic.download_bytes, 50))
    assert eta_seconds(50_000_000, 50) == pytest.approx(8.0)
    assert format_eta(8) == "8s"
    text = render_plan(with_ic)
    assert REQUIRED_LABEL in text
    assert "Video generation" in text
    assert "Soundtrack" in text
    assert "running" in text
    assert "ETA" in text

    for key in ("transformer", "text_encoder", "video_vae", "audio_vae", "spatial_upscaler"):
        weight = WEIGHT_FILES[key]
        _touch(empty, weight.dest_folder, weight.filename)
    present = assess("2.5", ["ltx25.ic_lora"], soundtrack=False, roots=[empty], mbps=50)
    fetched = [row.model.weight_key for row in present.rows if row.will_fetch]
    assert "transformer" not in fetched
    assert fetched == ["ic_lora"]
    assert present.download_bytes == extra


def test_disk_gate_blocks_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    calls = _forbid_hub(monkeypatch)
    monkeypatch.setattr("master_agent.models.selector.probe_disk", lambda _path: 1024)
    code, plan = apply_selection(
        "2.5",
        [],
        soundtrack=False,
        yes=True,
        wipe=False,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 1
    assert plan.disk_ok is False
    assert calls == []
    assert load_state() is None


def test_yes_calls_existing_download_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    seen: list[tuple] = []

    def _download(files, *, yes: bool, progress=print):
        assert yes is True
        seen.append(tuple(item.key for item in files))
        return []

    monkeypatch.setattr("master_agent.models.weights.download_files", _download)
    monkeypatch.setattr(
        "master_agent.heartmula.doctor.download_missing_slots",
        lambda **_k: (_ for _ in ()).throw(AssertionError("soundtrack off")),
    )
    code, _plan = apply_selection(
        "2.5",
        ["ltx25.distilled_lora"],
        soundtrack=False,
        yes=True,
        wipe=False,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 0
    assert seen
    assert "transformer" in seen[0]
    assert "distilled_lora" in seen[0]
    assert "ic_lora" not in seen[0]
    state = load_state()
    assert state is not None
    assert state["active_version"] == "2.5"
    assert state["per_job_version"] is False


def test_soundtrack_toggle_uses_heartmula_consent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    for key in ("transformer", "text_encoder", "video_vae", "audio_vae", "spatial_upscaler"):
        weight = WEIGHT_FILES[key]
        _touch(models, weight.dest_folder, weight.filename)

    def _slots(_group=None, root=None):
        return [
            {"repo_id": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "present": False, "dest_rel": "a"},
            {"repo_id": "HeartMuLa/HeartCodec-oss-20260123", "present": False, "dest_rel": "b"},
            {"repo_id": "HeartMuLa/HeartTranscriptor-oss", "present": False, "dest_rel": "c"},
        ]

    monkeypatch.setattr("master_agent.heartmula.doctor.scan_slots", _slots)
    heart: list[bool] = []

    def _pull(*, yes: bool, progress=print, root=None):
        assert yes is True
        heart.append(yes)
        return []

    monkeypatch.setattr("master_agent.heartmula.doctor.download_missing_slots", _pull)
    monkeypatch.setattr(
        "master_agent.models.weights.download_files",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("video pack already present")),
    )
    code, plan = apply_selection(
        "2.5",
        [],
        soundtrack=True,
        yes=True,
        wipe=False,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 0
    assert heart == [True]
    from master_agent.heartmula.config import (
        DEFAULT_CODEC_REPO,
        DEFAULT_MULA_REPO,
        DEFAULT_TRANSCRIPTOR_REPO,
    )

    expected = sum(
        HUB_USED_STORAGE_BYTES[repo]
        for repo in (DEFAULT_MULA_REPO, DEFAULT_CODEC_REPO, DEFAULT_TRANSCRIPTOR_REPO)
    )
    assert plan.download_bytes == expected
    assert load_state()["soundtrack_studio"] is True

    heart.clear()
    code, _plan = apply_selection(
        "2.5",
        [],
        soundtrack=False,
        yes=True,
        wipe=False,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 0
    assert heart == []


def test_keep_wipe_and_noninteractive_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    calls = _forbid_hub(monkeypatch)
    name = WEIGHT_FILES["transformer"].filename
    kept = _touch(models, "diffusion_models", name)
    sulphur = _touch(models, "diffusion_models", SULPHUR_Q3_GGUF)
    eros = _touch(models, "checkpoints", DEFAULT_ALL_IN_ONE_CKPT)
    outside = tmp_path / "outside" / name
    outside.parent.mkdir()
    outside.write_bytes(b"leave")
    ltx23_name = video_models("2.3")[0].filename
    ltx23_file = _touch(models, "diffusion_models", ltx23_name)
    save_state(
        {
            "schema": "buddy.model_selector.state/v1",
            "active_version": "2.5",
            "kept_versions": ["2.5"],
            "per_job_version": False,
            "soundtrack_studio": False,
            "optional_ids": [],
        }
    )

    code, _plan = apply_selection(
        "2.3",
        [],
        soundtrack=False,
        yes=False,
        wipe=False,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 2
    assert kept.is_file()
    assert load_state()["active_version"] == "2.5"
    assert calls == []

    code, _plan = apply_selection(
        "2.3",
        [],
        soundtrack=False,
        yes=False,
        wipe=False,
        keep=True,
        scan_only=False,
        roots=[models],
    )
    assert code == 2
    assert kept.is_file()
    state = load_state()
    assert state["per_job_version"] is True
    assert state["active_version"] == "2.3"
    assert set(state["kept_versions"]) == {"2.3", "2.5"}

    blocked = job_version_block("ltx25_t2v_i2v", roots=[models])
    assert blocked is not None
    assert "2.5" in blocked
    assert "models select --version" in blocked

    save_state(
        {
            "schema": "buddy.model_selector.state/v1",
            "active_version": "2.5",
            "kept_versions": ["2.5"],
            "per_job_version": False,
            "soundtrack_studio": False,
            "optional_ids": [],
        }
    )
    code, _plan = apply_selection(
        "2.3",
        [],
        soundtrack=False,
        yes=False,
        wipe=True,
        keep=False,
        scan_only=False,
        roots=[models],
    )
    assert code == 2
    assert not kept.exists()
    assert sulphur.is_file()
    assert eros.is_file()
    assert outside.is_file()
    assert ltx23_file.is_file()
    assert load_state()["active_version"] == "2.3"
    assert load_state()["per_job_version"] is False
    assert "sulphur" not in {name.lower() for name in version_wipe_filenames("2.3")}
    assert DEFAULT_ALL_IN_ONE_CKPT not in version_wipe_filenames("2.3")
    assert ask_keep_or_wipe("2.5") is None


def test_wipe_refuses_paths_outside_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    removed = wipe_version("2.5")
    assert removed == []
    stray = tmp_path / "not-models" / WEIGHT_FILES["transformer"].filename
    stray.parent.mkdir()
    stray.write_bytes(b"nope")
    assert wipe_version("2.5") == []
    assert stray.is_file()


def test_local_flags_detect_without_downloading(tmp_path: Path):
    root = tmp_path / "models"
    _touch(root, "loras/sulphur", "studio.safetensors")
    _touch(root, "loras", "not-sulphur.safetensors")
    _touch(root, "diffusion_models", SULPHUR_Q3_GGUF)
    _touch(root, "checkpoints", DEFAULT_ALL_IN_ONE_CKPT)
    plan = assess("2.3", [], soundtrack=False, roots=[root])
    flags = {row["id"]: row for row in plan.local_flags}
    assert flags["local.sulphur_lora"]["present"] is True
    assert "1 file" in flags["local.sulphur_lora"]["detail"]
    assert "studio.safetensors" not in flags["local.sulphur_lora"]["detail"]
    assert flags["local.sulphur_gguf"]["present"] is True
    assert flags["local.eros"]["present"] is True
    assert all(row["downloadable"] is False for row in plan.local_flags)
    assert version_for_variant("ltx23_i2v_distilled") == "2.3"
    assert version_for_variant("eros") == "2.3"
    assert version_for_variant("ltx25_t2v_i2v") == "2.5"
    assert version_for_variant("h3_t2v") is None


def test_missing_version_offers_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models = _isolate(monkeypatch, tmp_path)
    assert job_version_block("ltx25_t2v_i2v", roots=[models]) is None
    save_state(
        {
            "schema": "buddy.model_selector.state/v1",
            "active_version": "2.5",
            "kept_versions": ["2.5"],
            "per_job_version": False,
            "soundtrack_studio": False,
        }
    )
    other = job_version_block("ltx23_i2v_distilled", roots=[models])
    assert other is not None
    assert "will not silently" in other
    assert "--version 2.3" in other
    missing = job_version_block("ltx25_t2v_i2v", roots=[models])
    assert missing is not None
    assert "missing" in missing
    assert version_ready("2.5", roots=[models]) is False


def test_cli_scan_only_and_run_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    models = _isolate(monkeypatch, tmp_path)
    calls = _forbid_hub(monkeypatch)
    from master_agent.__main__ import cmd_models_select, cmd_run, main

    code = cmd_models_select(
        Namespace(
            ltx_version="2.5",
            optional=None,
            all_optional=False,
            soundtrack=False,
            mbps=50,
            yes=False,
            wipe=False,
            keep=False,
            scan_only=True,
            json=False,
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "LTX 2.5" in out
    assert "Video generation" in out
    assert REQUIRED_LABEL in out
    assert "Soundtrack Studio" in out
    assert load_state() is None
    assert calls == []

    code = main(["download-models", "--selector", "--version", "2.3", "--scan-only"])
    assert code == 0
    scanned = capsys.readouterr().out
    assert "LTX 2.3" in scanned
    assert "Sulphur" in scanned
    assert calls == []

    monkeypatch.setattr(
        "master_agent.models.selector.job_version_block",
        lambda variant: "LTX 2.3 is missing. python -m master_agent models select --version 2.3",
    )
    rc = cmd_run(
        Namespace(
            request="a garden",
            variant="ltx25_t2v_i2v",
            duration=5.0,
            duration_set=True,
        )
    )
    assert rc == 2
    assert "models select --version 2.3" in capsys.readouterr().out


def test_manifest_command_json(capsys):
    from master_agent.__main__ import main

    code = main(["models", "manifest"])
    assert code == 0
    text = capsys.readouterr().out
    start = text.index("{")
    document = json.loads(text[start:])
    assert document["schema"] == SCHEMA
    assert document["capabilities"][0]["id"] == "video_generation"
