"""Phase E web model selector. Same catalog as the CLI. No Hub fetch.

Run: PYTHONPATH=video_buddy python -m pytest tests/test_model_selector_web.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from master_agent.models.selector import (
    SCHEMA,
    STATE_SCHEMA,
    catalog_document,
    save_state,
    version_wipe_filenames,
)
from master_agent.models.weights import SULPHUR_Q3_GGUF, WEIGHT_FILES
from master_agent.web.app import STATIC_DIR, app


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr("master_agent.config.MODELS_DIR", models)
    monkeypatch.setattr("master_agent.config.STATE_DIR", state)
    monkeypatch.setattr("master_agent.models.weights.model_search_roots", lambda: [models])
    monkeypatch.setattr("master_agent.models.selector.probe_disk", lambda _path: 2_000_000_000_000)
    monkeypatch.delenv("HEARTMULA_MODELS_DIR", raising=False)
    return models


def _touch(root: Path, folder: str, name: str) -> Path:
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weight")
    return path


def _forbid_fetch(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"files": [], "soundtrack": [], "update": []}

    def _files(files, yes=False, progress=None):
        calls["files"].append((bool(yes), [item.key for item in files]))
        return []

    def _soundtrack(yes=False, progress=None, root=None):
        calls["soundtrack"].append(bool(yes))
        return []

    def _update(*_a, **_k):
        calls["update"].append(True)
        raise AssertionError("comfy update is out of bounds on the selector page")

    monkeypatch.setattr("master_agent.models.weights.download_files", _files)
    monkeypatch.setattr("master_agent.heartmula.doctor.download_missing_slots", _soundtrack)
    monkeypatch.setattr("master_agent.comfy.updates.apply_update", _update)
    monkeypatch.setattr("master_agent.comfy.updates.execute_plan", _update)
    return calls


def _state(version: str, **extra) -> None:
    payload = {
        "schema": STATE_SCHEMA,
        "active_version": version,
        "kept_versions": [version],
        "per_job_version": False,
        "soundtrack_studio": False,
        "optional_ids": [],
    }
    payload.update(extra)
    save_state(payload)


def _video(plan: dict, row_id: str) -> dict:
    return next(row for row in plan["rows"] if row["id"] == row_id)


@pytest.fixture
def client():
    with TestClient(app) as http:
        yield http


def test_models_tab_reuses_cli_labels_and_does_not_embed_the_catalog():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="selector-card"' in html
    assert 'name="sel-version"' in html
    assert 'value="2.3"' in html
    assert 'value="2.5"' in html
    assert "Enable Soundtrack Studio" in html
    assert "Scan only" in html
    assert "/api/models/selector" in html
    assert "download_label" in html
    assert "eta_label" in html
    assert "disk_free_label" in html
    assert "required_label" in html
    assert "size_label" in html
    assert "will_fetch" in html
    assert "toggle.label" in html
    assert "data.catalog" in html
    filename = WEIGHT_FILES["transformer"].filename
    assert filename not in html
    assert "comfy update" not in html.lower()
    assert "function formatMissing" in html
    assert "formatMissing(v.missing)" in html


def test_get_returns_cli_catalog_and_does_not_write_state(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    calls = _forbid_fetch(monkeypatch)
    response = client.get("/api/models/selector")
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == SCHEMA
    assert body["catalog"] == catalog_document()
    assert body["comfy_update"] is False
    assert body["state"] is None
    assert body["plan"]["version"] == "2.5"
    assert "ETA" in body["text"]
    assert body["plan"]["eta_label"]
    assert body["plan"]["download_label"]
    assert not (tmp_path / "state" / "model_selector.json").exists()
    assert calls["files"] == []
    assert calls["soundtrack"] == []
    assert calls["update"] == []


def test_saved_radio_soundtrack_and_optional_round_trip(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    _forbid_fetch(monkeypatch)
    _state("2.3", soundtrack_studio=True, optional_ids=["ltx23.upscaler"])
    response = client.get("/api/models/selector")
    plan = response.json()["plan"]
    assert plan["version"] == "2.3"
    assert plan["soundtrack_studio"] is True
    assert _video(plan, "ltx23.upscaler")["selected"] is True
    assert _video(plan, "ltx23.diffusion")["downloadable"] is False
    assert _video(plan, "ltx23.diffusion")["required"] is True
    forced = client.get("/api/models/selector?version=2.5")
    assert forced.json()["plan"]["version"] == "2.5"
    assert forced.json()["plan"]["soundtrack_studio"] is False


def test_plan_totals_skip_present_files(monkeypatch, tmp_path, client):
    models = _isolate(monkeypatch, tmp_path)
    _forbid_fetch(monkeypatch)
    bare = client.post("/api/models/selector/plan", json={"version": "2.5", "optional_ids": [], "mbps": 50})
    base = bare.json()["plan"]
    weight = WEIGHT_FILES["transformer"]
    _touch(models, weight.dest_folder, weight.filename)
    present = client.post(
        "/api/models/selector/plan",
        json={"version": "2.5", "optional_ids": ["ltx25.ic_lora"], "mbps": 50},
    )
    plan = present.json()["plan"]
    row = _video(plan, "ltx25.transformer")
    assert row["present"] is True
    assert row["skipped"] is True
    assert row["will_fetch"] is False
    assert plan["download_bytes"] == base["download_bytes"] - weight.size_bytes + WEIGHT_FILES["ic_lora"].size_bytes
    assert "skipped" in row["detail"]
    assert plan["eta_seconds"] > 0
    assert "GB" in plan["download_label"]


def test_ltx23_rows_are_scan_only_and_local_flags_do_not_download(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    calls = _forbid_fetch(monkeypatch)
    preview = client.post("/api/models/selector/plan", json={"version": "2.3", "soundtrack": False})
    plan = preview.json()["plan"]
    video = [row for row in plan["rows"] if row["capability"] == "video_generation"]
    assert video
    assert all(row["downloadable"] is False for row in video)
    assert plan["download_bytes"] == 0
    flags = {flag["id"]: flag for flag in plan["local_flags"]}
    assert flags["local.sulphur_gguf"]["local_only"] is True
    assert flags["local.sulphur_lora"]["downloadable"] is False
    assert flags["local.eros"]["downloadable"] is False
    optional = [row["id"] for row in video if not row["required"]]
    applied = client.post(
        "/api/models/selector/apply",
        json={
            "version": "2.3",
            "optional_ids": optional,
            "soundtrack": False,
            "scan_only": False,
            "yes": True,
        },
    )
    body = applied.json()
    assert applied.status_code == 200
    assert body["exit_code"] == 0
    assert body["plan"]["download_bytes"] == 0
    assert calls["files"] == []
    assert calls["soundtrack"] == []
    saved = json.loads((tmp_path / "state" / "model_selector.json").read_text(encoding="utf-8"))
    assert saved["active_version"] == "2.3"
    assert saved["schema"] == STATE_SCHEMA


def test_scan_only_skips_download_and_state(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    calls = _forbid_fetch(monkeypatch)
    response = client.post(
        "/api/models/selector/apply",
        json={
            "version": "2.5",
            "soundtrack": True,
            "scan_only": True,
            "yes": True,
            "keep": True,
            "wipe": False,
        },
    )
    body = response.json()
    assert body["exit_code"] == 0
    assert "nothing downloaded" in body["text"].lower()
    assert body["plan"]["soundtrack_studio"] is True
    assert body["plan"]["download_bytes"] > 0
    assert not (tmp_path / "state" / "model_selector.json").exists()
    assert calls["files"] == []
    assert calls["soundtrack"] == []
    assert calls["update"] == []


def test_yes_uses_existing_download_paths(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    calls = _forbid_fetch(monkeypatch)
    response = client.post(
        "/api/models/selector/apply",
        json={"version": "2.5", "soundtrack": True, "scan_only": False, "yes": True},
    )
    assert response.json()["exit_code"] == 0
    assert calls["files"]
    assert calls["files"][0][0] is True
    assert "transformer" in calls["files"][0][1]
    assert calls["soundtrack"] == [True]
    assert calls["update"] == []


def test_disk_short_does_not_fetch(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("master_agent.models.selector.probe_disk", lambda _path: 1)
    calls = _forbid_fetch(monkeypatch)
    response = client.post(
        "/api/models/selector/apply",
        json={"version": "2.5", "scan_only": False, "yes": True},
    )
    body = response.json()
    assert body["exit_code"] == 1
    assert body["plan"]["disk_ok"] is False
    assert "Nothing downloaded" in body["text"]
    assert calls["files"] == []
    assert not (tmp_path / "state" / "model_selector.json").exists()


def test_keep_and_wipe_match_cli(monkeypatch, tmp_path, client):
    models = _isolate(monkeypatch, tmp_path)
    calls = _forbid_fetch(monkeypatch)
    _state("2.5")
    weight = WEIGHT_FILES["transformer"]
    kept = _touch(models, weight.dest_folder, weight.filename)
    sulphur = _touch(models, "diffusion_models", SULPHUR_Q3_GGUF)
    outside = tmp_path / "outside" / weight.filename
    outside.parent.mkdir()
    outside.write_bytes(b"leave")

    missing = client.post(
        "/api/models/selector/apply",
        json={"version": "2.3", "scan_only": False, "yes": False},
    )
    assert missing.json()["exit_code"] == 2
    assert "--keep or --wipe" in missing.json()["text"]
    assert kept.is_file()
    saved = json.loads((tmp_path / "state" / "model_selector.json").read_text(encoding="utf-8"))
    assert saved["active_version"] == "2.5"

    both = client.post(
        "/api/models/selector/apply",
        json={"version": "2.3", "scan_only": False, "yes": False, "keep": True, "wipe": True},
    )
    assert both.status_code == 400

    keep = client.post(
        "/api/models/selector/apply",
        json={"version": "2.3", "scan_only": False, "yes": False, "keep": True},
    )
    assert keep.json()["exit_code"] == 2
    kept_state = keep.json()["state"]
    assert kept_state["per_job_version"] is True
    assert kept_state["active_version"] == "2.3"
    assert "2.5" in kept_state["kept_versions"]
    assert kept.is_file()

    _state("2.5")
    assert weight.filename in version_wipe_filenames("2.5")
    wipe = client.post(
        "/api/models/selector/apply",
        json={"version": "2.3", "scan_only": False, "yes": False, "wipe": True},
    )
    assert wipe.json()["exit_code"] == 2
    assert not kept.is_file()
    assert sulphur.is_file()
    assert outside.is_file()
    assert calls["files"] == []
    assert calls["update"] == []


def test_selector_rejects_unknown_input(monkeypatch, tmp_path, client):
    _isolate(monkeypatch, tmp_path)
    _forbid_fetch(monkeypatch)
    bad_version = client.post("/api/models/selector/plan", json={"version": "9.9"})
    assert bad_version.status_code == 400
    bad_optional = client.post(
        "/api/models/selector/plan",
        json={"version": "2.5", "optional_ids": ["not-a-row"]},
    )
    assert bad_optional.status_code == 400
    bad_rate = client.post("/api/models/selector/plan", json={"version": "2.5", "mbps": 0})
    assert bad_rate.status_code == 400
