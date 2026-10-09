"""MCP / A2A media paths are fenced to Buddy's media folders (U8)."""

from __future__ import annotations

from pathlib import Path

import master_agent.config as cfg
from master_agent.a2a.protocol import TaskStore, handle_rpc
from master_agent.media_paths import is_allowed_media_path, media_path_error


def test_outputs_and_uploads_are_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(cfg, "STATE_DIR", tmp_path / "state")
    assert is_allowed_media_path(tmp_path / "outputs" / "a.mp4")
    assert is_allowed_media_path(tmp_path / "state" / "uploads" / "x.png")
    assert not is_allowed_media_path(tmp_path / "state" / "budget.json")


def test_traversal_and_outside_paths_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.delenv("MEDIA_EXTRA_ROOTS", raising=False)
    assert not is_allowed_media_path(tmp_path / "outputs" / ".." / "secret.png")
    assert not is_allowed_media_path(Path.home() / ".hermes" / "auth.json")
    assert media_path_error("image", "/etc/passwd")
    assert media_path_error("image", "file:///etc/passwd")
    assert media_path_error("image", None) is None


def test_symlink_escape_is_refused(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"x")
    link = outputs / "link.png"
    link.symlink_to(secret)
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", outputs)
    monkeypatch.delenv("MEDIA_EXTRA_ROOTS", raising=False)
    assert not is_allowed_media_path(link)


def test_extra_roots_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_EXTRA_ROOTS", str(tmp_path))
    assert is_allowed_media_path(tmp_path / "clip.mp4")


def test_agent_create_video_refuses_outside_path(monkeypatch):
    monkeypatch.delenv("MEDIA_EXTRA_ROOTS", raising=False)
    from master_agent.agent_api import create_video, judge_asset

    out = create_video("a cat", image_path="/etc/hostname", dry_run=True)
    assert out["status"] == "error"
    assert "must be under" in out["error"]
    judged = judge_asset("/etc/hostname")
    assert judged["status"] == "error"


def test_a2a_refuses_outside_path_before_creating_task(monkeypatch):
    monkeypatch.delenv("MEDIA_EXTRA_ROOTS", raising=False)
    store = TaskStore()
    submitted = []
    resp = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "parts": [
                        {"type": "text", "text": "animate this"},
                        {"type": "file", "file": {"uri": "file:///etc/hostname", "mimeType": "image/png"}},
                    ]
                }
            },
        },
        store=store,
        submit=lambda tid, body: submitted.append(body),
    )
    assert resp["error"]["code"] == -32602
    assert submitted == []


def test_a2a_bare_comfy_input_name_passes_through(monkeypatch):
    monkeypatch.delenv("MEDIA_EXTRA_ROOTS", raising=False)
    store = TaskStore()
    submitted = []
    resp = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {"text": "animate"},
                "metadata": {"image_path": "face.png", "confirm": True},
            },
        },
        store=store,
        submit=lambda tid, body: submitted.append(body),
    )
    assert "result" in resp
    assert submitted[0]["image_path"] == "face.png"


def test_a2a_real_render_needs_confirm():
    store = TaskStore()
    submitted = []
    payload = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"text": "rain"}}}
    resp = handle_rpc(payload, store=store, submit=lambda tid, body: submitted.append(body))
    assert "confirm" in resp["error"]["message"]
    assert submitted == []
    payload["params"]["metadata"] = {"dry_run": True}
    assert "result" in handle_rpc(payload, store=store, submit=lambda tid, body: submitted.append(body))
    payload["params"]["metadata"] = {"confirm": True}
    assert "result" in handle_rpc(payload, store=store, submit=lambda tid, body: submitted.append(body))
    assert len(submitted) == 2
