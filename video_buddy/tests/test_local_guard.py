"""Studio stays loopback-only unless --allow-remote."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.testclient import TestClient

from master_agent.web.local_guard import LocalOnlyGuard, is_loopback_host


def _guarded_client() -> TestClient:
    app = FastAPI()

    @app.post("/ping")
    def ping():
        return {"ok": True}

    return TestClient(LocalOnlyGuard(app), base_url="http://127.0.0.1:8189")


def test_loopback_names():
    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"):
        assert is_loopback_host(host), host
    for host in ("0.0.0.0", "192.168.1.5", "evil.example", ""):
        assert not is_loopback_host(host), host


def test_loopback_request_passes():
    client = _guarded_client()
    assert client.post("/ping").status_code == 200
    assert client.post("/ping", headers={"Origin": "http://localhost:8189"}).status_code == 200


def test_rebound_host_is_refused():
    client = _guarded_client()
    r = client.post("/ping", headers={"Host": "evil.example:8189"})
    assert r.status_code == 403
    assert "host" in r.json()["detail"]


def test_cross_site_origin_is_refused():
    client = _guarded_client()
    r = client.post("/ping", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert "origin" in r.json()["detail"]


def test_cmd_ui_refuses_public_host_without_opt_in(monkeypatch, capsys):
    import uvicorn

    from master_agent.cli.studio import cmd_ui

    ran = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.append((app, kw)))
    rc = cmd_ui(argparse.Namespace(host="0.0.0.0", port=8189, allow_remote=False))
    assert rc == 2
    assert ran == []
    assert "--allow-remote" in capsys.readouterr().out

    assert cmd_ui(argparse.Namespace(host="127.0.0.1", port=8189, allow_remote=False)) == 0
    assert isinstance(ran[-1][0], LocalOnlyGuard)

    assert cmd_ui(argparse.Namespace(host="0.0.0.0", port=8189, allow_remote=True)) == 0
    assert not isinstance(ran[-1][0], LocalOnlyGuard)
