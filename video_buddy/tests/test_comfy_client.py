"""Comfy HTTP client keeps one pooled connection across polls. No live server.

Run: python -m pytest tests/test_comfy_client.py -q
"""

from __future__ import annotations

import httpx

from master_agent.comfy.client import ComfyClient


def test_prompt_history_and_health_share_one_client(monkeypatch):
    created: list[httpx.Client] = []
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={})
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "test"}})
        return httpx.Response(200, json={})

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        client = real(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("master_agent.comfy.client.httpx.Client", factory)
    ComfyClient.close_pool()
    try:
        first = ComfyClient("http://comfy.test")
        second = ComfyClient("http://comfy.test")
        assert first.queue_prompt({"1": {"class_type": "Note", "inputs": {}}}) == "p1"
        assert second.get_history("p1") == {}
        assert first.health()["system"]["os"] == "test"
        other = ComfyClient("http://other.test")
        assert other.health()["system"]["os"] == "test"
        assert len(created) == 2
        assert first._http() is second._http()
        assert first._http() is not other._http()
    finally:
        ComfyClient.close_pool()
