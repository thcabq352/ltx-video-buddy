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


def test_upload_audio_posts_once_and_raises(tmp_path, monkeypatch):
    import pytest

    from master_agent.comfy.client import ComfyClientError

    audio = tmp_path / "vo.wav"
    audio.write_bytes(b"RIFF")
    posts: list[str] = []

    class _Resp:
        def __init__(self, status, body="{}"):
            self.status_code = status
            self.text = body
            self._body = body

        def json(self):
            return {"name": "vo.wav"} if self.status_code < 400 else {}

    class _Client:
        is_closed = False

        def post(self, url, *args, **kwargs):
            posts.append(url)
            return _Resp(200 if len(posts) == 1 else 400, "nope")

        def close(self):
            self.is_closed = True

    monkeypatch.setattr("master_agent.comfy.client.httpx.Client", lambda *a, **k: _Client())
    try:
        client = ComfyClient("http://comfy.test")
        assert client.upload_audio(audio) == "vo.wav"
        assert posts == ["http://comfy.test/upload/image"]
        posts.clear()

        class _Bad(_Client):
            def post(self, url, *args, **kwargs):
                posts.append(url)
                return _Resp(400, "bad audio")

        monkeypatch.setattr("master_agent.comfy.client.httpx.Client", lambda *a, **k: _Bad())
        ComfyClient.close_pool()
        with pytest.raises(ComfyClientError, match="audio upload failed \\(400\\)"):
            ComfyClient("http://comfy.test").upload_audio(audio)
        assert len(posts) == 1
    finally:
        ComfyClient.close_pool()
