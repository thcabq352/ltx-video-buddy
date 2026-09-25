"""A2A protocol: agent card, message/send, tasks/get. No live network.

Run: .venv/Scripts/python.exe -m pytest tests/test_a2a.py -q
"""

from __future__ import annotations

from master_agent.a2a.protocol import TaskStore, a2a_task_state, agent_card, handle_rpc


def test_agent_card_lists_generate_skill():
    card = agent_card()
    assert card["protocolVersion"] == "0.2.9"
    assert card["url"].endswith(":8189/a2a")
    assert any(s["id"] == "generate_video" for s in card["skills"])
    assert card["endpoints"]["card"] == "/.well-known/agent.json"
    assert card["endpoints"]["card_v1"] == "/.well-known/agent-card.json"


def test_pipeline_status_maps_to_a2a_state():
    assert a2a_task_state("done") == "completed"
    assert a2a_task_state("done_with_warnings") == "completed"
    assert a2a_task_state("paused") == "input-required"
    assert a2a_task_state("held") == "input-required"
    assert a2a_task_state("running") == "working"
    assert a2a_task_state("started") == "working"
    assert a2a_task_state("error") == "failed"


def test_message_send_and_tasks_get_async_lifecycle():
    store = TaskStore()
    submitted = []

    def submit(task_id, body):
        submitted.append((task_id, body["request"]))
        store.set(task_id, state="completed", result={"status": "done", "video_path": "out.mp4"})

    rpc = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": {"parts": [{"type": "text", "text": "rain on a window"}]}},
        },
        store=store,
        submit=submit,
    )
    assert rpc["result"]["status"]["state"] == "working"
    tid = rpc["result"]["id"]
    assert submitted[0][1] == "rain on a window"
    got = handle_rpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": tid}},
        store=store,
        submit=submit,
    )
    assert got["result"]["status"]["state"] == "completed"
    assert got["result"]["artifacts"][0]["parts"][0]["text"] == "out.mp4"


def test_unknown_method_and_empty_message():
    store = TaskStore()
    miss = handle_rpc({"jsonrpc": "2.0", "id": 3, "method": "nope"}, store=store, submit=lambda *a: None)
    assert miss["error"]["code"] == -32601
    empty = handle_rpc(
        {"jsonrpc": "2.0", "id": 4, "method": "message/send", "params": {"message": {}}},
        store=store,
        submit=lambda *a: None,
    )
    assert empty["error"]["code"] == -32602


def test_message_send_defaults_quality_draft_and_duration():
    store = TaskStore()
    captured = {}

    def submit(task_id, body):
        captured.update(body)

    handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "message/send",
            "params": {"message": {"parts": [{"type": "text", "text": "neon rain"}]}},
        },
        store=store,
        submit=submit,
    )
    assert captured["quality"] == "draft"
    assert captured["duration_s"] == 5.0


def test_message_send_h3_photo_voice_includes_warning(tmp_path):
    from master_agent.orchestrator.talking import H3_R2V_AUDIO_LABEL

    store = TaskStore()
    image = tmp_path / "gator.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")
    rpc = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "message/send",
            "params": {
                "message": {
                    "parts": [
                        {"type": "text", "text": "hailuo, the gator says the line"},
                        {"type": "file", "file": {"uri": str(image), "mimeType": "image/png"}},
                        {"type": "file", "file": {"uri": str(audio), "mimeType": "audio/wav"}},
                    ]
                },
                "metadata": {"variant": "h3_r2v", "dry_run": True},
            },
        },
        store=store,
        submit=lambda *_a, **_k: None,
    )
    assert rpc["result"]["warning"] == H3_R2V_AUDIO_LABEL
    assert rpc["result"]["notes"] == H3_R2V_AUDIO_LABEL
    skill = next(s for s in agent_card()["skills"] if s["id"] == "generate_video")
    assert H3_R2V_AUDIO_LABEL in skill["description"]

    quiet = handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "message/send",
            "params": {
                "message": {
                    "parts": [
                        {"type": "text", "text": "she says the line"},
                        {"type": "file", "file": {"uri": str(image), "mimeType": "image/png"}},
                        {"type": "file", "file": {"uri": str(audio), "mimeType": "audio/wav"}},
                    ]
                }
            },
        },
        store=store,
        submit=lambda *_a, **_k: None,
    )
    assert "warning" not in quiet["result"]
    assert "notes" not in quiet["result"]


def test_submit_orchestrator_stores_h3_voice_warning(monkeypatch, tmp_path):
    import time

    from master_agent.a2a.protocol import submit_orchestrator
    from master_agent.orchestrator.talking import H3_R2V_AUDIO_LABEL

    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.dry_run_pipeline",
        lambda *_a, **_k: 0,
    )
    image = tmp_path / "gator.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")
    store = TaskStore()
    submit_orchestrator(
        "h3warnh3warn",
        {
            "request": "use hailuo on this still",
            "dry_run": True,
            "image_path": str(image),
            "audio_path": str(audio),
            "duration_s": 3.9,
        },
        store,
    )
    row = None
    for _ in range(80):
        row = store.get("h3warnh3warn")
        if row and row.get("state") in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert row and row["state"] == "completed"
    assert row["result"]["notes"] == H3_R2V_AUDIO_LABEL
    view = handle_rpc(
        {"jsonrpc": "2.0", "id": 23, "method": "tasks/get", "params": {"id": "h3warnh3warn"}},
        store=store,
        submit=lambda *_a, **_k: None,
    )
    assert H3_R2V_AUDIO_LABEL in view["result"]["status"]["message"]["parts"][0]["text"]


def test_message_send_forwards_photo_and_audio(tmp_path):
    store = TaskStore()
    captured = {}

    def submit(task_id, body):
        captured.update(body)

    image = tmp_path / "face.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")
    handle_rpc(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "message/send",
            "params": {
                "message": {
                    "parts": [
                        {"type": "text", "text": "she says the line"},
                        {"type": "file", "file": {"uri": str(image), "mimeType": "image/png"}},
                        {"type": "file", "file": {"uri": str(audio), "mimeType": "audio/wav"}},
                    ]
                }
            },
        },
        store=store,
        submit=submit,
    )
    assert captured["image_path"] == str(image)
    assert captured["audio_path"] == str(audio)
    assert captured["duration_s"] is None
    assert "image/png" in agent_card()["defaultInputModes"]
    assert "audio/wav" in agent_card()["defaultInputModes"]


def test_submit_orchestrator_uses_gpu_lock(monkeypatch):
    import time

    from master_agent.a2a.protocol import submit_orchestrator

    held: list[str] = []

    class FakeLock:
        def __enter__(self):
            held.append("enter")
            return self

        def __exit__(self, *a):
            held.append("exit")
            return False

    class FakeManager:
        def gpu_lock(self):
            return FakeLock()

    class Result:
        status = "done_with_warnings"

        def to_dict(self):
            return {"status": "done_with_warnings", "video_path": "out.mp4"}

    monkeypatch.setattr("master_agent.web.jobs.MANAGER", FakeManager())
    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.run_pipeline",
        lambda *a, **k: Result(),
    )
    store = TaskStore()
    submit_orchestrator("abc123abc123", {"request": "hi"}, store)
    for _ in range(80):
        row = store.get("abc123abc123")
        if row and row.get("state") in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert held == ["enter", "exit"]
    assert store.get("abc123abc123")["state"] == "completed"


def test_studio_routes_card_health_tasks_and_facade():
    from fastapi.testclient import TestClient

    from master_agent.web import app as web_app

    web_app._A2A = TaskStore()
    client = TestClient(web_app.app)
    paths = {getattr(route, "path", None) for route in web_app.app.routes}
    assert "/health" in paths
    assert "/.well-known/agent-card.json" in paths
    assert "/tasks/{task_id}" in paths
    card = client.get("/.well-known/agent.json").json()
    alias = client.get("/.well-known/agent-card.json").json()
    assert card["url"].endswith("/a2a")
    assert alias == card
    missing = client.get("/tasks/no-such-task")
    assert missing.status_code == 404
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "ltx"
    mux_models = client.get("/p/ltx/v1/models").json()
    assert mux_models["data"][0]["id"] == "ltx"
    def submit(task_id, body):
        web_app._A2A.set(task_id, state="completed", result={"status": "done", "video_path": "x.mp4"})

    from master_agent.hermes.adapter import hermes_complete

    out = hermes_complete(
        {"messages": [{"role": "user", "content": "rain on a window"}]},
        store=web_app._A2A,
        submit=submit,
    )
    tid = out["buddy"]["task_id"]
    got = client.get(f"/tasks/{tid}").json()
    assert got["status"]["state"] == "completed"
    rpc = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 9, "method": "tasks/get", "params": {"id": tid}},
    ).json()
    assert rpc["result"]["status"]["state"] == "completed"
