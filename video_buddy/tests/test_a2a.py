"""A2A protocol: agent card, message/send, tasks/get. No live network.

Run: .venv/Scripts/python.exe -m pytest tests/test_a2a.py -q
"""

from __future__ import annotations

from master_agent.a2a.protocol import TaskStore, a2a_task_state, agent_card, handle_rpc


def test_agent_card_lists_generate_skill():
    card = agent_card()
    assert card["protocolVersion"] == "0.2.9"
    assert card["url"].endswith(":8189/")
    assert any(s["id"] == "generate_video" for s in card["skills"])
    assert card["endpoints"]["card"] == "/.well-known/agent.json"


def test_pipeline_status_maps_to_a2a_state():
    assert a2a_task_state("done") == "completed"
    assert a2a_task_state("paused") == "input-required"
    assert a2a_task_state("held") == "input-required"
    assert a2a_task_state("running") == "working"
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
