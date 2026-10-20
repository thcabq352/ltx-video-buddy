"""TEMPORARY dormant MCP fallback (``master_agent.fallback_mcp``): off by default, pure forwarding.

Delete with master_agent/fallback_mcp.py. Tests that speak MCP skip when the
optional ``mcp`` package (requirements-fallback-mcp.txt) is not installed.

Run: python -m pytest tests/test_fallback_mcp.py -q
"""

from __future__ import annotations

import datetime
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from master_agent import agent_api, fallback_mcp
from master_agent.cli.parser import main

VIDEO_BUDDY = Path(__file__).resolve().parents[1]
ARCHITECTURE = VIDEO_BUDDY.parent / "docs" / "ARCHITECTURE.md"


def _result_json(result):
    """FastMCP.call_tool returns content blocks, or (content, structured) for non-dict returns."""
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def test_disabled_by_default(monkeypatch, capsys):
    monkeypatch.delenv(fallback_mcp.ENABLE_ENV, raising=False)
    monkeypatch.setattr(fallback_mcp, "build_server", lambda: pytest.fail("server built while disabled"))
    assert fallback_mcp.enabled() is False
    assert main(["fallback-mcp"]) == 2
    captured = capsys.readouterr()
    assert fallback_mcp.ENABLE_ENV in captured.err
    assert captured.out.strip() == ""


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_exact_opt_in_enables(monkeypatch, value):
    monkeypatch.setenv(fallback_mcp.ENABLE_ENV, value)
    assert fallback_mcp.enabled() is False


def test_enabled_without_mcp_package_explains_install(monkeypatch, capsys):
    monkeypatch.setenv(fallback_mcp.ENABLE_ENV, "1")
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setattr(fallback_mcp, "_serve_stdio", lambda _s: pytest.fail("served without mcp"))
    assert main(["fallback-mcp"]) == 2
    assert "requirements-fallback-mcp.txt" in capsys.readouterr().err


def test_normal_entrypoints_never_import_the_shim():
    code = (
        "import sys; import master_agent.__main__, master_agent.agent_api, master_agent.cli.agent; "
        "print('master_agent.fallback_mcp' in sys.modules, 'mcp' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(VIDEO_BUDDY), capture_output=True, text=True, timeout=120
    )
    assert proc.stdout.strip().splitlines()[-1] == "False False", proc.stderr


def test_base_install_unchanged_and_shim_marked_temporary():
    base = (VIDEO_BUDDY / "requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(line.strip().startswith("mcp") for line in base.splitlines())
    extra = (VIDEO_BUDDY / "requirements-fallback-mcp.txt").read_text(encoding="utf-8")
    assert "mcp" in extra and "TEMPORARY" in extra
    datetime.date.fromisoformat(fallback_mcp.REMOVE_AFTER)
    assert "TEMPORARY" in (fallback_mcp.__doc__ or "")
    doc = ARCHITECTURE.read_text(encoding="utf-8")
    assert "## Revert / fallback path" in doc
    assert fallback_mcp.REMOVE_AFTER in doc and fallback_mcp.ENABLE_ENV in doc


def test_tool_list_parity_with_agent_api():
    pytest.importorskip("mcp")
    import anyio

    tools = {t.name: t for t in anyio.run(fallback_mcp.build_server().list_tools)}
    assert set(tools) == set(agent_api.TOOLS)
    for name, fn in agent_api.TOOLS.items():
        params = inspect.signature(fn).parameters
        schema = tools[name].inputSchema
        assert set(schema.get("properties") or {}) == set(params), name
        required = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert set(schema.get("required") or []) == required, name
        assert tools[name].description == inspect.getdoc(fn)
    assert {"lightx2v", "lightx2v_strength"} <= set(tools["create_video"].inputSchema["properties"])


def test_call_round_trip_forwards_args_and_result_unchanged(monkeypatch):
    pytest.importorskip("mcp")
    import anyio

    seen: dict = {}

    def probe(request: str, k: int = 3, upscale: str | None = None) -> dict:
        """Probe tool."""
        seen.update(request=request, k=k, upscale=upscale, thread=threading.current_thread())
        return {"status": "done", "echo": request, "k": k, "nested": [1, {"a": None}]}

    monkeypatch.setitem(agent_api.TOOLS, "probe", probe)
    server = fallback_mcp.build_server()
    out = _result_json(anyio.run(server.call_tool, "probe", {"request": "a fox", "k": 5}))
    assert out == {"status": "done", "echo": "a fox", "k": 5, "nested": [1, {"a": None}]}
    assert seen["request"] == "a fox" and seen["k"] == 5 and seen["upscale"] is None
    assert seen["thread"] is not threading.main_thread()

    real = _result_json(anyio.run(server.call_tool, "create_video", {"request": "  "}))
    assert real == agent_api.create_video("  ")

    bad_strength = {"request": "a fox", "lightx2v": False, "lightx2v_strength": 5.0}
    forwarded = _result_json(anyio.run(server.call_tool, "create_video", bad_strength))
    assert forwarded == agent_api.create_video(**bad_strength)
    assert "lightx2v_strength" in forwarded["error"]


def test_stdio_round_trip_subprocess():
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {
        **os.environ,
        fallback_mcp.ENABLE_ENV: "1",
        "PYTHONPATH": str(VIDEO_BUDDY),
        "LLAMACPP_AUTOSTART": "0",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "master_agent", "fallback-mcp"], env=env, cwd=str(VIDEO_BUDDY)
    )

    async def session_round_trip():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                bad = await session.call_tool("create_video", {"request": "  "})
                about = await session.call_tool("about", {})
                return listed, bad, about

    async def bounded():
        with anyio.fail_after(120):
            return await session_round_trip()

    listed, bad, about = anyio.run(bounded)
    assert {t.name for t in listed.tools} == set(agent_api.TOOLS)
    assert json.loads(bad.content[0].text) == agent_api.create_video("  ")
    assert json.loads(about.content[0].text)["name"] == agent_api.about()["name"]
