"""MCP tools run off the event loop and stray prints never reach the protocol (U1/U7)."""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from master_agent import mcp_server  # noqa: E402

VIDEO_BUDDY = Path(__file__).resolve().parents[1]


def test_registered_tools_stay_sync_callables():
    assert not callable(getattr(mcp_server.create_video, "__await__", None))
    assert mcp_server.health.__name__ == "health"


def test_slow_tool_does_not_block_other_calls():
    server = FastMCP("t")
    reg = mcp_server._ThreadedTools(server)

    @reg.tool()
    def slow(seconds: float) -> str:
        time.sleep(seconds)
        return "slow"

    @reg.tool()
    def quick() -> str:
        return "quick"

    done: dict[str, float] = {}

    async def main():
        t0 = time.monotonic()

        async def run_slow():
            await server.call_tool("slow", {"seconds": 1.0})
            done["slow"] = time.monotonic() - t0

        async def run_quick():
            await anyio.sleep(0.1)
            await server.call_tool("quick", {})
            done["quick"] = time.monotonic() - t0

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_slow)
            tg.start_soon(run_quick)

    anyio.run(main)
    assert done["quick"] < 0.6 < done["slow"]


def test_stdout_prints_do_not_corrupt_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = tmp_path / "noisy_server.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys
            sys.path.insert(0, {str(VIDEO_BUDDY)!r})
            from master_agent import mcp_server as m

            @m.tools.tool()
            def noisy() -> str:
                print("junk from a worker thread")
                subprocess.run([sys.executable, "-c", "print('junk from a child process')"])
                return "ok"

            m.run_stdio()
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(VIDEO_BUDDY)

    async def main():
        params = StdioServerParameters(command=sys.executable, args=[str(script)], env=env)
        with anyio.fail_after(60), open(tmp_path / "stderr.txt", "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    return await session.call_tool("noisy", {})

    res = anyio.run(main)
    assert not res.isError
    assert res.content[0].text == "ok"
    err = (tmp_path / "stderr.txt").read_text()
    assert "junk from a worker thread" in err
    assert "junk from a child process" in err
