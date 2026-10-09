"""Hermes MCP server — exposes VIDEO BUDDY as sub-agent tools.

Run standalone (Hermes spawns it over stdio):
    .venv/Scripts/python.exe master_agent/mcp_server.py

Tools (bodies in ``master_agent.agent_api``): health, create_video,
plan_storyboard, judge_asset, search_workflows, search_runs, kb_ingest,
list_models, validate_workflow, create_character, train_lora.

NOTE: stdio is the protocol channel — anything the pipeline prints would
corrupt it. ``__main__`` hands the real stdout to the transport once and
points ``sys.stdout`` at stderr for the rest of the process.

Tools are plain sync functions (importable and callable directly). When
registered with FastMCP they run in a worker thread so a long render does
not block pings or other calls on the event loop.

Set ``LLAMACPP_AUTOSTART=0`` in the gateway's environment to stop this
process from launching its own llama-server (useful when several gateways
each spawn an MCP child).
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

# Allow running as a plain script (no package context / cwd independence)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from mcp.server.fastmcp import FastMCP  # noqa: E402
except ImportError:
    FastMCP = None


class _ToolPassthrough:
    """Call tool functions directly when the mcp package is not installed.

    ``python -m`` / the script entry still need mcp. Unit tests import
    ``create_video`` without starting the stdio server.
    """

    def tool(self):
        def deco(fn):
            return fn

        return deco

    def run(self) -> None:
        raise ImportError(
            "mcp is required to run the master-agent server. "
            "Install video_buddy/requirements.txt."
        )


mcp = FastMCP("master-agent") if FastMCP is not None else _ToolPassthrough()


class _ThreadedTools:
    """Register sync tools with FastMCP behind an async worker-thread wrapper.

    The decorated name stays the original sync function, so tests and other
    callers can call it directly.
    """

    def __init__(self, server):
        self._server = server

    def tool(self):
        def deco(fn):
            if FastMCP is None:
                return fn

            @functools.wraps(fn)
            async def _threaded(*args, **kwargs):
                import anyio

                return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

            self._server.tool()(_threaded)
            return fn

        return deco


tools = _ThreadedTools(mcp)


def run_stdio() -> None:
    """Serve MCP over stdio with the protocol on the real stdout only.

    Everything else that prints (pipeline logs, worker threads, child
    libraries) goes to stderr, for the whole process.
    """
    if FastMCP is None:
        mcp.run()
        return
    import io

    import anyio
    from mcp.server.stdio import stdio_server

    protocol_in = io.TextIOWrapper(
        os.fdopen(os.dup(sys.stdin.fileno()), "rb"), encoding="utf-8", errors="replace"
    )
    protocol_out = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()), "wb"), encoding="utf-8")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    # Child processes (ffmpeg, trainers, Comfy) must not read protocol bytes.
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)

    async def _serve() -> None:
        async with stdio_server(
            stdin=anyio.wrap_file(protocol_in),
            stdout=anyio.wrap_file(protocol_out),
        ) as (read_stream, write_stream):
            server = mcp._mcp_server
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)


from master_agent import agent_api  # noqa: E402

MCP_TOOLS = (
    "health",
    "create_video",
    "plan_storyboard",
    "judge_asset",
    "search_workflows",
    "search_runs",
    "kb_ingest",
    "list_models",
    "validate_workflow",
    "create_character",
    "train_lora",
)

for _name in MCP_TOOLS:
    globals()[_name] = tools.tool()(agent_api.TOOLS[_name])


if __name__ == "__main__":
    import time as _time

    _t = _time.time()
    # Warm heavy native deps on the MAIN thread before the event loop starts.
    # Loading them later inside anyio worker threads stalls ~30s per DLL on
    # this machine (240s+ per tool call). Main-thread import is ~1s.
    try:
        import numpy  # noqa: F401
        import chromadb  # noqa: F401

        from master_agent.kb.store import get_client

        get_client()
    except Exception as e:
        print(f"[mcp_server] warmup failed (tools will fall back): {e}", file=sys.stderr)
    # Tool bodies run in worker threads, so import their modules here too.
    for _mod in ("master_agent.orchestrator.pipeline", "master_agent.judge.judge"):
        try:
            __import__(_mod)
        except Exception as e:
            print(f"[mcp_server] warmup import {_mod} failed: {e}", file=sys.stderr)
    print(f"[mcp_server] warmup {_time.time() - _t:.1f}s", file=sys.stderr)
    try:
        from master_agent.control.versioned_config import announce_config

        print(announce_config(), file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        from master_agent.kb.ingest import schedule_knowledge_ingest

        schedule_knowledge_ingest(stderr=True)
    except Exception as e:
        print(f"[mcp_server] knowledge ingest not scheduled: {e}", file=sys.stderr)
    try:
        from master_agent.llm import prepare_local_llm

        prepare_local_llm()
    except Exception as e:
        print(f"[mcp_server] llama.cpp not started: {e}", file=sys.stderr)
    run_stdio()
