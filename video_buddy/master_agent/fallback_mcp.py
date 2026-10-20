"""TEMPORARY, DORMANT fallback: MCP stdio forwarder over ``agent_api.TOOLS``.

The gateway is ``python -m master_agent agent <tool>``. This module exists only
so the Hermes ``ltx`` bot can be pointed back at an MCP server (id
``master-agent``) if that path misbehaves in production. It holds no tool
logic: every MCP tool is an ``agent_api.TOOLS`` function, registered under the
same name with the same signature and docstring.

Off unless all of these hold: the optional ``mcp`` package is installed
(``requirements-fallback-mcp.txt``), the process was started with
``python -m master_agent fallback-mcp``, and ``BUDDY_FALLBACK_MCP=1``. Nothing
in the package imports this module.

Delete after REMOVE_AFTER once the criteria in docs/ARCHITECTURE.md
("Revert / fallback path") are met: this file, ``cmd_fallback_mcp`` and its
parser entry in ``cli/parser.py``, ``requirements-fallback-mcp.txt`` (and its
line in ``.github/workflows/pytest.yml``), and ``tests/test_fallback_mcp.py``.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys

ENABLE_ENV = "BUDDY_FALLBACK_MCP"
SERVER_NAME = "master-agent"
REMOVE_AFTER = "2026-11-02"
INSTALL_HINT = "pip install -r requirements-fallback-mcp.txt"


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip() == "1"


def _threaded(fn):
    # functools.wraps keeps __wrapped__, so FastMCP builds the schema from fn's signature.
    @functools.wraps(fn)
    async def run(**kwargs):
        import anyio

        return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

    return run


def build_server():
    """FastMCP server with one tool per ``agent_api.TOOLS`` entry. Raises ImportError without ``mcp``."""
    from mcp.server.fastmcp import FastMCP

    from master_agent.agent_api import TOOLS

    server = FastMCP(SERVER_NAME)
    for name, fn in TOOLS.items():
        server.add_tool(_threaded(fn), name=name, description=inspect.getdoc(fn))
    return server


def _warm_imports() -> None:
    # Native deps first loaded inside an anyio worker thread stalled ~30 s per DLL
    # on the Windows tower (240 s+ per tool call); main-thread import is ~1 s.
    for mod in ("numpy", "chromadb", "master_agent.orchestrator.pipeline", "master_agent.judge.judge"):
        try:
            __import__(mod)
        except Exception as exc:
            print(f"[fallback-mcp] warmup import {mod} failed: {exc}", file=sys.stderr)


def _serve_stdio(server) -> None:
    """Protocol on the real stdout only; pipeline prints and child processes never touch it."""
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
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)

    async def _serve() -> None:
        async with stdio_server(
            stdin=anyio.wrap_file(protocol_in),
            stdout=anyio.wrap_file(protocol_out),
        ) as (read_stream, write_stream):
            low = server._mcp_server
            await low.run(read_stream, write_stream, low.create_initialization_options())

    anyio.run(_serve)


def main() -> int:
    if not enabled():
        print(
            f"fallback-mcp is a dormant fallback and is off. Set {ENABLE_ENV}=1 to serve it "
            "(see docs/ARCHITECTURE.md, 'Revert / fallback path'). "
            "The gateway entry is: python -m master_agent agent <tool>",
            file=sys.stderr,
        )
        return 2
    try:
        server = build_server()
    except ImportError as exc:
        print(f"fallback-mcp needs the optional mcp package ({exc}). {INSTALL_HINT}", file=sys.stderr)
        return 2
    print(
        f"[fallback-mcp] TEMPORARY MCP fallback '{SERVER_NAME}' over agent_api. "
        f"Scheduled for removal after {REMOVE_AFTER}.",
        file=sys.stderr,
        flush=True,
    )
    _warm_imports()
    _serve_stdio(server)
    return 0
