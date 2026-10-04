"""Late-bound ComfyClient so tests can patch master_agent.__main__.ComfyClient."""

from __future__ import annotations


def comfy_client(*args, **kwargs):
    """Construct the CLI Comfy client.

    Resolved from the entry module at call time. ``cmd_run`` / ``cmd_comfy``
    tests replace ``master_agent.__main__.ComfyClient``.
    """
    import master_agent.__main__ as entry

    return entry.ComfyClient(*args, **kwargs)
