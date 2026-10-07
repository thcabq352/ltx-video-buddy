"""Ingest a ComfyUI API graph, learn its widgets, dry-run, and queue it.

Phase A is generic graphs only. Catalog variants, inoutpaint, sulphur, and
lipsync stay on their existing ``--variant`` paths. Storage is
``state/ingested/<slug>/`` and is not promoted into ``workflows/``.
"""

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.validate import MissingCustomNodeError

__all__ = ["IngestError", "MissingCustomNodeError"]
