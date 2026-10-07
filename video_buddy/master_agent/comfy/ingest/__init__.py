"""Ingest a ComfyUI API graph, learn its widgets, dry-run, and queue it.

Storage is ``state/ingested/<slug>/``. ``comfy promote`` can copy a draft
into the local ``workflows/`` checkout. That draft is not a catalog default.
"""

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.validate import MissingCustomNodeError

__all__ = ["IngestError", "MissingCustomNodeError"]
