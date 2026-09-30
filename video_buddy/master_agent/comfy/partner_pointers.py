"""Seedance 2.5 Draft field-shape records. Not an execution path.

Pack C video generation is local Comfy only (``http://127.0.0.1:8188``) on
the existing LTX catalog. These ids remember the Partner node shape. The
agent does not load, enable, or queue those graphs, and does not call
comfy.org, BytePlus, ModelArk, or KIE.

They are absent from ``WORKFLOW_FILES``, ``/api/variants``, and the studio
picker. Do not add ``workflows/**/*seedance*.json``: the catalog scanner
would advertise a hosted graph as a local variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

# Recorded Partner shape (ComfyUI #16529). Not a queue target.
PROMOTE_NODE = "ByteDance2DraftToFinalVideoNode"
MODEL_OPTION = "Seedance 2.5 Draft"
API_MODEL_ID = "dreamina-seedance-2-5-260628"
DOC_REL = "docs/SEEDANCE_2_5_DRAFT.md"

FORBIDDEN_CLASS_TYPES = frozenset(
    {
        "ByteDance2TextToVideoNode",
        "ByteDance2FirstLastFrameNode",
        "ByteDance2ReferenceNodeV2",
        "ByteDance2DraftToFinalVideoNode",
    }
)

# Hosts that would leave the machine for video inference. Matched against
# the URL host, so a test double such as ``comfy.test`` is not included.
CLOUD_VIDEO_HOST_MARKERS = (
    "comfy.org",
    "byteplus",
    "modelark",
    "volces.com",
    "volcengineapi.com",
    "kie.ai",
)

_REQUEST_RE = re.compile(
    r"(?i)(?:"
    r"\bseedance\b|"
    r"api_seedance|"
    r"draft_task_id|"
    r"dreamina-seedance|"
    r"draft[\s-]+to[\s-]+final"
    r")"
)


class PartnerPointerError(ValueError):
    """Raised when a caller asks Buddy to queue a Partner / cloud Seedance graph."""


@dataclass(frozen=True)
class PartnerPointer:
    id: str
    template: str
    mode: str
    title: str
    scout_node: str
    executable: bool = False

    def keys(self) -> tuple[str, ...]:
        return (self.id, self.template)

    def as_list_item(self) -> dict[str, str | bool]:
        local = local_variant_for_mode(self.mode)
        return {
            "id": self.id,
            "path": "",
            "name": self.title,
            "kind": "pointer",
            "family": "partner",
            "queueable": False,
            "executable": False,
            "template": self.template,
            "localVariant": local,
            "description": (
                f"Field-shape record of {self.template}. Not executable. "
                f"Generate on http://127.0.0.1:8188 with {local}."
            ),
        }


POINTERS: tuple[PartnerPointer, ...] = (
    PartnerPointer(
        id="seedance25_draft_t2v",
        template="api_seedance2_5_draft_t2v",
        mode="t2v",
        title="Seedance 2.5 Draft: Text to Video",
        scout_node="ByteDance2TextToVideoNode",
    ),
    PartnerPointer(
        id="seedance25_draft_i2v",
        template="api_seedance2_5_draft_i2v",
        mode="i2v",
        title="Seedance 2.5 Draft: Image to Video",
        scout_node="ByteDance2FirstLastFrameNode",
    ),
    PartnerPointer(
        id="seedance25_draft_r2v",
        template="api_seedance2_5_draft_r2v",
        mode="r2v",
        title="Seedance 2.5 Draft: Reference to Video",
        scout_node="ByteDance2ReferenceNodeV2",
    ),
)

_BY_KEY: dict[str, PartnerPointer] = {}
for _pointer in POINTERS:
    for _key in _pointer.keys():
        _BY_KEY[_key.lower()] = _pointer


def partner_graphs_executable() -> bool:
    """Hard closed. Partner Seedance graphs are never an execution path."""
    from master_agent.config import PACK_C_LOCAL_ONLY

    if PACK_C_LOCAL_ONLY.get("partnerGraphsExecutable"):
        return True
    return any(item.executable for item in POINTERS)


def lookup_pointer(key: str | None) -> PartnerPointer | None:
    text = (key or "").strip().replace("\\", "/")
    if not text:
        return None
    found = _BY_KEY.get(text.lower())
    if found is not None:
        return found
    # Template paths operators paste (`templates/api_seedance2_5_draft_t2v.json`).
    stem = text.rsplit("/", 1)[-1]
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return _BY_KEY.get(stem.lower())


def request_asks_seedance(text: str | None) -> bool:
    return bool(_REQUEST_RE.search(text or ""))


def pointer_for_request(text: str | None) -> PartnerPointer:
    """Field-shape stub when the brief names Seedance but not a template id."""
    raw = (text or "").lower()
    if any(
        token in raw
        for token in ("r2v", "reference-to-video", "reference to video", "draft_r2v")
    ):
        return POINTERS[2]
    if any(
        token in raw
        for token in (
            "i2v",
            "image-to-video",
            "image to video",
            "first frame",
            "first-last",
            "flf",
            "draft_i2v",
        )
    ):
        return POINTERS[1]
    return POINTERS[0]


def local_variant_for_mode(mode: str) -> str:
    from master_agent.config import PACK_C_LOCAL_ONLY

    packs = PACK_C_LOCAL_ONLY["localPacks"]
    return str(packs.get(mode) or packs["t2v"])


def local_catalog_for_pack_c_brief(text: str | None) -> str | None:
    """Local catalog id for a Pack C brief. None when the text is not Pack C.

    First-last / flf maps to the local FLF graph. A single first frame maps
    to the local I2V graph. Reference maps to the local multi-reference graph.
    """
    if not request_asks_seedance(text):
        return None
    from master_agent.config import PACK_C_LOCAL_ONLY

    packs = PACK_C_LOCAL_ONLY["localPacks"]
    raw = (text or "").lower()
    if any(
        token in raw
        for token in ("r2v", "reference-to-video", "reference to video", "draft_r2v")
    ):
        return str(packs["r2v"])
    if any(token in raw for token in ("flf2v", "flf", "first-last", "first last")):
        return str(packs["flf"])
    if any(
        token in raw
        for token in (
            "i2v",
            "image-to-video",
            "image to video",
            "first frame",
            "draft_i2v",
        )
    ):
        return str(packs["i2v"])
    return str(packs["t2v"])


def pack_c_video_url_allowed(url: str | None = None) -> bool:
    """True only for loopback Comfy on port 8188."""
    from master_agent.config import COMFYUI_URL

    raw = (COMFYUI_URL if url is None else url).strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"}:
        return False
    port = parsed.port
    if port is None:
        port = 80
    return port == 8188


def cloud_video_host(url: str | None) -> str | None:
    """Cloud video-inference host marker, if this URL would leave the machine."""
    raw = (url or "").strip().lower()
    if not raw:
        return None
    host = (urlparse(raw).hostname or "").lower()
    for marker in CLOUD_VIDEO_HOST_MARKERS:
        if marker in host:
            return marker
    return None


def _explicit_local_variant(variant: str | None) -> bool:
    forced = (variant or "").strip()
    if not forced or forced.lower() == "auto":
        return False
    return lookup_pointer(forced) is None


def format_local_only_refusal(pointer: PartnerPointer | None = None) -> str:
    from master_agent.config import PACK_C_COMFY_URL

    local = local_variant_for_mode(pointer.mode) if pointer is not None else local_variant_for_mode("t2v")
    recorded = pointer.template if pointer is not None else "api_seedance2_5_draft_*"
    return (
        "Local-only is a hard requirement for Pack C Seedance Draft→Final. "
        "No cloud APIs, no cloud services, no hosted inference. "
        f"Partner template {recorded} is a field-shape record and is not executable. "
        f"Generate on {PACK_C_COMFY_URL} with local catalog id {local}."
    )


def format_nonlocal_refusal() -> str:
    from master_agent.config import COMFYUI_URL, PACK_C_COMFY_URL

    return (
        "Local-only is a hard requirement for Pack C Seedance Draft→Final. "
        "No cloud APIs, no cloud services, no hosted inference. "
        f"Video generation stays on {PACK_C_COMFY_URL}. "
        f"Refusing COMFYUI_URL {COMFYUI_URL!r}."
    )


def route_pack_c(
    request: str | None,
    variant: str | None,
    *,
    match_request: bool = True,
) -> tuple[str | None, str | None]:
    """Fail closed onto a local catalog id.

    Returns ``(variant, error)``. ``error`` is set when this call would run
    Partner/cloud Seedance or would send the Pack C burn off loopback.
    A concrete local variant (``ltx25_t2v_i2v``, ``wan22``, ``h3_t2v``, …)
    is kept. Partner stub ids are rewritten to the local pack for that mode.
    """
    pointer = lookup_pointer(variant)
    asks = match_request and request_asks_seedance(request)
    explicit_local = _explicit_local_variant(variant)
    if pointer is None and not asks:
        return variant, None
    if not pack_c_video_url_allowed():
        return None, format_nonlocal_refusal()
    if pointer is not None:
        return local_variant_for_mode(pointer.mode), None
    if asks and not explicit_local:
        return local_catalog_for_pack_c_brief(request), None
    return variant, None


def partner_refusal(
    request: str | None,
    variant: str | None,
    *,
    match_request: bool = True,
) -> str | None:
    """Error when *this key* is still a Partner graph.

    Entrances should call :func:`route_pack_c` first so a Seedance brief
    becomes a local catalog id. A leftover pointer id fails closed here
    and is not queued.
    """
    pointer = lookup_pointer(variant)
    if pointer is not None:
        return format_local_only_refusal(pointer)
    if match_request and request_asks_seedance(request) and not _explicit_local_variant(variant):
        if not pack_c_video_url_allowed():
            return format_nonlocal_refusal()
    return None


def partner_class_types_in(workflow: Any) -> set[str]:
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("class_type", "type"):
                value = node.get(key)
                if isinstance(value, str) and value in FORBIDDEN_CLASS_TYPES:
                    found.add(value)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(workflow)
    return found


def reject_partner_or_cloud_queue(workflow: Any, base_url: str | None) -> str | None:
    """Block Partner graphs and cloud video hosts before ``/prompt``."""
    from master_agent.config import PACK_C_COMFY_URL

    classes = partner_class_types_in(workflow)
    if classes:
        names = ", ".join(sorted(classes))
        return (
            "Local-only is a hard requirement for Pack C Seedance Draft→Final. "
            f"Partner class {names} is not executable. "
            f"Generate on {PACK_C_COMFY_URL} with a local LTX, Wan, or H3 graph."
        )
    host = cloud_video_host(base_url)
    if host:
        return (
            "Local-only is a hard requirement. "
            f"Refusing cloud video host {host!r}. "
            f"Generate on {PACK_C_COMFY_URL}."
        )
    return None
