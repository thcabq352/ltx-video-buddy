"""Comfy Partner template pointers. Not queueable local graphs.

Video Buddy's catalog is on-disk Comfy graphs (LTX, MiniMax H3, Wan, Flux).
It has no ByteDance / Partner HTTP client. These ids exist so ``workflows``
and a refused ``run`` can point operators at official Comfy templates.

They are absent from ``WORKFLOW_FILES``, ``/api/variants``, and the studio
picker. Do not add ``workflows/**/*_api.json`` for them: the catalog scanner
would advertise a cloud graph as a local variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Official workflow_templates ids (index date 2026-09-25, min ComfyUI 0.37.3).
# Scout widgets use model "Seedance 2.5 Draft" at 480p. Promote node is shared.
PROMOTE_NODE = "ByteDance2DraftToFinalVideoNode"
MODEL_OPTION = "Seedance 2.5 Draft"
API_MODEL_ID = "dreamina-seedance-2-5-260628"
DOC_REL = "docs/SEEDANCE_2_5_DRAFT.md"

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
    """Raised when a caller asks Buddy to queue a Partner template."""


@dataclass(frozen=True)
class PartnerPointer:
    id: str
    template: str
    mode: str
    title: str
    scout_node: str

    def keys(self) -> tuple[str, ...]:
        return (self.id, self.template)

    def as_list_item(self) -> dict[str, str | bool]:
        return {
            "id": self.id,
            "path": "",
            "name": self.title,
            "kind": "pointer",
            "family": "partner",
            "queueable": False,
            "template": self.template,
            "description": (
                f"Comfy Partner template {self.template}. Not a local graph. "
                f"Scout {self.scout_node}; promote {PROMOTE_NODE}."
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
    """Best stub when the brief names Seedance but not a template id."""
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


def format_refusal(pointer: PartnerPointer) -> str:
    others = ", ".join(item.template for item in POINTERS)
    return (
        "Seedance 2.5 Draft is a Comfy Partner template, not a Video Buddy graph. "
        "Buddy does not call ByteDance.\n"
        f"Open Comfy template {pointer.template} ({pointer.title}). "
        f"Stub id {pointer.id}.\n"
        f"Scout: model {MODEL_OPTION!r} on {pointer.scout_node} locks 480p and "
        f"returns draft_task_id (API draft:true, model {API_MODEL_ID}). "
        "Fix the seed (turn off randomize) before promote. "
        f"Promote: {PROMOTE_NODE} (ByteDance Seedance 2.5 Draft to Final Video) "
        "→ native 1080p. Draft id lasts about 7 days.\n"
        f"Templates: {others}. Doc: video_buddy/{DOC_REL}. "
        "To render on this machine instead, pass --variant ltx25_t2v_i2v "
        "(or another on-disk catalog id)."
    )


def partner_refusal(
    request: str | None,
    variant: str | None,
    *,
    match_request: bool = True,
) -> str | None:
    """Refusal text when this call would pretend to queue Seedance.

    A concrete local variant (``ltx25_t2v_i2v``, ``base``, …) is honored even
    if the brief mentions Seedance. ``None`` / ``auto`` follow the brief.
    """
    named = lookup_pointer(variant)
    if named is not None:
        return format_refusal(named)
    forced = (variant or "").strip().lower()
    if forced and forced != "auto":
        return None
    if match_request and request_asks_seedance(request):
        return format_refusal(pointer_for_request(request))
    return None
