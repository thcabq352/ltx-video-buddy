"""Character bible — LLM-distilled identity spec for one character.

A bible pins a name, a rare trigger token, a fixed appearance descriptor
(reused verbatim in every prompt and caption), and 12-16 sheet shot prompts
covering angles / expressions / framings. Malformed LLM output is retried
once, then a deterministic fallback bible is built from the description.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

_TRIGGER_RE = re.compile(r"^[a-z0-9_]{3,20}$")

# Fallback shot grid: angle x expression x framing, simple backgrounds.
_FALLBACK_GRID = [
    ("front view", "neutral expression", "close-up portrait"),
    ("front view", "neutral expression", "medium shot"),
    ("front view", "slight smile", "full body shot"),
    ("three-quarter view", "neutral expression", "close-up portrait"),
    ("three-quarter view", "slight smile", "medium shot"),
    ("three-quarter view", "serious expression", "full body shot"),
    ("profile view", "neutral expression", "close-up portrait"),
    ("profile view", "serious expression", "medium shot"),
    ("back view", "neutral expression", "full body shot"),
    ("back view", "looking over shoulder", "medium shot"),
    ("front view", "serious expression", "close-up portrait"),
    ("three-quarter view", "neutral expression", "full body shot"),
]

_PROMPT = """Design a reusable character identity for image generation.

Character description: {description}

Return STRICT JSON only (no markdown, no commentary) with this exact shape:
{{
  "name": "short lowercase character name (one or two words)",
  "trigger_word": "a single rare token, [a-z0-9_] 3-20 chars, e.g. zxc_mira",
  "appearance": "fixed comma-style descriptor block: age, gender, face, hair, eyes, build, signature clothing — reused verbatim in every prompt",
  "shots": ["12-16 shot prompts, each a short phrase"]
}}

The shots MUST cover: front / three-quarter / profile / back angles,
neutral / smile / serious expressions, closeup / medium / full framings,
all on simple plain backgrounds. Each shot is appended after the appearance
block, so shots must NOT repeat the appearance details.
"""


@dataclass
class CharacterBible:
    """Identity spec for one character."""

    name: str
    trigger_word: str
    appearance: str
    shots: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _slugify(text: str, max_words: int = 3) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "character"


def sanitize_trigger(raw: str, fallback_slug: str) -> str:
    """Coerce a trigger word to a single rare [a-z0-9_]{3,20} token."""
    token = re.sub(r"[^a-z0-9_]", "_", (raw or "").strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    if not _TRIGGER_RE.match(token):
        token = f"zxc_{_slugify(fallback_slug)}"
    if len(token) < 3:
        token = f"zxc_{token}"
    return token[:20]


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _bible_from_dict(data: dict, description: str) -> CharacterBible:
    name = _slugify(str(data.get("name") or description), max_words=2)
    appearance = str(data.get("appearance") or "").strip() or description.strip()
    shots = [str(s).strip() for s in (data.get("shots") or []) if str(s).strip()]
    if len(shots) < 12:
        return _fallback_bible(description)
    trigger = sanitize_trigger(str(data.get("trigger_word") or ""), name)
    return CharacterBible(name=name, trigger_word=trigger, appearance=appearance, shots=shots[:16])


def _fallback_bible(description: str) -> CharacterBible:
    name = _slugify(description, max_words=2)
    shots = [
        f"{angle}, {expr}, {framing}, simple plain background"
        for angle, expr, framing in _FALLBACK_GRID
    ]
    return CharacterBible(
        name=name,
        trigger_word=sanitize_trigger("", name),
        appearance=description.strip(),
        shots=shots,
    )


def make_character_bible(description: str, llm=None) -> CharacterBible:
    """Distill a free-text character description into a CharacterBible.

    ``llm`` is any LangChain-style chat model (string prompt in, AIMessage
    with ``.content`` out); defaults to ``get_llm(temperature=0.4)``.
    """
    if llm is None:
        from master_agent.llm import get_llm

        llm = get_llm(temperature=0.4)
    prompt = _PROMPT.format(description=description.strip())
    for _attempt in range(2):  # malformed output retried once
        try:
            resp = llm.invoke(prompt)
            data = _extract_json(getattr(resp, "content", str(resp)))
        except Exception:
            data = None
        if data:
            return _bible_from_dict(data, description)
    return _fallback_bible(description)
