"""Failure taxonomy for the learning loop."""

from __future__ import annotations

import re
from typing import Any

TAGS = ("oom", "timeout", "bad_prompt", "model_bug", "judge_rejection")

_RULES = (
    ("oom", re.compile(r"out of memory|cuda oom|cudaoom|allocation on device|\boom\b", re.I)),
    ("timeout", re.compile(r"timeout|timed out|deadline exceeded", re.I)),
    ("judge_rejection", re.compile(r"judge|vision reject|score below|rewrite", re.I)),
    ("bad_prompt", re.compile(r"nsfw|safety|prompt rejected|empty prompt|unrenderable", re.I)),
    ("model_bug", re.compile(r"node .* not found|class_type|shape mismatch|nan|inf", re.I)),
)


def classify_failure(text: str | BaseException) -> dict[str, Any]:
    blob = str(text or "")
    for tag, pat in _RULES:
        if pat.search(blob):
            return {"tag": tag, "detail": blob[:240], "known": True}
    return {"tag": "model_bug", "detail": blob[:240], "known": False}
