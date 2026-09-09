"""Grok Imagine image generation. Ported from ltx_research_agent/imagine/client.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from master_agent.config import XAI_BASE_URL
from master_agent.xai_oauth import resolve_access_token

XAI_IMAGE_MODEL = "grok-imagine-image-2.0"


def generate_image(
    prompt: str,
    dest: Path,
    *,
    aspect_ratio: str = "16:9",
    model: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    bearer = token or resolve_access_token()
    payload = {
        "model": model or XAI_IMAGE_MODEL,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "n": 1,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{XAI_BASE_URL}/images/generations",
            headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
            json=payload,
        )
    metrics = {"http_status": resp.status_code, "bytes": 0, "path": str(dest)}
    if resp.status_code == 403:
        raise RuntimeError(
            "Imagine returned HTTP 403. SuperGrok OAuth may be gated; set XAI_API_KEY and retry."
        )
    resp.raise_for_status()
    data = resp.json()
    item = (data.get("data") or [{}])[0]
    url = item.get("url")
    b64 = item.get("b64_json")
    if url:
        with httpx.Client(timeout=60.0) as client:
            image = client.get(url)
            image.raise_for_status()
            dest.write_bytes(image.content)
            metrics["bytes"] = len(image.content)
    elif b64:
        import base64

        raw = base64.b64decode(b64)
        dest.write_bytes(raw)
        metrics["bytes"] = len(raw)
    else:
        raise RuntimeError(f"Imagine response missing image: {data}")
    return metrics
