"""Hybrid page fetch: httpx for static HTML, Playwright+stealth on walls.

Ported from ltx2.5-research-agent/src/ltx_research_agent/scrape/website.py
and upgraded with a challenge fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from master_agent.scrape.detect import looks_like_challenge
from master_agent.scrape.robots import USER_AGENT, robots_allowed

MAX_RESPONSE_BYTES = 5 * 1024 * 1024
SCRAPE_TIMEOUT_S = 20.0

PlaywrightFetch = Callable[[str], str]
RobotsCheck = Callable[..., bool]


@dataclass
class ScrapeResult:
    url: str
    title: str = ""
    text: str = ""
    headings: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    robots_ok: bool = True
    bytes_read: int = 0
    truncated: bool = False
    backend: str = ""
    error: str = ""


def _apply_stealth(page) -> None:
    try:
        from playwright_stealth import stealth_sync

        stealth_sync(page)
        return
    except Exception:
        pass
    try:
        from playwright_stealth import Stealth

        Stealth().apply_stealth_sync(page)
    except Exception:
        pass


def playwright_fetch_live(url: str) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            _apply_stealth(page)
            page.goto(url, wait_until="load", timeout=int(SCRAPE_TIMEOUT_S * 1000))
            return page.content()
        finally:
            browser.close()


def _read_stream(resp: httpx.Response) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in resp.iter_bytes():
        if total + len(chunk) > MAX_RESPONSE_BYTES:
            keep = MAX_RESPONSE_BYTES - total
            if keep > 0:
                chunks.append(chunk[:keep])
                total += keep
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), truncated


def _decode(raw: bytes, encoding: str | None) -> str:
    return raw.decode(encoding or "utf-8", errors="replace")


def _cap_html(html: str) -> tuple[str, int, bool]:
    raw = html.encode("utf-8", errors="ignore")
    if len(raw) <= MAX_RESPONSE_BYTES:
        return html, len(raw), False
    clipped = raw[:MAX_RESPONSE_BYTES]
    return clipped.decode("utf-8", errors="replace"), len(clipped), True


def scrape_website(
    url: str,
    *,
    client: Optional[httpx.Client] = None,
    playwright_fetch: Optional[PlaywrightFetch] = None,
    robots_check: Optional[RobotsCheck] = None,
) -> ScrapeResult:
    from master_agent.scrape.extract import extract_html

    if not url:
        return ScrapeResult(url="", error="empty url")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ScrapeResult(url=url, error="only http(s) urls are allowed")
    allowed_fn = robots_check or robots_allowed
    if not allowed_fn(url, USER_AGENT):
        return ScrapeResult(url=url, robots_ok=False, error="robots.txt disallows this URL")

    own_client = client is None
    http = client or httpx.Client(
        timeout=SCRAPE_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with http.stream("GET", url) as resp:
            raw, truncated = _read_stream(resp)
            encoding = resp.encoding
            status = resp.status_code
            content_type = (resp.headers.get("content-type") or "").lower()
        html = _decode(raw, encoding)
        html_ok = (not content_type) or ("html" in content_type) or ("xml" in content_type)
        if html_ok and looks_like_challenge(html, status):
            fetch = playwright_fetch or playwright_fetch_live
            html = fetch(url)
            html, n, pw_trunc = _cap_html(html)
            result = extract_html(html, url)
            result.backend = "playwright"
            result.robots_ok = True
            result.bytes_read = n
            result.truncated = pw_trunc
            return result
        if status >= 400:
            return ScrapeResult(url=url, robots_ok=True, error=f"HTTP {status}")
        result = extract_html(html, url)
        result.backend = "httpx"
        result.robots_ok = True
        result.bytes_read = len(raw)
        result.truncated = truncated
        return result
    except Exception as exc:
        return ScrapeResult(url=url, robots_ok=True, error=str(exc))
    finally:
        if own_client:
            http.close()


fetch_page = scrape_website
