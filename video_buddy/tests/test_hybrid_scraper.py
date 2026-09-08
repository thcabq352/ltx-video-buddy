"""Hybrid scraper: httpx first, Playwright on login walls / challenges.

Run: .venv/Scripts/python.exe -m pytest tests/test_hybrid_scraper.py -q
"""

from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest

from master_agent.scrape import scrape_website
from master_agent.scrape.detect import looks_like_challenge
from master_agent.scrape.hybrid import MAX_RESPONSE_BYTES


FACEBOOK_CHALLENGE = """
<html><head><title>Facebook</title></head>
<body>
  <form id="login_form">Log in to Facebook</form>
  <p>You must log in to continue.</p>
</body></html>
"""

ARTICLE_HTML = """
<html><head><title>Demo Brand</title>
<meta property="og:image" content="/og.jpg" />
</head>
<body>
  <script>void 0</script>
  <main>
    <h1>Hero</h1>
    <h2>Materials</h2>
    <p>Brushed steel and warm oak.</p>
    <img src="/hero.jpg" alt="hero plate" />
  </main>
</body></html>
"""


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_facebook_challenge_routes_to_playwright():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=FACEBOOK_CHALLENGE.encode(),
        )

    playwright_fetch = Mock(return_value=ARTICLE_HTML)
    result = scrape_website(
        "https://www.facebook.com/watch?v=1",
        client=_client(handler),
        playwright_fetch=playwright_fetch,
        robots_check=lambda url, ua=None: True,
    )
    playwright_fetch.assert_called_once()
    assert result.backend == "playwright"
    assert "Brushed steel" in result.text
    assert result.error == ""


def test_robots_deny_skips_httpx_and_playwright():
    playwright_fetch = Mock(return_value=ARTICLE_HTML)
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, text=ARTICLE_HTML)

    result = scrape_website(
        "https://example.com/secret",
        client=_client(handler),
        playwright_fetch=playwright_fetch,
        robots_check=lambda url, ua=None: False,
    )
    assert result.robots_ok is False
    assert "robots.txt" in result.error
    assert hits["n"] == 0
    playwright_fetch.assert_not_called()


def test_static_httpx_extracts_text_and_images():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=ARTICLE_HTML.encode(),
        )

    playwright_fetch = Mock(side_effect=AssertionError("playwright must not run"))
    result = scrape_website(
        "https://example.com/page",
        client=_client(handler),
        playwright_fetch=playwright_fetch,
        robots_check=lambda url, ua=None: True,
    )
    assert result.backend == "httpx"
    assert result.title == "Demo Brand"
    assert "Hero" in result.headings
    assert "Brushed steel" in result.text
    assert "https://example.com/hero.jpg" in result.image_urls
    assert "https://example.com/og.jpg" in result.image_urls
    assert any(img.get("alt") == "hero plate" for img in result.images)


def test_streamed_body_over_5mib_sets_truncated():
    blob = b"<html><body>" + (b"x" * (MAX_RESPONSE_BYTES + 2048)) + b"</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=blob,
        )

    result = scrape_website(
        "https://example.com/huge",
        client=_client(handler),
        playwright_fetch=Mock(side_effect=AssertionError("no playwright")),
        robots_check=lambda url, ua=None: True,
    )
    assert result.backend == "httpx"
    assert result.truncated is True
    assert result.bytes_read <= MAX_RESPONSE_BYTES


def test_looks_like_challenge_on_facebook_wall():
    assert looks_like_challenge(FACEBOOK_CHALLENGE, 200) is True
    assert looks_like_challenge(ARTICLE_HTML, 200) is False
    assert looks_like_challenge("<html></html>", 403) is True


def test_rejects_non_http():
    result = scrape_website("file:///etc/passwd")
    assert result.error
