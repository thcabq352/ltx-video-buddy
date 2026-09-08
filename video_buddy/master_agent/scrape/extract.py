"""Visible text plus image refs. Ported from ltx2.5-research-agent scrape."""

from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup

from master_agent.scrape.hybrid import ScrapeResult


def extract_html(html: str, base_url: str = "") -> ScrapeResult:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    headings = [
        node.get_text(" ", strip=True)
        for node in soup.find_all(["h1", "h2", "h3"])
        if node.get_text(strip=True)
    ][:40]
    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(article.get_text(" ", strip=True).split())
    images: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(src: str, alt: str = "") -> None:
        if not src or src.startswith("data:"):
            return
        resolved = urljoin(base_url, src)
        if resolved in seen:
            return
        seen.add(resolved)
        images.append({"url": resolved, "alt": alt})

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        srcset = img.get("srcset") or ""
        alt = str(img.get("alt") or "")
        if src:
            _add(str(src), alt)
        if srcset:
            first = str(srcset).split(",")[0].strip().split()[0]
            _add(first, alt)
        if len(images) >= 12:
            break
    if len(images) < 12:
        og = soup.find("meta", attrs={"property": "og:image"}) or soup.find(
            "meta", attrs={"name": "og:image"}
        )
        if og and og.get("content"):
            _add(str(og.get("content")), "og:image")
    return ScrapeResult(
        url=base_url,
        title=title,
        text=text[:20_000],
        headings=headings,
        image_urls=[row["url"] for row in images],
        images=images,
        bytes_read=len(html.encode("utf-8", errors="ignore")),
    )
