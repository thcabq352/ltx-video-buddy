"""robots.txt allow check with in-process per-host cache."""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "VideoBuddyBot/1.0"

_CACHE: dict[str, RobotFileParser] = {}


def robots_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    host = f"{parsed.scheme}://{parsed.netloc}"
    parser = _CACHE.get(host)
    if parser is None:
        parser = RobotFileParser()
        robots_url = f"{host}/robots.txt"
        try:
            parser.set_url(robots_url)
            parser.read()
        except Exception:
            return True
        _CACHE[host] = parser
    try:
        return parser.can_fetch(user_agent, url)
    except Exception:
        return True
