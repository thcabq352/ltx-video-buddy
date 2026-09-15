"""Multi-profile Hermes gateway discovery. Ported from SOS/hermes_gateways.py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

ProbeFn = Callable[..., dict[str, Any]]

STUDIO_PORT = 8189
BUDDY_ADAPTER_SOURCE = "buddy-adapter"
PROBE_TIMEOUT_S = 0.4

KNOWN_PORTS = {
    "default": 8642,
    "hardline": 8643,
    "forge": 8644,
    "coder": 8645,
    "cinegen": 8646,
    "research": 8647,
    "ltx": 8642,
}

SOS_MUX_PROFILES = ("hardline", "coder", "research", "ltx")

HERMES_PROFILE_LABELS = {
    "default": "Ringmaster",
    "hardline": "Hardline",
    "forge": "Forge",
    "coder": "Coder",
    "cinegen": "Cinegen",
    "research": "Research",
    "ltx": "Ltx",
}

LTX_RESEARCH_SYSTEM = (
    "You are Ltx, Hermes profile `ltx`, the LTX research seat. "
    "Specialize in LTX Video on the Video Buddy portable ComfyUI. "
    "Stay short. Never print tokens, API keys, or bearers."
)


@dataclass
class Gateway:
    profile: str
    name: str
    port: int
    chat_url: str
    healthy: bool = False
    can_speak: bool = False
    office: str = ""
    source: str = "hermes-gateway"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _port_from_config(data: dict[str, Any], fallback: int) -> int:
    platforms = data.get("platforms")
    if not isinstance(platforms, dict):
        return fallback
    api = platforms.get("api_server")
    if not isinstance(api, dict):
        return fallback
    extra = api.get("extra")
    if not isinstance(extra, dict):
        return fallback
    try:
        return int(extra.get("port") or fallback)
    except (TypeError, ValueError):
        return fallback


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def list_profiles(home: Path | None = None) -> list[str]:
    root = home or hermes_home()
    names = ["default"]
    profiles = root / "profiles"
    if profiles.is_dir():
        for child in sorted(profiles.iterdir()):
            if child.is_dir() and (child / "config.yaml").is_file():
                names.append(child.name)
    return names


def models_url(chat_url: str) -> str:
    trimmed = chat_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed[: -len("/chat/completions")] + "/models"
    return trimmed + "/models"


def probe_gateway(
    gateway: Gateway,
    *,
    request: Optional[ProbeFn] = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> Gateway:
    url = models_url(gateway.chat_url)
    try:
        if request:
            reply = request("GET", url)
            ok = bool(reply.get("ok"))
        else:
            import httpx

            response = httpx.get(url, timeout=timeout)
            ok = response.status_code < 400
        gateway.healthy = ok
        gateway.can_speak = ok
    except Exception:
        gateway.healthy = False
        gateway.can_speak = False
    return gateway


def buddy_adapter_gateway(*, host: str, office: str = "") -> Gateway:
    return Gateway(
        profile="ltx",
        name=HERMES_PROFILE_LABELS.get("ltx", "Ltx"),
        port=STUDIO_PORT,
        chat_url=f"http://{host}:{STUDIO_PORT}/p/ltx/v1/chat/completions",
        office=office,
        source=BUDDY_ADAPTER_SOURCE,
    )


def discover_gateways(
    *,
    home: Path | None = None,
    office: str = "",
    host: str | None = None,
    probe: bool = True,
    request: Optional[ProbeFn] = None,
) -> list[Gateway]:
    root = home or hermes_home()
    host = host or os.environ.get("SOS_HERMES_HOST", "127.0.0.1")
    found: list[Gateway] = []
    for profile in list_profiles(root):
        config_path = root / "config.yaml" if profile == "default" else root / "profiles" / profile / "config.yaml"
        port = _port_from_config(_read_yaml(config_path), KNOWN_PORTS.get(profile, 8642))
        if profile == "research" and port == 8642:
            port = KNOWN_PORTS["research"]
        own_url = f"http://{host}:{port}/v1/chat/completions"
        mux_url = f"http://{host}:8642/p/{profile}/v1/chat/completions"
        chat_url = own_url if profile == "default" else mux_url
        if profile == "forge":
            chat_url = own_url
        found.append(
            Gateway(
                profile=profile,
                name=HERMES_PROFILE_LABELS.get(profile, profile.title()),
                port=port,
                chat_url=chat_url,
                office=office,
                source="hermes-gateway",
            )
        )
    seated = {item.profile for item in found}
    for profile in SOS_MUX_PROFILES:
        if profile in seated:
            continue
        found.append(
            Gateway(
                profile=profile,
                name=HERMES_PROFILE_LABELS.get(profile, profile.title()),
                port=KNOWN_PORTS.get(profile, 8642),
                chat_url=f"http://{host}:8642/v1/chat/completions",
                office=office,
                source="sos-mux",
            )
        )
    if probe:
        for item in found:
            probe_gateway(item, request=request)
    if not any(item.profile == "ltx" and item.healthy for item in found):
        adapter = buddy_adapter_gateway(host=host, office=office)
        if probe:
            probe_gateway(adapter, request=request)
        found.append(adapter)
    return found


def discover_primary_seat(
    *,
    home: Path | None = None,
    office: str = "",
    host: str | None = None,
    probe: bool = True,
    request: Optional[ProbeFn] = None,
) -> Optional[Gateway]:
    rows = discover_gateways(home=home, office=office, host=host, probe=probe, request=request)
    healthy_ltx = [g for g in rows if g.profile == "ltx" and g.healthy]
    if healthy_ltx:
        hermes = [g for g in healthy_ltx if g.source != BUDDY_ADAPTER_SOURCE]
        return (hermes or healthy_ltx)[0]
    adapters = [g for g in rows if g.source == BUDDY_ADAPTER_SOURCE]
    return adapters[0] if adapters else None


def seat_system_prompt(gateway: Gateway) -> str:
    if gateway.profile == "ltx":
        return LTX_RESEARCH_SYSTEM
    return (
        f"You are {gateway.name}, Hermes profile `{gateway.profile}`, "
        "seated with Video Buddy. Speak in first person as yourself."
    )
