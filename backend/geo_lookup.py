from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

"""
Best-effort forward geocoding (city/state/country -> lat/lng).

Used to convert explicit location overrides (e.g. "NYC") into coordinates so we can
use Ticketmaster's preferred geoPoint + radius search.

Implementation notes:
- Uses OpenStreetMap Nominatim (no API key), with short timeouts.
- Includes a basic in-memory cache to reduce repeated lookups.
- If it fails for any reason, callers should fall back to city/state/country search.
"""


_CACHE: dict[str, tuple[float, float, float]] = {}
# key -> (expires_at_epoch, lat, lng)


def _cache_get(key: str) -> Optional[Tuple[float, float]]:
    v = _CACHE.get(key)
    if not v:
        return None
    expires_at, lat, lng = v
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return (lat, lng)


def _cache_set(key: str, lat: float, lng: float, *, ttl_seconds: int = 3600) -> None:
    _CACHE[key] = (time.time() + max(30, int(ttl_seconds)), float(lat), float(lng))


def forward_geocode_city(
    *,
    city: str,
    state: str = "",
    country: str = "",
    timeout: int = 3,
) -> Optional[Tuple[float, float]]:
    """
    Return (lat, lng) for a city/state/country, or None if not found / on error.
    """
    c = (city or "").strip()
    if not c:
        return None
    st = (state or "").strip()
    co = (country or "").strip()

    key = f"{c}|{st}|{co}".lower()
    cached = _cache_get(key)
    if cached:
        return cached

    # Nominatim search endpoint (JSON). Use a descriptive User-Agent per usage policy.
    q_parts = [c]
    if st:
        q_parts.append(st)
    if co:
        q_parts.append(co)
    q = ", ".join(q_parts)

    params = {
        "q": q,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    }
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ShowSherpa/0.1 (educational project)",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw or "[]")
    except Exception:
        return None

    if not isinstance(data, list) or not data:
        return None
    hit = data[0] if isinstance(data[0], dict) else {}
    try:
        lat = float(hit.get("lat"))
        lng = float(hit.get("lon"))
    except Exception:
        return None

    _cache_set(key, lat, lng, ttl_seconds=3600)
    return (lat, lng)


