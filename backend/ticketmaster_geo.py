from __future__ import annotations

"""
Ticketmaster Discovery API prefers geo searches via `geoPoint` (a geohash).
This module provides a tiny geohash encoder (no extra dependency).
"""

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, *, precision: int = 9) -> str:
    """
    Encode lat/lon into a geohash string.
    Precision 9 ~ a few meters to tens of meters; suitable for "near me" searches.
    """
    # Clamp lat/lon to valid ranges
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))

    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0

    bits = []
    even = True
    while len(bits) < precision * 5:
        if even:
            mid = (lon_min + lon_max) / 2.0
            if lon >= mid:
                bits.append(1)
                lon_min = mid
            else:
                bits.append(0)
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2.0
            if lat >= mid:
                bits.append(1)
                lat_min = mid
            else:
                bits.append(0)
                lat_max = mid
        even = not even

    out = []
    for i in range(0, len(bits), 5):
        chunk = bits[i : i + 5]
        val = 0
        for b in chunk:
            val = (val << 1) | b
        out.append(_BASE32[val])
    return "".join(out)[:precision]


