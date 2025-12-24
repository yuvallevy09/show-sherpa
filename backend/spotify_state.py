from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from spotify_client import expires_at_from_now, spotify_refresh_token

"""
Shared Spotify token state for the backend and agent graph.

This module exists to avoid circular imports:
- `backend/main.py` imports the agent graph (`agent/showsherpa_graph.py`)
- The agent graph needs access to Spotify access tokens to fetch stats

So tokens and refresh logic live here (no dependency on `main.py`).
"""


# In-memory Spotify token storage (incremental step). Do NOT expose to frontend.
SPOTIFY_TOKENS: dict[str, str] = {}


def spotify_is_connected() -> bool:
    return bool(SPOTIFY_TOKENS.get("access_token") and SPOTIFY_TOKENS.get("refresh_token"))


def spotify_token_expired() -> bool:
    exp = SPOTIFY_TOKENS.get("expires_at") or ""
    if not exp:
        return True
    try:
        # expires_at stored as Z
        dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= dt
    except Exception:
        return True


def spotify_get_access_token() -> str:
    if not spotify_is_connected():
        raise RuntimeError("Spotify is not connected.")
    if spotify_token_expired():
        refreshed = spotify_refresh_token(refresh_token=SPOTIFY_TOKENS["refresh_token"])
        SPOTIFY_TOKENS["access_token"] = refreshed["access_token"]
        SPOTIFY_TOKENS["expires_at"] = expires_at_from_now(int(refreshed.get("expires_in") or 3600))
        if refreshed.get("refresh_token"):
            SPOTIFY_TOKENS["refresh_token"] = refreshed["refresh_token"]
    return SPOTIFY_TOKENS["access_token"]


def spotify_set_tokens(*, access_token: str, refresh_token: str | None, expires_in: int | None) -> None:
    SPOTIFY_TOKENS["access_token"] = str(access_token or "")
    if refresh_token:
        SPOTIFY_TOKENS["refresh_token"] = str(refresh_token)
    SPOTIFY_TOKENS["expires_at"] = expires_at_from_now(int(expires_in or 3600))


def spotify_clear_tokens() -> None:
    SPOTIFY_TOKENS.clear()


