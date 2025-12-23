from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


SPOTIFY_ACCOUNTS_BASE = "https://accounts.spotify.com"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


class SpotifyAuthError(RuntimeError):
    pass


def _b64_basic(user: str, password: str) -> str:
    token = f"{user}:{password}".encode("utf-8")
    return base64.b64encode(token).decode("ascii")


def _form_body(data: dict[str, Any]) -> bytes:
    return urllib.parse.urlencode({k: v for k, v in data.items() if v is not None}).encode("utf-8")


def _http_json(req: urllib.request.Request, *, timeout: int = 15) -> dict[str, Any]:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        try:
            return json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON from Spotify: {e}. Raw={raw[:2000]}")


def spotify_exchange_code(*, code: str, code_verifier: Optional[str]) -> dict[str, Any]:
    """
    Exchange authorization code for access token.
    Supports PKCE (recommended) by including code_verifier.
    """
    client_id = (os.getenv("SPOTIFY_CLIENT_ID") or "").strip()
    if not client_id:
        raise SpotifyAuthError("Missing SPOTIFY_CLIENT_ID on backend.")

    redirect_uri = (os.getenv("SPOTIFY_REDIRECT_URI") or "").strip()
    if not redirect_uri:
        raise SpotifyAuthError("Missing SPOTIFY_REDIRECT_URI on backend.")

    client_secret = (os.getenv("SPOTIFY_CLIENT_SECRET") or "").strip()

    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    if client_secret:
        headers["Authorization"] = f"Basic {_b64_basic(client_id, client_secret)}"

    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        # PKCE public client exchange (no secret) needs client_id + code_verifier
        "client_id": None if client_secret else client_id,
        "code_verifier": code_verifier,
    }

    req = urllib.request.Request(
        f"{SPOTIFY_ACCOUNTS_BASE}/api/token",
        method="POST",
        headers=headers,
        data=_form_body(body),
    )
    out = _http_json(req)
    if "access_token" not in out:
        raise SpotifyAuthError(f"Spotify token exchange failed: {out}")
    return out


def spotify_refresh_token(*, refresh_token: str) -> dict[str, Any]:
    client_id = (os.getenv("SPOTIFY_CLIENT_ID") or "").strip()
    if not client_id:
        raise SpotifyAuthError("Missing SPOTIFY_CLIENT_ID on backend.")

    client_secret = (os.getenv("SPOTIFY_CLIENT_SECRET") or "").strip()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    if client_secret:
        headers["Authorization"] = f"Basic {_b64_basic(client_id, client_secret)}"

    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": None if client_secret else client_id,
    }

    req = urllib.request.Request(
        f"{SPOTIFY_ACCOUNTS_BASE}/api/token",
        method="POST",
        headers=headers,
        data=_form_body(body),
    )
    out = _http_json(req)
    if "access_token" not in out:
        raise SpotifyAuthError(f"Spotify token refresh failed: {out}")
    return out


def spotify_api_get(*, access_token: str, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = f"{SPOTIFY_API_BASE}{path}"
    if params:
        url = url + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
    )
    return _http_json(req)


def compute_top_genres_from_artists(items: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    freq: dict[str, int] = {}
    for a in items or []:
        for g in (a.get("genres") or []):
            gs = str(g).strip()
            if not gs:
                continue
            freq[gs] = freq.get(gs, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [g for g, _ in ranked[: max(0, int(limit))]]


def expires_at_from_now(expires_in: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=max(0, int(expires_in) - 30))
    # keep as ISO8601 Z
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


