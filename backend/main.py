from __future__ import annotations

import os
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi import HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.showsherpa_graph import run_agent
from spotify_client import (
    SpotifyAuthError,
    compute_top_genres_from_artists,
    expires_at_from_now,
    spotify_api_get,
    spotify_exchange_code,
    spotify_refresh_token,
)
from spotify_state import (
    SPOTIFY_TOKENS as _SPOTIFY_TOKENS,
    spotify_clear_tokens,
    spotify_get_access_token,
    spotify_is_connected,
    spotify_set_tokens,
)
from ticketmaster_geo import geohash_encode


def _load_env_file(path: str) -> None:
    """
    Minimal dotenv loader (no extra dependency).
    Loads KEY=VALUE lines into os.environ without overriding already-set vars.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if not key:
                    continue
                os.environ.setdefault(key, value)
    except FileNotFoundError:
        return


# Auto-load backend/.env if present (useful for local dev).
_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))


class Location(BaseModel):
    city: str = ""
    country: str = ""
    state: str = ""
    # Optional coordinates (used for geo-radius searches)
    coordinates: Optional[dict[str, float]] = None


class SpotifyProfile(BaseModel):
    display_name: str = "Music Lover"
    top_artists: list[str] = Field(default_factory=list)
    top_genres: list[str] = Field(default_factory=list)


class User(BaseModel):
    id: str = "local-user"
    full_name: str = "ShowSherpa User"
    email: str = "user@example.com"
    spotify_connected: bool = False
    spotify_profile: Optional[SpotifyProfile] = None
    location: Location = Field(default_factory=Location)
    custom_genres: list[str] = Field(default_factory=list)
    custom_artists: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    content: str
    concerts: list[dict[str, Any]] = Field(default_factory=list)


app = FastAPI(title="ShowSherpa API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory user store for the first incremental step.
_USER: User = User(
    spotify_connected=False,
    spotify_profile=None,
    location=Location(city="New York", country="US", state="NY"),
)

def _spotify_get_access_token() -> str:
    try:
        return spotify_get_access_token()
    except RuntimeError:
        raise HTTPException(status_code=401, detail="Spotify is not connected.")


def _spotify_sync_profile() -> SpotifyProfile:
    """
    Pull Spotify user profile + top artists and update the in-memory user store.
    """
    access_token = _spotify_get_access_token()
    me = spotify_api_get(access_token=access_token, path="/me")
    top = spotify_api_get(
        access_token=access_token,
        path="/me/top/artists",
        params={"limit": 20, "time_range": "medium_term"},
    )
    items = top.get("items") or []
    top_artists = [str(a.get("name") or "").strip() for a in items if str(a.get("name") or "").strip()]
    top_genres = compute_top_genres_from_artists(items, limit=8)
    display_name = str(me.get("display_name") or _USER.full_name or "Spotify User")
    profile = SpotifyProfile(display_name=display_name, top_artists=top_artists[:20], top_genres=top_genres)
    return profile


class SpotifyExchangeRequest(BaseModel):
    code: str
    code_verifier: str | None = None


class SpotifySyncResponse(BaseModel):
    spotify_connected: bool
    spotify_profile: Optional[SpotifyProfile] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/me", response_model=User)
def get_me() -> User:
    return _USER


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _deep_merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


@app.patch("/me", response_model=User)
def patch_me(patch: Dict[str, Any]) -> User:
    global _USER
    current = _USER.model_dump()
    merged = _deep_merge_dict(current, patch)
    _USER = User.model_validate(merged)
    return _USER


@app.get("/spotify/status", response_model=SpotifySyncResponse)
def spotify_status() -> SpotifySyncResponse:
    return SpotifySyncResponse(spotify_connected=spotify_is_connected(), spotify_profile=_USER.spotify_profile)


@app.post("/spotify/exchange", response_model=SpotifySyncResponse)
def spotify_exchange(req: SpotifyExchangeRequest) -> SpotifySyncResponse:
    """
    Exchange an OAuth authorization code for tokens, then fetch Spotify profile + top artists.
    PKCE is supported via `code_verifier`.
    """
    global _USER
    try:
        tok = spotify_exchange_code(code=req.code, code_verifier=req.code_verifier)
        spotify_set_tokens(
            access_token=tok["access_token"],
            refresh_token=tok.get("refresh_token"),
            expires_in=int(tok.get("expires_in") or 3600),
        )
        profile = _spotify_sync_profile()
    except SpotifyAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Spotify exchange/sync failed: {e}")

    _USER = User.model_validate({**_USER.model_dump(), "spotify_connected": True, "spotify_profile": profile.model_dump()})
    return SpotifySyncResponse(spotify_connected=True, spotify_profile=_USER.spotify_profile)


@app.post("/spotify/sync", response_model=SpotifySyncResponse)
def spotify_sync() -> SpotifySyncResponse:
    """
    Refresh token if needed, then refresh Spotify profile/top artists into the user store.
    """
    global _USER
    try:
        profile = _spotify_sync_profile()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Spotify sync failed: {e}")

    _USER = User.model_validate({**_USER.model_dump(), "spotify_connected": True, "spotify_profile": profile.model_dump()})
    return SpotifySyncResponse(spotify_connected=True, spotify_profile=_USER.spotify_profile)


@app.post("/spotify/disconnect", response_model=SpotifySyncResponse)
def spotify_disconnect() -> SpotifySyncResponse:
    global _USER
    spotify_clear_tokens()
    _USER = User.model_validate({**_USER.model_dump(), "spotify_connected": False, "spotify_profile": None})
    return SpotifySyncResponse(spotify_connected=False, spotify_profile=None)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pick_best_image(images: list[dict[str, Any]] | None) -> str:
    if not images:
        return ""
    # Prefer 16_9 images when available; otherwise choose the largest by area.
    preferred = [img for img in images if str(img.get("ratio", "")).lower() == "16_9"]
    candidates = preferred or images
    best = max(candidates, key=lambda img: (int(img.get("width") or 0) * int(img.get("height") or 0)))
    return str(best.get("url") or "")


def _format_price(price_ranges: list[dict[str, Any]] | None) -> str:
    if not price_ranges:
        return ""
    pr = price_ranges[0] or {}
    currency = pr.get("currency") or ""
    min_p = pr.get("min")
    max_p = pr.get("max")
    if min_p is None and max_p is None:
        return ""
    # Ticketmaster typically returns USD for US events; format loosely.
    if min_p is not None and max_p is not None and min_p != max_p:
        return f"{min_p:g}–{max_p:g} {currency}".strip()
    v = min_p if min_p is not None else max_p
    if v is None:
        return ""
    return f"From {v:g} {currency}".strip()


def _normalize_event(ev: dict[str, Any]) -> dict[str, Any]:
    dates = (ev.get("dates") or {}).get("start") or {}
    local_date = dates.get("localDate") or ""
    local_time = dates.get("localTime") or ""

    embedded = ev.get("_embedded") or {}
    venues = embedded.get("venues") or []
    venue_name = (venues[0] or {}).get("name") if venues else ""

    attractions = embedded.get("attractions") or []
    artist_name = (attractions[0] or {}).get("name") if attractions else ""

    classifications = ev.get("classifications") or []
    genre = ""
    if classifications:
        c0 = classifications[0] or {}
        genre = ((c0.get("genre") or {}).get("name")) or ((c0.get("segment") or {}).get("name")) or ""

    return {
        # Shape matches what the frontend ConcertCard expects.
        "name": str(ev.get("name") or ""),
        "artist": str(artist_name or ""),
        "venue": str(venue_name or ""),
        "date": str(local_date or ""),
        # Frontend currently expects a display string; keep it as-is.
        "time": str(local_time or ""),
        "price": _format_price(ev.get("priceRanges")),
        "genre": str(genre or ""),
        "image": _pick_best_image(ev.get("images")),
        "ticketUrl": str(ev.get("url") or ""),
        # Keep raw id for follow-ups later (event details lookup).
        "id": str(ev.get("id") or ""),
    }


@app.get("/ticketmaster/events")
def ticketmaster_events(
    keyword: Optional[str] = None,
    city: Optional[str] = None,
    state_code: Optional[str] = Query(default=None, alias="stateCode"),
    country_code: Optional[str] = Query(default=None, alias="countryCode"),
    postal_code: Optional[str] = Query(default=None, alias="postalCode"),
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    geo_point: Optional[str] = Query(default=None, alias="geoPoint"),
    radius: Optional[int] = None,
    unit: str = "miles",
    classification_name: str = Query(default="music", alias="classificationName"),
    start_date_time: Optional[str] = Query(default=None, alias="startDateTime"),
    end_date_time: Optional[str] = Query(default=None, alias="endDateTime"),
    size: int = 10,
    page: int = 0,
    sort: str = "date,asc",
) -> dict[str, Any]:
    """
    Proxy to Ticketmaster Discovery API Event Search and normalize the response.

    Auth is via API key query param (`apikey`) as described in Ticketmaster docs:
    https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
    """
    api_key = os.getenv("TICKETMASTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing TICKETMASTER_API_KEY on backend.")

    # Default time window: next 30 days (UTC) if not provided.
    now = datetime.now(timezone.utc)
    if not start_date_time:
        start_date_time = _iso_utc(now)
    if not end_date_time:
        end_date_time = _iso_utc(now + timedelta(days=30))

    params: dict[str, Any] = {
        "apikey": api_key,
        "classificationName": classification_name,
        "startDateTime": start_date_time,
        "endDateTime": end_date_time,
        "size": max(1, min(int(size), 50)),
        "page": max(0, int(page)),
        "sort": sort,
    }

    if keyword:
        params["keyword"] = keyword
    # Prefer geoPoint (geohash) + radius when coordinates are supplied.
    resolved_geo = (geo_point or "").strip()
    if not resolved_geo and lat is not None and lng is not None:
        try:
            resolved_geo = geohash_encode(float(lat), float(lng), precision=9)
        except Exception:
            resolved_geo = ""

    if resolved_geo:
        params["geoPoint"] = resolved_geo
        if radius is not None:
            params["radius"] = max(1, min(int(radius), 500))
            params["unit"] = "km" if str(unit).lower().strip() in ("km", "kilometers", "kilometres") else "miles"
    elif postal_code:
        params["postalCode"] = postal_code
    elif city:
        params["city"] = city
    if state_code and not resolved_geo:
        params["stateCode"] = state_code
    if country_code and not resolved_geo:
        params["countryCode"] = country_code

    url = "https://app.ticketmaster.com/discovery/v2/events.json?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ticketmaster request failed: {e}")

    events = ((data.get("_embedded") or {}).get("events")) or []
    normalized = [_normalize_event(ev) for ev in events]

    return {
        "events": normalized,
        "page": data.get("page") or {},
        "attribution": "Data provided by Ticketmaster Discovery API",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    LangGraph-powered chat endpoint.

    Uses an LLM (Groq via LangChain) to decide whether to call Ticketmaster tools and how to respond.
    If GROQ_API_KEY is not configured, returns a clear error so the frontend can fall back to
    Ticketmaster-only behavior.
    """
    try:
        out = run_agent(req.message, _USER.model_dump(), history=req.history or [])
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=f"Chat agent is not configured or failed to run: {e}",
        )
    return ChatResponse(content=out.get("content") or "", concerts=out.get("events") or [])


