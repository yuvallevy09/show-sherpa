from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class Location(BaseModel):
    city: str = ""
    country: str = ""
    state: str = ""


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
    spotify_connected=True,
    spotify_profile=SpotifyProfile(
        display_name="Music Lover",
        top_artists=["Tame Impala", "The Midnight", "CHVRCHES", "Japanese Breakfast"],
        top_genres=["indie rock", "synthwave", "indie pop", "psychedelic rock"],
    ),
    location=Location(city="New York", country="US", state="NY"),
)


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


