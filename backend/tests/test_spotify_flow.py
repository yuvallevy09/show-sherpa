from __future__ import annotations

import json
import sys
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class _DummyResp:
    def __init__(self, payload: dict[str, Any]):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:  # urllib response API
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_urlopen_factory():
    """
    Fake Spotify HTTP responses for:
    - POST https://accounts.spotify.com/api/token
    - GET  https://api.spotify.com/v1/me
    - GET  https://api.spotify.com/v1/me/top/artists
    """

    def fake_urlopen(req: urllib.request.Request, timeout: int = 15):
        url = getattr(req, "full_url", None) or req.get_full_url()

        if url.endswith("/api/token"):
            body = (req.data or b"").decode("utf-8")
            parsed = dict(urllib.parse.parse_qsl(body))
            if parsed.get("grant_type") == "authorization_code":
                return _DummyResp(
                    {
                        "access_token": "access_token_1",
                        "refresh_token": "refresh_token_1",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    }
                )
            if parsed.get("grant_type") == "refresh_token":
                return _DummyResp(
                    {
                        "access_token": "access_token_2",
                        "refresh_token": "refresh_token_1",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    }
                )
            return _DummyResp({"error": "unsupported_grant_type"})

        if url.endswith("/v1/me"):
            return _DummyResp({"display_name": "Test Spotify User"})

        if "/v1/me/top/artists" in url:
            return _DummyResp(
                {
                    "items": [
                        {"name": "Radiohead", "genres": ["alternative rock", "art rock"]},
                        {"name": "Tame Impala", "genres": ["psychedelic rock", "indie rock"]},
                        {"name": "CHVRCHES", "genres": ["synthpop", "indie pop"]},
                    ]
                }
            )

        raise RuntimeError(f"Unexpected urlopen URL: {url}")

    return fake_urlopen


@pytest.fixture(autouse=True)
def _reset_in_memory_state(monkeypatch: pytest.MonkeyPatch):
    # Ensure Spotify env vars exist for tests
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client_id_123")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "client_secret_abc")
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5173/spotify/callback")

    # Reset globals between tests
    main._SPOTIFY_TOKENS.clear()
    main._USER = main.User(
        spotify_connected=False,
        spotify_profile=None,
        location=main.Location(city="New York", country="US", state="NY"),
    )


def test_spotify_exchange_sets_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_factory())

    client = TestClient(main.app)
    res = client.post("/spotify/exchange", json={"code": "abc", "code_verifier": "verifier"})
    assert res.status_code == 200
    data = res.json()
    assert data["spotify_connected"] is True
    assert data["spotify_profile"]["display_name"] == "Test Spotify User"
    assert "Radiohead" in data["spotify_profile"]["top_artists"]
    assert "indie rock" in data["spotify_profile"]["top_genres"]

    me = client.get("/me").json()
    assert me["spotify_connected"] is True
    assert me["spotify_profile"]["display_name"] == "Test Spotify User"


def test_spotify_sync_requires_connected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_factory())

    client = TestClient(main.app)
    res = client.post("/spotify/sync")
    assert res.status_code == 401


def test_spotify_disconnect(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_factory())
    client = TestClient(main.app)

    client.post("/spotify/exchange", json={"code": "abc", "code_verifier": "verifier"})
    res = client.post("/spotify/disconnect")
    assert res.status_code == 200
    assert res.json()["spotify_connected"] is False
    me = client.get("/me").json()
    assert me["spotify_connected"] is False
    assert me["spotify_profile"] is None


