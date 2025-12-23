from __future__ import annotations

import json
import sys
from pathlib import Path
import urllib.request
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class _DummyResp:
    def __init__(self, payload: dict[str, Any]):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_ticketmaster_events_normalizes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TICKETMASTER_API_KEY", "tm_key")

    def fake_urlopen(req: urllib.request.Request, timeout: int = 10):
        url = getattr(req, "full_url", None) or req.get_full_url()
        assert "discovery/v2/events.json" in url
        return _DummyResp(
            {
                "_embedded": {
                    "events": [
                        {
                            "id": "E1",
                            "name": "Test Concert",
                            "url": "https://tickets.example/e1",
                            "dates": {"start": {"localDate": "2025-12-31", "localTime": "20:00:00"}},
                            "classifications": [{"segment": {"name": "Music"}, "genre": {"name": "Rock"}}],
                            "images": [{"url": "https://img.example/1.jpg", "width": 640, "height": 360, "ratio": "16_9"}],
                            "_embedded": {
                                "venues": [{"name": "Test Venue"}],
                                "attractions": [{"name": "Test Artist"}],
                            },
                            "priceRanges": [{"min": 25, "max": 50, "currency": "USD"}],
                        }
                    ]
                },
                "page": {"size": 1, "totalElements": 1, "totalPages": 1, "number": 0},
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = TestClient(main.app)
    res = client.get("/ticketmaster/events?city=New%20York&stateCode=NY&countryCode=US&size=1")
    assert res.status_code == 200
    data = res.json()
    assert data["events"][0]["name"] == "Test Concert"
    assert data["events"][0]["artist"] == "Test Artist"
    assert data["events"][0]["venue"] == "Test Venue"
    assert data["events"][0]["date"] == "2025-12-31"
    assert data["events"][0]["price"].startswith("25")


