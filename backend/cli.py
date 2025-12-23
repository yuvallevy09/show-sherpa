from __future__ import annotations

import json
import sys

from agent.showsherpa_graph import run_agent
from main import _USER  # reuse the same in-memory profile as the API server


def _print_events(events: list[dict], limit: int = 5) -> None:
    if not events:
        return
    print("\nEvents (raw, from Ticketmaster tool):")
    for ev in events[:limit]:
        name = ev.get("name") or ""
        venue = ev.get("venue") or ""
        date = ev.get("date") or ""
        url = ev.get("ticketUrl") or ""
        print(f"- {name} @ {venue} ({date}) {url}".strip())


def main() -> int:
    print("ShowSherpa CLI (type 'exit' to quit)")
    history: list[dict[str, str]] = []

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0

        out = run_agent(user, _USER.model_dump(), history=history)
        content = out.get("content") or ""
        events = out.get("events") or []

        print("\nSherpa:\n" + content)
        _print_events(events)

        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    raise SystemExit(main())


