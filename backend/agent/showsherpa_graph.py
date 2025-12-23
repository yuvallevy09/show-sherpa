from __future__ import annotations

import json
import os
import operator
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Literal

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage,
)
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pick_best_image(images: list[dict[str, Any]] | None) -> str:
    if not images:
        return ""
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
        "name": str(ev.get("name") or ""),
        "artist": str(artist_name or ""),
        "venue": str(venue_name or ""),
        "date": str(local_date or ""),
        "time": str(local_time or ""),
        "price": _format_price(ev.get("priceRanges")),
        "genre": str(genre or ""),
        "image": _pick_best_image(ev.get("images")),
        "ticketUrl": str(ev.get("url") or ""),
        "id": str(ev.get("id") or ""),
    }


def _ticketmaster_search_raw(
    *,
    keyword: Optional[str],
    city: str,
    state_code: str,
    country_code: str,
    start_date_time: str,
    end_date_time: str,
    classification_name: str = "music",
    size: int = 10,
    sort: str = "date,asc",
) -> dict[str, Any]:
    api_key = os.getenv("TICKETMASTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing TICKETMASTER_API_KEY on backend.")

    params: dict[str, Any] = {
        "apikey": api_key,
        "classificationName": classification_name,
        "city": city,
        "stateCode": state_code,
        "countryCode": country_code,
        "startDateTime": start_date_time,
        "endDateTime": end_date_time,
        "size": max(1, min(int(size), 50)),
        "sort": sort,
    }
    if keyword:
        params["keyword"] = keyword

    url = "https://app.ticketmaster.com/discovery/v2/events.json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


@tool
def search_ticketmaster_events(
    city: str,
    stateCode: str,
    countryCode: str,
    keyword: str | None = None,
    days: int = 30,
    size: int = 10,
) -> str:
    """
    Search real, bookable music events from Ticketmaster Discovery API.

    Args:
      city: City (e.g. "New York")
      stateCode: State/region code (e.g. "NY")
      countryCode: Country code (e.g. "US")
      keyword: Optional keyword/artist to filter by
      days: Lookahead window in days (default 30)
      size: Max results (default 10)

    Returns:
      JSON string with shape: { "events": [ ...normalized events... ] }
    """
    now = datetime.now(timezone.utc)
    start = _iso_utc(now)
    end = _iso_utc(now + timedelta(days=max(1, min(int(days), 180))))

    data = _ticketmaster_search_raw(
        keyword=keyword,
        city=city,
        state_code=stateCode,
        country_code=countryCode,
        start_date_time=start,
        end_date_time=end,
        size=size,
    )
    events_raw = ((data.get("_embedded") or {}).get("events")) or []
    events = [_normalize_event(ev) for ev in events_raw]
    return json.dumps({"events": events})


def _get_llm():
    """
    Return an LLM client.

    Priority:
    1) Groq (if GROQ_API_KEY is set)
    2) Google AI Studio / Gemini (if GOOGLE_API_KEY is set)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)

    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    raise RuntimeError("Missing GROQ_API_KEY and GOOGLE_API_KEY on backend.")


class SherpaState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    events: list[dict[str, Any]]
    used_tool: bool


def _system_profile(profile: dict[str, Any]) -> str:
    loc = profile.get("location") or {}
    spotify = profile.get("spotify_profile") or {}
    custom_genres = profile.get("custom_genres") or []
    custom_artists = profile.get("custom_artists") or []
    return (
        "You are ShowSherpa, a live-music concierge.\n"
        "Critical rules:\n"
        "- Never invent events. Only discuss concerts returned by the tool.\n"
        "- If no results, say you found none and suggest broadening the search.\n"
        "- When unsure, ask a clarifying question.\n\n"
        "User profile (may be partial):\n"
        f"- location: city={loc.get('city','')}, state={loc.get('state','')}, country={loc.get('country','')}\n"
        f"- spotify_top_artists: {spotify.get('top_artists', [])}\n"
        f"- spotify_top_genres: {spotify.get('top_genres', [])}\n"
        f"- custom_genres: {custom_genres}\n"
        f"- custom_artists: {custom_artists}\n"
    )


def _render_event_line(ev: dict[str, Any]) -> str:
    name = str(ev.get("name") or "").strip()
    artist = str(ev.get("artist") or "").strip()
    venue = str(ev.get("venue") or "").strip()
    date = str(ev.get("date") or "").strip()
    time = str(ev.get("time") or "").strip()
    price = str(ev.get("price") or "").strip()
    url = str(ev.get("ticketUrl") or "").strip()
    bits = [b for b in [artist, venue, (date + (f" {time}" if time else "")).strip(), price] if b]
    meta = " — ".join(bits)
    if url:
        return f"- **{name}**{(' — ' + meta) if meta else ''}\n  {url}"
    return f"- **{name}**{(' — ' + meta) if meta else ''}"


def _render_events(events: list[dict[str, Any]], *, limit: int = 5) -> str:
    shown = (events or [])[: max(0, int(limit))]
    return "\n".join(_render_event_line(ev) for ev in shown if (ev.get("name") or "").strip())


def _coerce_history_messages(history: list[dict[str, Any]] | None) -> list[AnyMessage]:
    out: list[AnyMessage] = []
    for m in history or []:
        role = str(m.get("role") or "").lower().strip()
        content = str(m.get("content") or "")
        if not content:
            continue
        if role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _plan_node_factory(profile: dict[str, Any]):
    llm = _get_llm()
    tools = [search_ticketmaster_events]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def plan_node(state: SherpaState) -> dict[str, Any]:
        sys = SystemMessage(
            content=_system_profile(profile)
            + "\nDecide whether to call the tool.\n"
            "If the user asks for concerts, you SHOULD call the tool unless location is missing.\n"
            "If location is missing, ask a clarifying question for city/state/country.\n"
        )
        msg = llm_with_tools.invoke([sys] + state["messages"])
        return {"messages": [msg]}

    def tool_node(state: SherpaState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": [], "events": [], "used_tool": False}
        out_messages: list[AnyMessage] = []
        collected: list[dict[str, Any]] = []
        for tc in last.tool_calls:
            tool_obj = tools_by_name.get(tc["name"])
            if not tool_obj:
                continue
            observation = tool_obj.invoke(tc["args"])
            out_messages.append(ToolMessage(content=observation, tool_call_id=tc["id"]))
            try:
                parsed = json.loads(observation or "{}")
                evs = parsed.get("events") or []
                if isinstance(evs, list):
                    collected.extend(evs)
            except Exception:
                pass
        return {"messages": out_messages, "events": collected, "used_tool": True}

    def respond_node(state: SherpaState) -> dict[str, Any]:
        # If we didn't call any tool, keep the planner's response (often a clarifying question).
        if not state.get("used_tool"):
            return {"messages": []}

        # Use the tool results (events) as the only “facts”.
        events = state.get("events") or []
        if not events:
            return {
                "messages": [
                    AIMessage(
                        content="I searched Ticketmaster but didn’t find any matching music events in your area for the selected window. Try a nearby city, broaden the date range, or add a keyword/artist."
                    )
                ]
            }

        # Strong hallucination control: ask the LLM to pick event IDs + reasons, then we render facts ourselves.
        compact_events = [
            {
                "id": ev.get("id") or "",
                "name": ev.get("name") or "",
                "artist": ev.get("artist") or "",
                "venue": ev.get("venue") or "",
                "date": ev.get("date") or "",
                "time": ev.get("time") or "",
                "price": ev.get("price") or "",
                "genre": ev.get("genre") or "",
                "ticketUrl": ev.get("ticketUrl") or "",
            }
            for ev in events
        ]

        sys = SystemMessage(
            content=_system_profile(profile)
            + "\nYou will be given Ticketmaster events as JSON.\n"
            "Return ONLY valid JSON with this shape:\n"
            '{ "intro": string, "picks": [ { "id": string, "why": string } ], "question": string|null }\n'
            "- Picks must use ONLY IDs that appear in the provided events.\n"
            "- Do not include event facts (date/venue/price) in the JSON; we will render those ourselves.\n"
            "- Keep intro concise.\n"
        )
        user = HumanMessage(content=json.dumps({"events": compact_events}, ensure_ascii=False))
        draft = llm.invoke([sys, user]).content or ""

        picks: list[dict[str, str]] = []
        intro = "Here are a few real, bookable concerts I found:"
        question: str | None = None
        try:
            parsed = json.loads(draft)
            intro = str(parsed.get("intro") or intro)
            question_val = parsed.get("question")
            question = None if question_val in (None, "", "null") else str(question_val)
            raw_picks = parsed.get("picks") or []
            if isinstance(raw_picks, list):
                for p in raw_picks[:3]:
                    if not isinstance(p, dict):
                        continue
                    pid = str(p.get("id") or "").strip()
                    why = str(p.get("why") or "").strip()
                    if pid:
                        picks.append({"id": pid, "why": why})
        except Exception:
            # If parsing fails, fall back to deterministic rendering.
            picks = []

        by_id = {str(ev.get("id") or ""): ev for ev in events}
        chosen = [by_id.get(p["id"]) for p in picks if by_id.get(p["id"])]
        chosen = [c for c in chosen if c]
        if not chosen:
            # deterministic fallback: show first 5
            content = intro + "\n\n" + _render_events(events, limit=5)
            return {"messages": [AIMessage(content=content)]}

        lines = []
        for p in picks:
            ev = by_id.get(p["id"])
            if not ev:
                continue
            line = _render_event_line(ev)
            why = p.get("why") or ""
            if why:
                line = line + f"\n  _Why_: {why}"
            lines.append(line)

        content = intro.strip() + "\n\n" + "\n".join(lines)
        if question:
            content += "\n\n" + str(question).strip()
        return {"messages": [AIMessage(content=content)]}

    def should_continue(state: SherpaState) -> Literal["tool_node", "respond_node"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_node"
        return "respond_node"

    builder = StateGraph(SherpaState)
    builder.add_node("plan_node", plan_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("respond_node", respond_node)
    builder.add_edge(START, "plan_node")
    builder.add_conditional_edges("plan_node", should_continue, ["tool_node", "respond_node"])
    builder.add_edge("tool_node", "respond_node")
    builder.add_edge("respond_node", END)
    graph = builder.compile()

    return graph


def run_agent(message: str, profile: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Execute the ShowSherpa LangGraph agent.
    Returns: { "content": str, "events": [...] }
    """
    graph = _plan_node_factory(profile)
    prior = _coerce_history_messages(history)
    state = graph.invoke({"messages": prior + [HumanMessage(content=message)], "events": [], "used_tool": False})
    messages = state.get("messages") or []
    last_text = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_text = m.content or ""
            break
    return {"content": last_text, "events": state.get("events") or []}


