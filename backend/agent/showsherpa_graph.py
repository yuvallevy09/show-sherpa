from __future__ import annotations

import json
import os
import operator
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Literal

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    AnyMessage,
)
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from spotify_client import compute_top_genres_from_artists, spotify_api_get
from spotify_state import spotify_get_access_token, spotify_is_connected

from ticketmaster_geo import geohash_encode
from geo_lookup import forward_geocode_city

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
    geo_point: Optional[str] = None,
    radius: Optional[int] = None,
    unit: str = "miles",
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
        "startDateTime": start_date_time,
        "endDateTime": end_date_time,
        "size": max(1, min(int(size), 50)),
        "sort": sort,
    }
    # Prefer geoPoint when supplied (Ticketmaster recommended).
    if geo_point:
        params["geoPoint"] = geo_point
        if radius is not None:
            params["radius"] = max(1, min(int(radius), 500))
            params["unit"] = "km" if str(unit).lower().strip() in ("km", "kilometers", "kilometres") else "miles"
    else:
        params["city"] = city
        params["stateCode"] = state_code
        params["countryCode"] = country_code
    if keyword:
        params["keyword"] = keyword

    url = "https://app.ticketmaster.com/discovery/v2/events.json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


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
    # Node 0 input (raw user message for this turn). Node 0 appends it to messages.
    user_input: str
    # Node 1 output: short, conservative summaries for context continuity.
    memory: dict[str, Any]
    # Node A output (routing + slots). Not yet used by downstream nodes, but stored for later nodes.
    route: dict[str, Any]
    # Node C output: normalized time window (ISO8601 Z)
    time_window: dict[str, Any]
    # Node D output: fetched spotify stats for this turn (when needed)
    spotify: dict[str, Any]
    # Node E status (Ticketmaster deterministic search)
    ticketmaster_ok: bool
    ticketmaster_error: str
    # Node F output (best-effort constraint filtering)
    filter_notes: str
    filter_stats: dict[str, Any]
    # Node G output (ranking)
    rank_stats: dict[str, Any]
    # Node I output (strict picks + why)
    picks_payload: dict[str, Any]
    # Node X output (failure branch)
    failure: dict[str, Any]


class RouteLocation(BaseModel):
    city: str = ""
    state: str = ""
    country: str = ""


class RouteConstraints(BaseModel):
    # MVP: keep this minimal and expand later.
    after_time: str = ""  # e.g., "20:30"
    weekend_only: bool = False
    genres: list[str] = Field(default_factory=list)


class SpotifyStatsRequest(BaseModel):
    kind: Literal["artists", "tracks", "genres", "profile"] = "artists"
    time_range: Literal["short_term", "medium_term", "long_term"] = "medium_term"
    limit: int = 10
    offset: int = 0


class RouteOutput(BaseModel):
    # Core classification
    intent: Literal["EVENT_SEARCH", "EVENT_DETAILS", "SPOTIFY_STATS", "OTHER"] = "OTHER"
    subjectivity: Literal["OBJECTIVE", "TASTE_RANKING", "MIXED"] = "OBJECTIVE"

    # What to fetch next
    needs_spotify: bool = False
    needs_ticketmaster: bool = False
    needs_clarification: bool = False
    clarifying_question: str = ""

    # Slots
    time_phrase: str = ""  # e.g., "next weekend", "this month", "soon"
    artist_query: str = ""  # e.g., "Eminem"
    location_override: RouteLocation = Field(default_factory=RouteLocation)
    proximity_miles: int = 0  # 0 means unset; later default for "near me" can be applied.
    constraints: RouteConstraints = Field(default_factory=RouteConstraints)
    # For SPOTIFY_STATS intent, specify what to fetch.
    spotify_request: SpotifyStatsRequest = Field(default_factory=SpotifyStatsRequest)


class MemoryOutput(BaseModel):
    # Keep these intentionally short. Node 1 enforces caps via instruction; validate non-empty strings.
    literal_summary: list[str] = Field(default_factory=list)
    intent_summary: list[str] = Field(default_factory=list)


class TimeWindowOutput(BaseModel):
    # Deterministic Node C output stored in state.time_window.
    phrase: str = ""
    type: Literal["RELATIVE", "ABSOLUTE", "OPEN_ENDED", "DEFAULT"] = "DEFAULT"
    startDateTime: str = ""
    endDateTime: str = ""


class PickWhyItem(BaseModel):
    id: str
    why: str = ""


class PicksWhyOutput(BaseModel):
    intro: str = "Here are a few real, bookable concerts I found:"
    picks: list[PickWhyItem] = Field(default_factory=list)
    question: Optional[str] = None


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

    def ingest_node(state: SherpaState) -> dict[str, Any]:
        """
        Node 0: ingest user input into the message list.
        - Appends a HumanMessage from state.user_input
        - Keeps only the last N messages (simple cap to avoid prompt blowup)
        """
        user_text = str(state.get("user_input") or "").strip()
        if not user_text:
            # No-op; downstream will handle empty content if needed.
            return {"messages": []}
        max_msgs = 20  # last 10 "exchanges" ~= 20 messages
        # 'messages' is Annotated with operator.add, so we return a list to append.
        appended = [HumanMessage(content=user_text)]
        # Apply trimming deterministically by emitting a special marker in state? In MVP, we trim here by
        # returning no-op and relying on run_agent to cap history. We'll still do a best-effort trim by
        # rewriting the list via a state update (non-additive) in a separate key.
        #
        # LangGraph's operator.add aggregation makes it awkward to overwrite messages in a node update,
        # so we keep run_agent history bounded and just append here.
        return {"messages": appended}

    def _clean_bullets(items: list[str], *, limit: int) -> list[str]:
        out: list[str] = []
        for it in items or []:
            s = str(it or "").strip()
            if not s:
                continue
            # Avoid overly long bullets.
            if len(s) > 220:
                s = s[:217].rstrip() + "..."
            out.append(s)
            if len(out) >= max(0, int(limit)):
                break
        return out

    def memory_node(state: SherpaState) -> dict[str, Any]:
        """
        Node 1: maintain short, conservative memory summaries.

        Output:
          state.memory = { literal_summary: [...], intent_summary: [...] }

        Design principles:
        - Only include facts the user explicitly stated or clearly asked for.
        - No assumptions/inferences (avoid "user likes X" unless they said it).
        - Keep it short (caps).
        - One retry on schema failure, then safe fallback (keep previous memory).
        """
        prev_mem = state.get("memory") or {}

        # Provide a very small window of context; we already keep messages bounded.
        ctx = (state.get("messages") or [])[-12:]

        base_sys = SystemMessage(
            content=(
                "You are maintaining short conversation memory for a music concierge app.\n"
                "Return ONLY structured data matching the schema.\n"
                "Rules (must follow):\n"
                "- Be conservative: include ONLY what the user explicitly said or asked for.\n"
                "- Do NOT infer preferences or add new facts.\n"
                "- Keep it short: literal_summary max 5 bullets; intent_summary max 3 bullets.\n"
                "- Bullets should be simple phrases, no markdown.\n"
            )
        )

        def _call_memory(sys_msg: SystemMessage) -> MemoryOutput:
            mem_llm = llm.with_structured_output(MemoryOutput)
            # Provide previous memory as context in a compact way (not as facts; just to preserve continuity).
            prior = HumanMessage(
                content=json.dumps(
                    {
                        "previous_memory": {
                            "literal_summary": prev_mem.get("literal_summary", []),
                            "intent_summary": prev_mem.get("intent_summary", []),
                        }
                    },
                    ensure_ascii=False,
                )
            )
            return mem_llm.invoke([sys_msg, prior] + ctx)

        try:
            mem = _call_memory(base_sys)
        except Exception:
            try:
                retry_sys = SystemMessage(
                    content=base_sys.content
                    + "\nCRITICAL: Output MUST match schema exactly. No extra keys. No prose."
                )
                mem = _call_memory(retry_sys)
            except Exception:
                # Safe fallback: keep previous memory as-is (or empty).
                lit = _clean_bullets(list(prev_mem.get("literal_summary") or []), limit=5)
                intent = _clean_bullets(list(prev_mem.get("intent_summary") or []), limit=3)
                return {"memory": {"literal_summary": lit, "intent_summary": intent}}

        lit = _clean_bullets(mem.literal_summary, limit=5)
        intent = _clean_bullets(mem.intent_summary, limit=3)
        return {"memory": {"literal_summary": lit, "intent_summary": intent}}

    def _heuristic_route(user_text: str, *, profile: dict[str, Any]) -> RouteOutput:
        """
        Deterministic fallback if structured routing fails.
        Keep this intentionally conservative to avoid bad tool calls.
        """
        text = (user_text or "").strip()
        lower = text.lower()

        # intent
        if any(k in lower for k in ["most listened", "top artists", "top tracks", "favorite artists", "favorite genres"]):
            intent: Literal["EVENT_SEARCH", "EVENT_DETAILS", "SPOTIFY_STATS", "OTHER"] = "SPOTIFY_STATS"
        elif any(k in lower for k in ["how much", "price", "cost", "tickets cost"]):
            intent = "EVENT_DETAILS"
        elif any(k in lower for k in ["concert", "concerts", "show", "shows", "gig", "gigs", "live", "events", "perform"]):
            intent = "EVENT_SEARCH"
        else:
            intent = "OTHER"

        # subjectivity
        subj = "OBJECTIVE"
        if any(k in lower for k in ["good shows", "good concerts", "best", "recommend", "for me", "match my taste"]):
            subj = "TASTE_RANKING"

        # needs flags
        needs_ticketmaster = intent in ("EVENT_SEARCH", "EVENT_DETAILS")
        needs_spotify = intent == "SPOTIFY_STATS" or subj != "OBJECTIVE"

        # spotify request defaults for stats
        spotify_kind: Literal["artists", "tracks", "genres", "profile"] = "artists"
        if "track" in lower:
            spotify_kind = "tracks"
        elif "genre" in lower:
            spotify_kind = "genres"
        elif "me" in lower and ("profile" in lower or "who am i" in lower):
            spotify_kind = "profile"

        time_range: Literal["short_term", "medium_term", "long_term"] = "medium_term"
        if "past year" in lower or "last year" in lower or "long term" in lower:
            time_range = "long_term"
        elif "recent" in lower or "past month" in lower or "this month" in lower or "short term" in lower:
            time_range = "short_term"

        limit = 10
        if spotify_kind == "artists" and time_range == "short_term":
            limit = 6
        if spotify_kind == "artists" and time_range == "long_term":
            limit = 10

        # slots (minimal)
        time_phrase = ""
        for phrase in ["next weekend", "this weekend", "this month", "next month", "soon", "tonight"]:
            if phrase in lower:
                time_phrase = phrase
                break

        # Very naive artist extraction: "is X performing", "taylor swift concert", etc.
        artist_query = ""
        m = re.search(r"\bis\s+(.+?)\s+perform", lower)
        if m:
            artist_query = text[m.start(1) : m.end(1)].strip()
        if not artist_query:
            m2 = re.search(r"\b(?:cost|price)\s+to\s+go\s+to\s+(?:a|an)\s+(.+?)\s+concert", lower)
            if m2:
                artist_query = text[m2.start(1) : m2.end(1)].strip()

        # Default “near me” radius intent if user uses near-me phrasing.
        proximity_miles = 0
        if "near me" in lower and needs_ticketmaster:
            proximity_miles = 100

        # If Ticketmaster needed and profile has no location at all, request clarification.
        loc = (profile.get("location") or {}) if isinstance(profile, dict) else {}
        has_default_loc = bool((loc.get("city") or "").strip() and (loc.get("state") or "").strip() and (loc.get("country") or "").strip())
        needs_clarification = bool(needs_ticketmaster and not has_default_loc)
        clarifying_question = ""
        if needs_clarification:
            clarifying_question = "What city/state/country should I search in?"

        return RouteOutput(
            intent=intent,
            subjectivity=subj,  # type: ignore[arg-type]
            needs_spotify=bool(needs_spotify),
            needs_ticketmaster=bool(needs_ticketmaster),
            needs_clarification=bool(needs_clarification),
            clarifying_question=clarifying_question,
            time_phrase=time_phrase,
            artist_query=artist_query,
            proximity_miles=int(proximity_miles),
            spotify_request=SpotifyStatsRequest(kind=spotify_kind, time_range=time_range, limit=limit, offset=0),
        )

    def route_and_extract_node(state: SherpaState) -> dict[str, Any]:
        """
        Node A: strict router + slot extraction.
        Stores result in state.route (dict) for later nodes.
        Downstream nodes are not yet using this; we keep existing behavior intact.
        """
        # Identify the latest user message text.
        last_user = ""
        for m in reversed(state.get("messages") or []):
            if isinstance(m, HumanMessage):
                last_user = (m.content or "").strip()
                break

        # Default route if empty input.
        if not last_user:
            out = RouteOutput(
                intent="OTHER",
                subjectivity="OBJECTIVE",
                needs_spotify=False,
                needs_ticketmaster=False,
                needs_clarification=True,
                clarifying_question="What can I help you with? You can ask about concerts near you or your Spotify listening stats.",
            )
            return {"route": out.model_dump()}

        # Build strict structured-output router.
        base_sys = SystemMessage(
            content=_system_profile(profile)
            + "\nYou are a router for ShowSherpa.\n"
            + "Task: classify the user's latest request and extract slots.\n"
            + "Return ONLY structured data that matches the provided schema; do NOT include prose.\n"
            + "Guidelines:\n"
            + "- intent=SPOTIFY_STATS for questions about the user's top artists/tracks/genres/time ranges.\n"
            + "- intent=EVENT_DETAILS for price/cost/details about a specific artist/event.\n"
            + "- intent=EVENT_SEARCH for finding upcoming shows/concerts.\n"
            + "- subjectivity=TASTE_RANKING when user asks 'good/best/recommend' or 'match my taste'. Otherwise OBJECTIVE.\n"
            + "- needs_ticketmaster=true for EVENT_SEARCH/EVENT_DETAILS.\n"
            + "- needs_spotify=true for SPOTIFY_STATS or when subjectivity!=OBJECTIVE.\n"
            + "- If location is required and missing from profile AND not provided by user, set needs_clarification=true and provide a short clarifying_question.\n"
            + "- time_phrase: capture relative phrase if present (e.g., 'next weekend', 'this month', 'soon'); otherwise empty.\n"
            + "- artist_query: capture explicit artist if present; otherwise empty.\n"
            + "- spotify_request: for SPOTIFY_STATS, set kind to one of artists|tracks|genres and time_range to short_term|medium_term|long_term.\n"
            + "  Use: short_term for 'recent/past month', medium_term for 'favorite', long_term for 'past year'.\n"
        )

        def _call_router(sys_msg: SystemMessage) -> RouteOutput:
            # Use LangChain structured output API; if the model doesn't support strict schemas,
            # this will raise and we'll fall back.
            router_llm = llm.with_structured_output(RouteOutput)
            # Provide last few messages for context, but keep it small.
            ctx = (state.get("messages") or [])[-10:]
            return router_llm.invoke([sys_msg] + ctx)

        try:
            routed = _call_router(base_sys)
        except Exception:
            # One retry with even stricter instructions.
            try:
                retry_sys = SystemMessage(
                    content=base_sys.content
                    + "\nCRITICAL: Output MUST be valid and complete per schema. No markdown. No extra keys."
                )
                routed = _call_router(retry_sys)
            except Exception:
                routed = _heuristic_route(last_user, profile=profile)

        return {"route": routed.model_dump()}

    def _has_location(loc: dict[str, Any] | None) -> bool:
        if not isinstance(loc, dict):
            return False
        return bool((loc.get("city") or "").strip() and (loc.get("state") or "").strip() and (loc.get("country") or "").strip())

    def clarify_node(state: SherpaState) -> dict[str, Any]:
        """
        Node B: deterministic clarification step.
        Asks exactly one clarifying question and ends the turn.
        """
        route = state.get("route") or {}
        q = str(route.get("clarifying_question") or "").strip()
        if not q:
            q = "What city/state/country should I search in?"
        return {"messages": [AIMessage(content=q)]}

    def _start_of_day_utc(dt: datetime) -> datetime:
        x = dt.astimezone(timezone.utc)
        return x.replace(hour=0, minute=0, second=0, microsecond=0)

    def _end_of_day_utc(dt: datetime) -> datetime:
        x = dt.astimezone(timezone.utc)
        return x.replace(hour=23, minute=59, second=59, microsecond=0)

    def _end_of_month_utc(dt: datetime) -> datetime:
        # Jump to first of next month then subtract one second.
        x = dt.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if x.month == 12:
            nxt = x.replace(year=x.year + 1, month=1)
        else:
            nxt = x.replace(month=x.month + 1)
        return nxt - timedelta(seconds=1)

    def _next_weekend_window(now: datetime) -> tuple[datetime, datetime]:
        """
        Policy: next weekend = Fri 18:00 UTC through Sun 23:59:59 UTC.
        (We use UTC as a safe default; later we can use user's timezone.)
        """
        base = now.astimezone(timezone.utc).replace(microsecond=0)
        # weekday: Mon=0..Sun=6
        wd = base.weekday()
        # days until next Friday (4). If today is Fri/Sat/Sun, take the following week's Friday.
        delta = (4 - wd) % 7
        if delta == 0 and wd >= 4:
            delta = 7
        friday = (base + timedelta(days=delta)).replace(hour=18, minute=0, second=0, microsecond=0)
        sunday = friday + timedelta(days=2)
        end = sunday.replace(hour=23, minute=59, second=59, microsecond=0)
        return friday, end

    def normalize_time_node(state: SherpaState) -> dict[str, Any]:
        """
        Node C: deterministic normalization of time phrases into ISO start/end (UTC).
        Stores results in state.time_window for future Ticketmaster calls (pushdown filters).
        """
        route = state.get("route") or {}
        phrase_raw = str(route.get("time_phrase") or "").strip()
        phrase = phrase_raw.lower().strip()
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Defaults: 30 days lookahead
        tw_type: Literal["RELATIVE", "ABSOLUTE", "OPEN_ENDED", "DEFAULT"] = "DEFAULT"
        start = now
        end = now + timedelta(days=30)

        if phrase in ("soon",):
            tw_type = "RELATIVE"
            start, end = now, now + timedelta(days=30)
        elif phrase in ("this month",):
            tw_type = "RELATIVE"
            start, end = now, _end_of_month_utc(now)
        elif phrase in ("next month",):
            tw_type = "RELATIVE"
            first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if first.month == 12:
                first_next = first.replace(year=first.year + 1, month=1)
            else:
                first_next = first.replace(month=first.month + 1)
            # next month window
            start = first_next
            end = _end_of_month_utc(first_next)
        elif phrase in ("this weekend", "next weekend"):
            tw_type = "RELATIVE"
            # If "this weekend" and we're early in week, take upcoming Fri→Sun; if already weekend, take the current.
            if phrase == "this weekend":
                # nearest Friday for this week (could be in the past if already weekend)
                base = now.astimezone(timezone.utc).replace(microsecond=0)
                wd = base.weekday()
                # Friday=4
                delta = 4 - wd
                friday = (base + timedelta(days=delta)).replace(hour=18, minute=0, second=0, microsecond=0)
                if delta < 0:
                    # It's already weekend; use last Friday 18:00
                    friday = (base + timedelta(days=delta)).replace(hour=18, minute=0, second=0, microsecond=0)
                sunday = friday + timedelta(days=2)
                start, end = friday, sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            else:
                start, end = _next_weekend_window(now)
        elif phrase in ("tonight",):
            tw_type = "RELATIVE"
            start = now
            end = _end_of_day_utc(now)

        out = TimeWindowOutput(
            phrase=phrase_raw,
            type=tw_type,
            startDateTime=_iso_utc(start),
            endDateTime=_iso_utc(end),
        )
        return {"time_window": out.model_dump()}

    def spotify_fetch_node(state: SherpaState) -> dict[str, Any]:
        """
        Node D: conditional Spotify fetch.
        - If route.needs_spotify is false: no-op
        - If Spotify is not connected: store spotify error and (for spotify-only intent) respond with a grounded prompt.
        - For SPOTIFY_STATS intent (spotify-only), respond deterministically with stats (no event tools).
        - For taste-based event queries, populate state.spotify with small, bounded taste profile (artists/genres).
        """
        route_raw = state.get("route") or {}
        try:
            route = RouteOutput.model_validate(route_raw)
        except Exception:
            # If route can't be parsed, do nothing here.
            return {"spotify": {}}

        if not route.needs_spotify:
            return {"spotify": {}}

        # Connection check
        if not spotify_is_connected():
            msg = (
                "I can answer Spotify listening questions after you connect Spotify.\n\n"
                "If you're asking about concerts, you can also tell me 1–3 favorite artists and I’ll search Ticketmaster."
            )
            out = {"spotify": {"error": "not_connected"}}
            if route.intent == "SPOTIFY_STATS" and not route.needs_ticketmaster:
                return {**out, "messages": [AIMessage(content=msg)]}
            return out

        # Access token (may refresh)
        try:
            access_token = spotify_get_access_token()
        except Exception as e:
            out = {"spotify": {"error": f"token_error: {e}"}}
            if route.intent == "SPOTIFY_STATS" and not route.needs_ticketmaster:
                return {**out, "messages": [AIMessage(content="Spotify auth failed while refreshing tokens. Please reconnect Spotify and try again.")]}
            return out

        # Hard caps
        limit = max(1, min(int(route.spotify_request.limit or 10), 50))
        offset = max(0, min(int(route.spotify_request.offset or 0), 49))
        time_range = route.spotify_request.time_range

        def _render_artists(items: list[dict[str, Any]], *, n: int) -> str:
            names = [str(a.get("name") or "").strip() for a in items if str(a.get("name") or "").strip()]
            names = names[:n]
            if not names:
                return "I couldn’t find any top artists for that time range."
            lines = "\n".join(f"- {nm}" for nm in names)
            return lines

        def _render_tracks(items: list[dict[str, Any]], *, n: int) -> str:
            lines_out: list[str] = []
            for t in items[:n]:
                name = str(t.get("name") or "").strip()
                artists = t.get("artists") or []
                a0 = str((artists[0] or {}).get("name") or "").strip() if isinstance(artists, list) and artists else ""
                if not name:
                    continue
                lines_out.append(f"- {name}" + (f" — {a0}" if a0 else ""))
            if not lines_out:
                return "I couldn’t find any top tracks for that time range."
            return "\n".join(lines_out)

        # Spotify-only intent: return deterministic stats response.
        if route.intent == "SPOTIFY_STATS" and not route.needs_ticketmaster:
            kind = route.spotify_request.kind
            if kind == "profile":
                me = spotify_api_get(access_token=access_token, path="/me")
                display_name = str(me.get("display_name") or "Spotify User")
                return {"spotify": {"me": me}, "messages": [AIMessage(content=f"You're connected as **{display_name}**.")]}

            if kind in ("artists", "genres"):
                top = spotify_api_get(
                    access_token=access_token,
                    path="/me/top/artists",
                    params={"limit": limit, "offset": offset, "time_range": time_range},
                )
                items = top.get("items") or []
                if kind == "artists":
                    content = "Here are your top artists:\n" + _render_artists(items, n=limit)
                    return {"spotify": {"top_artists": items, "time_range": time_range}, "messages": [AIMessage(content=content)]}
                # genres from artists
                genres = compute_top_genres_from_artists(items, limit=10)
                if not genres:
                    content = "I couldn’t infer top genres from your top artists for that time range."
                else:
                    content = "Here are your top genres (derived from your top artists):\n" + "\n".join(f"- {g}" for g in genres)
                return {"spotify": {"top_artists": items, "top_genres": genres, "time_range": time_range}, "messages": [AIMessage(content=content)]}

            if kind == "tracks":
                top = spotify_api_get(
                    access_token=access_token,
                    path="/me/top/tracks",
                    params={"limit": limit, "offset": offset, "time_range": time_range},
                )
                items = top.get("items") or []
                content = "Here are your top tracks:\n" + _render_tracks(items, n=limit)
                return {"spotify": {"top_tracks": items, "time_range": time_range}, "messages": [AIMessage(content=content)]}

            # default fallback
            return {"spotify": {"error": f"unsupported_kind: {kind}"}, "messages": [AIMessage(content="I can show your top artists, tracks, or genres. What would you like?")]}

        # Taste-based (subjective) event flow: fetch small taste profile for later ranking.
        # Keep it bounded and fast: top 25 artists (names only) + derived top genres.
        try:
            top = spotify_api_get(
                access_token=access_token,
                path="/me/top/artists",
                params={"limit": 25, "offset": 0, "time_range": "long_term"},
            )
            items = top.get("items") or []
            artists = [str(a.get("name") or "").strip() for a in items if str(a.get("name") or "").strip()]
            genres = compute_top_genres_from_artists(items, limit=10)
            return {"spotify": {"taste": {"top_artists": artists[:25], "top_genres": genres}}}
        except Exception as e:
            return {"spotify": {"error": f"fetch_error: {e}"}}

    def _pick_location_for_search(route: RouteOutput) -> tuple[str, str, str]:
        # Prefer explicit override if complete; otherwise fall back to profile location.
        ov = route.location_override.model_dump()
        if _has_location(ov):
            return (ov.get("city", ""), ov.get("state", ""), ov.get("country", ""))
        loc = profile.get("location") if isinstance(profile, dict) else {}
        if isinstance(loc, dict):
            return (str(loc.get("city") or ""), str(loc.get("state") or ""), str(loc.get("country") or ""))
        return ("", "", "")

    def _pick_geo_for_search(route: RouteOutput) -> tuple[str, Optional[int]]:
        """
        Returns: (geoPoint, radius_miles)
        geoPoint is a geohash (Ticketmaster). Empty string means unavailable.
        """
        # Decide radius: route.proximity_miles if set, else "near me" default 100 if phrase suggests.
        radius = int(route.proximity_miles or 0) or None

        # 1) If user provided an explicit location override (city/state/country), try to geocode it (best-effort).
        ov = route.location_override.model_dump()
        if _has_location(ov):
            try:
                res = forward_geocode_city(
                    city=str(ov.get("city") or ""),
                    state=str(ov.get("state") or ""),
                    country=str(ov.get("country") or ""),
                    timeout=3,
                )
            except Exception:
                res = None
            if res:
                lat, lng = res
                try:
                    return (geohash_encode(float(lat), float(lng), precision=9), radius)
                except Exception:
                    pass

        # 2) Otherwise use stored profile coordinates (if present).
        loc = profile.get("location") if isinstance(profile, dict) else {}
        if not isinstance(loc, dict):
            return ("", radius)
        coords = loc.get("coordinates") or {}
        if not isinstance(coords, dict):
            return ("", radius)
        lat = coords.get("lat")
        lng = coords.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            try:
                return (geohash_encode(float(lat), float(lng), precision=9), radius)
            except Exception:
                return ("", radius)
        return ("", radius)

    def ticketmaster_search_node(state: SherpaState) -> dict[str, Any]:
        """
        Node E: deterministic Ticketmaster search (grounded).

        - Uses Node C time_window (startDateTime/endDateTime) when available.
        - Uses location override if provided, else profile location.
        - Uses artist_query when present (objective lookups).
        - For subjective discovery without artist_query, uses Spotify taste keywords (top artist, then optional top genre fallback).
        - Caps: max 2 passes, <= 50 events per pass, <= 100 total candidates.
        """
        route_raw = state.get("route") or {}
        try:
            route = RouteOutput.model_validate(route_raw)
        except Exception:
            return {
                "ticketmaster_ok": False,
                "ticketmaster_error": "Routing output invalid; cannot search Ticketmaster.",
                "messages": [AIMessage(content="I couldn't understand the request well enough to search Ticketmaster. What artist and what city should I search?")],
            }

        if not route.needs_ticketmaster:
            return {"ticketmaster_ok": True, "ticketmaster_error": "", "events": [], "used_tool": False}

        geo_point, radius_miles = _pick_geo_for_search(route)
        city, state_code, country_code = _pick_location_for_search(route)
        if not geo_point and not (city.strip() and state_code.strip() and country_code.strip()):
            return {
                "ticketmaster_ok": False,
                "ticketmaster_error": "Missing location",
                "messages": [AIMessage(content="What city/state/country should I search in?")],
            }

        tw = state.get("time_window") or {}
        start_dt = str(tw.get("startDateTime") or "").strip()
        end_dt = str(tw.get("endDateTime") or "").strip()
        # Fallback to the tool default horizon if Node C did not set a window.
        if not start_dt or not end_dt:
            now = datetime.now(timezone.utc)
            start_dt = _iso_utc(now)
            end_dt = _iso_utc(now + timedelta(days=30))

        # Candidate keyword strategy (MVP caps).
        keywords: list[str | None] = []
        artist_q = (route.artist_query or "").strip()
        if artist_q:
            keywords = [artist_q]
        else:
            # For subjective discovery, prefer a top Spotify artist keyword (then optional genre fallback).
            taste = ((state.get("spotify") or {}).get("taste") or {}) if isinstance(state.get("spotify"), dict) else {}
            top_artists = taste.get("top_artists") if isinstance(taste.get("top_artists"), list) else []
            top_genres = taste.get("top_genres") if isinstance(taste.get("top_genres"), list) else []
            k1 = str(top_artists[0]).strip() if top_artists else ""
            k2 = str(top_genres[0]).strip() if top_genres else ""
            if k1:
                keywords.append(k1)
            if k2 and k2 != k1:
                keywords.append(k2)
            if not keywords:
                # Broad search (no keyword)
                keywords.append(None)

        # Hard caps
        max_passes = 2
        per_pass_size = 50
        max_total = 100

        collected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _append_events(events: list[dict[str, Any]]):
            for ev in events:
                eid = str(ev.get("id") or "")
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                collected.append(ev)
                if len(collected) >= max_total:
                    break

        # Execute up to max_passes; only use fallback when it's a taste-based search (no explicit artist_query).
        passes = 0
        for kw in keywords[:max_passes]:
            passes += 1
            try:
                data = _ticketmaster_search_raw(
                    keyword=kw,
                    city=city,
                    state_code=state_code,
                    country_code=country_code,
                    geo_point=geo_point or None,
                    radius=radius_miles,
                    unit="miles",
                    start_date_time=start_dt,
                    end_date_time=end_dt,
                    classification_name="music",
                    size=per_pass_size,
                )
                events_raw = ((data.get("_embedded") or {}).get("events")) or []
                normalized = [_normalize_event(ev) for ev in events_raw]
                _append_events(normalized)
            except Exception as e:
                return {
                    "ticketmaster_ok": False,
                    "ticketmaster_error": str(e),
                    "messages": [],
                }

            # Stop conditions:
            # - If explicit artist query: no fallback attempts.
            # - If we already have enough candidates.
            if artist_q:
                break
            if len(collected) >= 10:
                break

        return {"ticketmaster_ok": True, "ticketmaster_error": "", "events": collected, "used_tool": True}

    def api_failure_node(state: SherpaState) -> dict[str, Any]:
        """
        Node X: grounded failure/recovery branch.

        Covers:
        - Spotify not connected or Spotify token errors when Spotify is needed
        - Ticketmaster request errors/timeouts

        Always returns a deterministic message and ends the turn.
        """
        route = state.get("route") or {}
        spotify_info = state.get("spotify") or {}
        tm_ok = bool(state.get("ticketmaster_ok"))
        tm_err = str(state.get("ticketmaster_error") or "").strip()

        spotify_err = ""
        if isinstance(spotify_info, dict):
            spotify_err = str(spotify_info.get("error") or "").strip()

        # Ticketmaster failure takes priority if it happened.
        if not tm_ok and tm_err:
            content = (
                "I couldn't fetch events from Ticketmaster right now.\n\n"
                f"Reason: {tm_err}\n\n"
                "Try one of these:\n"
                "- Retry the same request\n"
                "- Narrow the time window (e.g., next 14 days)\n"
                "- Change the location (city/state/country)\n"
            )
            return {"failure": {"kind": "ticketmaster", "error": tm_err}, "messages": [AIMessage(content=content)]}

        # Spotify failures / missing connection when Spotify is needed.
        if spotify_err:
            # If the user asked for Spotify stats, the right recovery is "connect Spotify".
            if str(route.get("intent") or "") == "SPOTIFY_STATS":
                content = (
                    "I can answer Spotify listening questions after you connect Spotify.\n\n"
                    "Please connect Spotify in Settings / onboarding, then ask again."
                )
                return {"failure": {"kind": "spotify", "error": spotify_err}, "messages": [AIMessage(content=content)]}

            # If user asked for 'good shows for me' but Spotify isn't available, ask for manual artists.
            content = (
                "I can't access your Spotify taste right now.\n\n"
                "If you want, tell me 1–3 artists you like and I’ll search Ticketmaster near you."
            )
            return {"failure": {"kind": "spotify", "error": spotify_err}, "messages": [AIMessage(content=content)]}

        # Fallback
        return {"failure": {"kind": "unknown"}, "messages": [AIMessage(content="Something went wrong. Please try again.")]}

    def _parse_hhmm(s: str) -> Optional[int]:
        """
        Parse 'HH:MM' or 'HH:MM:SS' into minutes since midnight.
        Returns None if unparseable.
        """
        raw = (s or "").strip()
        if not raw:
            return None
        parts = raw.split(":")
        if len(parts) < 2:
            return None
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except Exception:
            return None
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return hh * 60 + mm

    def filter_best_effort_node(state: SherpaState) -> dict[str, Any]:
        """
        Node F: apply best-effort constraint filtering to Ticketmaster candidates.

        Supports (MVP):
        - weekend_only
        - after_time (e.g. "20:30")
        - genres (best-effort substring match vs event.genre)

        Important: if event time is missing/TBA, we do NOT filter it out for after_time;
        we keep it and record a disclaimer in filter_notes.
        """
        route_raw = state.get("route") or {}
        try:
            route = RouteOutput.model_validate(route_raw)
        except Exception:
            return {"filter_notes": "", "filter_stats": {}}

        events = list(state.get("events") or [])
        if not events:
            return {"filter_notes": "", "filter_stats": {"kept": 0, "dropped": 0}}

        constraints = route.constraints
        want_weekend = bool(constraints.weekend_only)
        want_after = (constraints.after_time or "").strip()
        after_min = _parse_hhmm(want_after) if want_after else None
        want_genres = [g.strip().lower() for g in (constraints.genres or []) if str(g).strip()]

        unknown_time = 0
        dropped_weekend = 0
        dropped_time = 0
        dropped_genre = 0

        kept: list[dict[str, Any]] = []
        for ev in events:
            # Weekend filter
            if want_weekend:
                ds = str(ev.get("date") or "").strip()
                ok_weekend = True
                try:
                    # date is YYYY-MM-DD
                    dt = datetime.fromisoformat(ds)
                    ok_weekend = dt.weekday() >= 5
                except Exception:
                    # If date missing/unparseable, do not drop (best-effort).
                    ok_weekend = True
                if not ok_weekend:
                    dropped_weekend += 1
                    continue

            # Genre filter (best-effort)
            if want_genres:
                g = str(ev.get("genre") or "").lower()
                if g and not any(wg in g for wg in want_genres):
                    dropped_genre += 1
                    continue
                # If genre missing, keep (best-effort).

            # After-time filter (best-effort)
            if after_min is not None:
                ts = str(ev.get("time") or "").strip()
                tmin = _parse_hhmm(ts)
                if tmin is None:
                    # time missing/TBA: keep, but note we couldn't strictly enforce.
                    unknown_time += 1
                else:
                    if tmin < after_min:
                        dropped_time += 1
                        continue

            kept.append(ev)

        notes: list[str] = []
        if after_min is not None and unknown_time:
            notes.append(
                f"Note: {unknown_time} event(s) have no listed start time (TBA), so I couldn’t strictly enforce “after {want_after}”."
            )
        if want_weekend:
            notes.append("Weekend filtering is based on the listed event date; venue time zones/door times may differ.")

        filter_notes = "\n".join(notes).strip()
        stats = {
            "input": len(events),
            "kept": len(kept),
            "dropped": len(events) - len(kept),
            "dropped_weekend": dropped_weekend,
            "dropped_time": dropped_time,
            "dropped_genre": dropped_genre,
            "unknown_time": unknown_time,
        }
        return {"events": kept, "filter_notes": filter_notes, "filter_stats": stats}

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "").strip().lower())

    def rank_node(state: SherpaState) -> dict[str, Any]:
        """
        Node G: deterministic scoring + ranking.

        Goals (MVP):
        - If explicit artist_query is present: prioritize exact/strong matches.
        - If subjective/taste-based: prioritize Spotify top-artist matches, then genre overlap.
        - Penalize missing time when user asked for an after_time constraint (but do not drop events).
        - Keep sort stable by original order for ties.
        """
        route_raw = state.get("route") or {}
        try:
            route = RouteOutput.model_validate(route_raw)
        except Exception:
            return {"rank_stats": {"skipped": True}}

        events = list(state.get("events") or [])
        if not events:
            return {"rank_stats": {"input": 0, "kept": 0}}

        # Inputs
        artist_q = _norm(route.artist_query)
        want_after = (route.constraints.after_time or "").strip()
        after_min = _parse_hhmm(want_after) if want_after else None

        # Taste inputs (from Node D)
        taste = ((state.get("spotify") or {}).get("taste") or {}) if isinstance(state.get("spotify"), dict) else {}
        top_artists = [_norm(a) for a in (taste.get("top_artists") or []) if str(a).strip()]
        top_genres = [_norm(g) for g in (taste.get("top_genres") or []) if str(g).strip()]

        # Caps for scoring work (avoid worst-case costs)
        top_artists = top_artists[:25]
        top_genres = top_genres[:10]

        scored: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        missing_time_count = 0

        for idx, ev in enumerate(events):
            name = _norm(ev.get("name") or "")
            artist = _norm(ev.get("artist") or "")
            genre = _norm(ev.get("genre") or "")
            t = str(ev.get("time") or "").strip()
            tmin = _parse_hhmm(t)

            breakdown: dict[str, Any] = {}
            score = 0.0

            # 1) Explicit artist query match (strongest)
            if artist_q:
                # exact-ish match gets big boost
                if artist_q and (artist_q == artist or artist_q == name):
                    score += 5.0
                    breakdown["artist_query_match"] = "exact"
                elif artist_q and (artist_q in artist or artist_q in name or artist in artist_q or name in artist_q):
                    score += 3.5
                    breakdown["artist_query_match"] = "partial"
                else:
                    breakdown["artist_query_match"] = "none"

            # 2) Taste match via Spotify top artists (medium-strong)
            if top_artists:
                if artist and any(ta == artist for ta in top_artists):
                    score += 2.5
                    breakdown["spotify_artist_match"] = "exact"
                elif artist and any(ta in artist or artist in ta for ta in top_artists[:10]):
                    score += 1.5
                    breakdown["spotify_artist_match"] = "partial"

            # 3) Genre overlap (medium)
            if top_genres and genre:
                if any(g in genre for g in top_genres):
                    score += 0.8
                    breakdown["genre_overlap"] = True

            # 4) Time constraint fit (small boost / penalty for missingness)
            if after_min is not None:
                if tmin is None:
                    missing_time_count += 1
                    score -= 0.15  # slight penalty so known-good times rank above unknown
                    breakdown["time_after"] = "unknown"
                elif tmin >= after_min:
                    score += 0.2
                    breakdown["time_after"] = "ok"
                else:
                    # Should usually be filtered out by Node F, but keep defensive.
                    score -= 1.0
                    breakdown["time_after"] = "fails"

            scored.append((score, idx, ev, breakdown))

        # Stable sort: score desc, then original index asc
        scored.sort(key=lambda tup: (-tup[0], tup[1]))
        ranked_events = [ev for _, _, ev, _ in scored]

        stats = {
            "input": len(events),
            "kept": len(ranked_events),
            "missing_time": missing_time_count,
            "has_artist_query": bool(artist_q),
            "has_taste": bool(top_artists or top_genres),
        }

        # Keep breakdowns only for debugging (not used by renderer yet), but store a small sample.
        sample = []
        for s, _, ev, b in scored[: min(5, len(scored))]:
            sample.append({"id": ev.get("id"), "score": s, "breakdown": b})
        stats["top_sample"] = sample

        return {"events": ranked_events, "rank_stats": stats}

    def picks_why_node(state: SherpaState) -> dict[str, Any]:
        """
        Node I: strict structured output for 'picks + why'.

        - Input: ranked events (state.events)
        - Output: state.picks_payload = {intro, picks:[{id,why}], question}
        - Constraints:
          - IDs must exist in provided events
          - Max 3 picks
          - One retry, then deterministic fallback to top 3
        """
        events = list(state.get("events") or [])
        if not events:
            payload = PicksWhyOutput(
                intro="I searched Ticketmaster but didn’t find any matching music events for that window.",
                picks=[],
                question="Want to broaden the date range or try a nearby city?",
            )
            return {"picks_payload": payload.model_dump()}

        # Build compact events for the LLM: include only identifiers + high-level descriptors.
        compact_events = [
            {
                "id": str(ev.get("id") or ""),
                "name": str(ev.get("name") or ""),
                "artist": str(ev.get("artist") or ""),
                "venue": str(ev.get("venue") or ""),
                "date": str(ev.get("date") or ""),
                "time": str(ev.get("time") or ""),
                "genre": str(ev.get("genre") or ""),
                "price": str(ev.get("price") or ""),
            }
            for ev in events[:50]
        ]

        # Provide small scoring hints if available.
        rank_stats = state.get("rank_stats") or {}
        score_hints = rank_stats.get("top_sample") if isinstance(rank_stats, dict) else None

        sys = SystemMessage(
            content=_system_profile(profile)
            + "\nYou are selecting the best 3 concerts from a list of Ticketmaster events.\n"
            + "Return ONLY structured data matching the schema.\n"
            + "Rules:\n"
            + "- Never invent events or facts.\n"
            + "- Picks must use ONLY IDs that appear in the provided events.\n"
            + "- Provide at most 3 picks.\n"
            + "- 'why' should be short and should reference general reasoning (taste match, genre, timing) without inventing facts.\n"
        )

        def _call(strict_extra: str = "") -> PicksWhyOutput:
            out_llm = llm.with_structured_output(PicksWhyOutput)
            user = HumanMessage(
                content=json.dumps(
                    {"events": compact_events, "score_hints": score_hints, "user_request": (state.get("user_input") or "")},
                    ensure_ascii=False,
                )
            )
            if strict_extra:
                sys2 = SystemMessage(content=sys.content + "\n" + strict_extra)
                return out_llm.invoke([sys2, user])
            return out_llm.invoke([sys, user])

        try:
            draft = _call()
        except Exception:
            try:
                draft = _call("CRITICAL: Output MUST match schema exactly. No extra keys. No prose.")
            except Exception:
                draft = PicksWhyOutput()

        # Validate IDs and cap to 3
        by_id = {str(ev.get("id") or ""): ev for ev in events if str(ev.get("id") or "")}
        valid: list[PickWhyItem] = []
        for p in (draft.picks or [])[:3]:
            pid = str(getattr(p, "id", "") or "").strip()
            if not pid or pid not in by_id:
                continue
            why = str(getattr(p, "why", "") or "").strip()
            valid.append(PickWhyItem(id=pid, why=why))

        if not valid:
            # Deterministic fallback: top 3 ranked events.
            fallback = []
            for ev in events[:3]:
                pid = str(ev.get("id") or "").strip()
                if not pid:
                    continue
                fallback.append(PickWhyItem(id=pid, why="Popular match based on your query and the available event details."))
            payload = PicksWhyOutput(intro=str(draft.intro or PicksWhyOutput().intro), picks=fallback, question=draft.question)
            return {"picks_payload": payload.model_dump()}

        payload = PicksWhyOutput(
            intro=str(draft.intro or PicksWhyOutput().intro),
            picks=valid,
            question=draft.question,
        )
        return {"picks_payload": payload.model_dump()}
    def render_node(state: SherpaState) -> dict[str, Any]:
        """
        Node J: deterministic renderer for the Ticketmaster pipeline.
        Never calls the LLM. Renders ONLY from Ticketmaster event objects + picks_payload.
        """
        events = list(state.get("events") or [])
        if not events:
            return {"messages": [AIMessage(content="I searched Ticketmaster but didn’t find any matching music events for the selected window.")]}

        picks_payload = state.get("picks_payload") or {}
        parsed: Optional[PicksWhyOutput] = None
        if isinstance(picks_payload, dict):
            try:
                parsed = PicksWhyOutput.model_validate(picks_payload)
            except Exception:
                parsed = None

        by_id = {str(ev.get("id") or ""): ev for ev in events}
        intro = "Here are a few real, bookable concerts I found:"
        question: Optional[str] = None
        lines: list[str] = []

        if parsed and parsed.picks:
            intro = (parsed.intro or intro).strip()
            question = parsed.question
            for p in parsed.picks[:3]:
                ev = by_id.get(p.id)
                if not ev:
                    continue
                line = _render_event_line(ev)
                why = (p.why or "").strip()
                if why:
                    line = line + f"\n  _Why_: {why}"
                lines.append(line)

        if not lines:
            # Fallback: show top 5 events (already ranked by Node G)
            lines = [_render_event_line(ev) for ev in events[:5] if (ev.get("name") or "").strip()]

        content = intro.strip() + "\n\n" + "\n".join(lines)
        if question:
            content += "\n\n" + str(question).strip()

        filter_notes = str(state.get("filter_notes") or "").strip()
        if filter_notes:
            content += "\n\n" + filter_notes

        return {"messages": [AIMessage(content=content)]}
    def fallback_respond_node(state: SherpaState) -> dict[str, Any]:
        """
        Deterministic responder for requests that don't require Ticketmaster and aren't Spotify-only.
        Prevents empty outputs when intent is OTHER.
        """
        route = state.get("route") or {}
        q = str(route.get("clarifying_question") or "").strip()
        if q:
            return {"messages": [AIMessage(content=q)]}
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can help with:\n"
                        "- Finding real, bookable concerts near you (Ticketmaster)\n"
                        "- Answering Spotify listening questions (top artists/tracks/genres)\n\n"
                        "What would you like to do?"
                    )
                )
            ]
        }

    def should_clarify_or_continue(
        state: SherpaState,
    ) -> Literal["clarify_node", "spotify_fetch_node", "normalize_time_node", "fallback_respond_node"]:
        """
        After routing, decide whether we need a clarifying question before proceeding.
        Deterministic gate in addition to Node A's suggestion.
        """
        route = state.get("route") or {}
        # Spotify-only stats flow can proceed directly to spotify_fetch_node.
        if str(route.get("intent") or "") == "SPOTIFY_STATS" and not bool(route.get("needs_ticketmaster")):
            return "spotify_fetch_node"

        needs_tm = bool(route.get("needs_ticketmaster"))
        if not needs_tm:
            return "fallback_respond_node"

        # If user provided a location override in slots, accept it (even partial) for now.
        override = route.get("location_override") or {}
        has_override = _has_location(override if isinstance(override, dict) else None)

        # Otherwise require profile location.
        prof_loc = profile.get("location") if isinstance(profile, dict) else None
        has_prof = _has_location(prof_loc if isinstance(prof_loc, dict) else None)

        # If Node A asked for clarification, honor it.
        if bool(route.get("needs_clarification")):
            return "clarify_node"
        if not has_override and not has_prof:
            return "clarify_node"
        return "normalize_time_node"

    builder = StateGraph(SherpaState)
    builder.add_node("ingest_node", ingest_node)
    builder.add_node("memory_node", memory_node)
    builder.add_node("route_and_extract_node", route_and_extract_node)
    builder.add_node("clarify_node", clarify_node)
    builder.add_node("fallback_respond_node", fallback_respond_node)
    builder.add_node("normalize_time_node", normalize_time_node)
    builder.add_node("spotify_fetch_node", spotify_fetch_node)
    builder.add_node("ticketmaster_search_node", ticketmaster_search_node)
    builder.add_node("filter_best_effort_node", filter_best_effort_node)
    builder.add_node("rank_node", rank_node)
    builder.add_node("picks_why_node", picks_why_node)
    builder.add_node("render_node", render_node)
    builder.add_node("api_failure_node", api_failure_node)
    builder.add_edge(START, "ingest_node")
    builder.add_edge("ingest_node", "memory_node")
    builder.add_edge("memory_node", "route_and_extract_node")
    builder.add_conditional_edges(
        "route_and_extract_node",
        should_clarify_or_continue,
        ["clarify_node", "spotify_fetch_node", "normalize_time_node", "fallback_respond_node"],
    )
    builder.add_edge("clarify_node", END)
    builder.add_edge("fallback_respond_node", END)
    builder.add_edge("normalize_time_node", "spotify_fetch_node")

    def after_spotify_next(
        state: SherpaState,
    ) -> Literal["ticketmaster_search_node", "api_failure_node", "fallback_respond_node", "__end__"]:
        """
        After Node D:
        - If Spotify-only intent already responded, end.
        - If Ticketmaster is needed, go to Node E deterministic search.
        - Otherwise respond deterministically.
        """
        route = state.get("route") or {}
        if str(route.get("intent") or "") == "SPOTIFY_STATS" and not bool(route.get("needs_ticketmaster")):
            return "__end__"
        if bool(route.get("needs_ticketmaster")):
            # If Spotify is required but unavailable for a taste-based query with no explicit artist,
            # route to failure node to ask for manual artists.
            spotify_err = ""
            sp = state.get("spotify") or {}
            if isinstance(sp, dict):
                spotify_err = str(sp.get("error") or "").strip()
            if spotify_err and not str(route.get("artist_query") or "").strip():
                return "api_failure_node"
            return "ticketmaster_search_node"
        return "fallback_respond_node"

    builder.add_conditional_edges(
        "spotify_fetch_node",
        after_spotify_next,
        ["ticketmaster_search_node", "api_failure_node", "fallback_respond_node", END],
    )
    builder.add_edge("api_failure_node", END)

    def after_ticketmaster_next(state: SherpaState) -> Literal["filter_best_effort_node", "api_failure_node"]:
        return "filter_best_effort_node" if bool(state.get("ticketmaster_ok")) else "api_failure_node"

    builder.add_conditional_edges(
        "ticketmaster_search_node",
        after_ticketmaster_next,
        ["filter_best_effort_node", "api_failure_node"],
    )
    builder.add_edge("filter_best_effort_node", "rank_node")
    builder.add_edge("rank_node", "picks_why_node")
    builder.add_edge("picks_why_node", "render_node")
    builder.add_edge("render_node", END)
    graph = builder.compile()

    return graph


def run_agent(message: str, profile: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Execute the ShowSherpa LangGraph agent.
    Returns: { "content": str, "events": [...] }
    """
    graph = _plan_node_factory(profile)
    prior = _coerce_history_messages(history)
    # Node 0 ingests user_input into messages. Keep history bounded here.
    prior_bounded = prior[-20:]
    state = graph.invoke(
        {
            "messages": prior_bounded,
            "user_input": message,
            "events": [],
            "used_tool": False,
            "memory": {"literal_summary": [], "intent_summary": []},
            "route": {},
            "time_window": {},
            "spotify": {},
            "ticketmaster_ok": True,
            "ticketmaster_error": "",
            "filter_notes": "",
            "filter_stats": {},
            "rank_stats": {},
            "picks_payload": {},
            "failure": {},
        }
    )
    messages = state.get("messages") or []
    last_text = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_text = m.content or ""
            break
    return {"content": last_text, "events": state.get("events") or []}


