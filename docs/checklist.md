### Must-have features 

1. **Spotify auth + stats endpoint**

   * Use Spotify OAuth and fetch `/me/top/{artists|tracks}` with `user-top-read` for personalization/stats. ([developer.spotify.com][2])
   * Handle token refresh (access tokens expire; refresh tokens are used to obtain new ones). ([developer.spotify.com][3])

2. **Ticketmaster grounded event search**

   * Search events via Discovery API with your `apikey` and push down time/location filters. ([The Ticketmaster Developer Portal][4])
   * Prefer `geoPoint` over deprecated `latlong` when you have geo coordinates. ([documenter.getpostman.com][5])

3. **Strict structured outputs in exactly two places**

   * Node A (`route_and_extract`) and Node I (`picks_why`) must be **schema-validated**, with **one retry** then a safe fallback (ask a clarifying question or do deterministic picks).

4. **Deterministic “render facts ourselves” response**

   * The LLM never invents event facts. You render event name/date/venue/link **only from Ticketmaster objects**, and Spotify stats only from Spotify responses. (This is your strongest hallucination-control story.)

5. **Clarification + context**

   * If Ticketmaster is needed and location is missing → ask once (city/postal code + radius), remember it for the session (and only persist if user confirms).

6. **Failure branch**

   * If Spotify fails → offer “give me 1–3 artists manually” and still do Ticketmaster.
   * If Ticketmaster fails → offer retry/narrow window/change location.
   * This is what makes hallucination handling look robust.

### One small upgrade I’d add (still MVP-simple)

Allow **one fallback Ticketmaster query** if the first returns 0 results (e.g., try “top artist” keyword, then try “top genre” keyword). Keep hard caps so it can’t blow up rate limits.

If you implement Plan 3 with the checklist above, you’ll be able to produce transcripts that clearly demonstrate **two-API fusion**, **multi-turn context**, and **hallucination mitigation**—which is exactly what they’re grading.

[1]: https://docs.langchain.com/oss/python/langgraph/graph-api?utm_source=chatgpt.com "Graph API overview - Docs by LangChain"
[2]: https://developer.spotify.com/documentation/web-api/reference/get-users-top-artists-and-tracks?utm_source=chatgpt.com "Get User's Top Items"
[3]: https://developer.spotify.com/documentation/web-api/tutorials/refreshing-tokens?utm_source=chatgpt.com "Refreshing tokens"
[4]: https://developer.ticketmaster.com/products-and-docs/tutorials/events-search/search_events_with_discovery_api.html?utm_source=chatgpt.com "Get started with The Discovery API"
[5]: https://documenter.getpostman.com/view/1034536/2s847FwEYE?utm_source=chatgpt.com "Ticketmaster Discovery API"