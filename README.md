# ShowSherpa

Vite + React frontend with a FastAPI backend (in `backend/`).

## Running the app

### Backend (FastAPI)

Set your Ticketmaster API key (Discovery API uses API key auth via the `apikey` query param):

```bash
export TICKETMASTER_API_KEY=YOUR_TICKETMASTER_API_KEY
```

Optional: enable the LangGraph agent (Groq via LangChain). If `GROQ_API_KEY` is not set, the frontend will fall back to Ticketmaster-only behavior.

```bash
export GROQ_API_KEY=YOUR_GROQ_API_KEY
export GROQ_MODEL=llama-3.3-70b-versatile
```

Optional fallback: Google AI Studio (Gemini). If `GROQ_API_KEY` is not set but `GOOGLE_API_KEY` is, the agent will use Gemini.

```bash
export GOOGLE_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY
export GOOGLE_MODEL=gemini-1.5-flash
```

```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

### Spotify (Web API)

This project uses **Authorization Code with PKCE** so we can read **your top artists/genres** and fuse them into Ticketmaster searches.

- Register a Spotify app (select **Web API**) and add this redirect URI (use loopback IP, not `localhost`):
  - `http://127.0.0.1:5173/spotify/callback`

Set these backend env vars (see `backend/env.example`):

```bash
export SPOTIFY_CLIENT_ID=YOUR_SPOTIFY_CLIENT_ID
export SPOTIFY_CLIENT_SECRET=YOUR_SPOTIFY_CLIENT_SECRET
export SPOTIFY_REDIRECT_URI=http://127.0.0.1:5173/spotify/callback
```

Set these frontend env vars (see `frontend/env.example`):

```bash
export VITE_SPOTIFY_CLIENT_ID=YOUR_SPOTIFY_CLIENT_ID
export VITE_SPOTIFY_REDIRECT_URI=http://127.0.0.1:5173/spotify/callback
```

Quick sanity check (should return `{ "events": [...] }`):

```bash
curl "http://localhost:8000/ticketmaster/events?classificationName=music&city=New%20York&stateCode=NY&countryCode=US&size=5"
```

Agent sanity check (should return `{ "content": "...", "concerts": [...] }` when `GROQ_API_KEY` is set):

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message":"Any concerts near me this month?"}'
```

### Frontend (Vite)

Set the backend URL (Vite env var):

```bash
export VITE_API_URL=http://localhost:8000
```

```bash
cd frontend
npm install
npm run dev
```

## Building the app

```bash
cd frontend
npm run build
```

## Repo layout

- **`frontend/`**: Vite + React app (Tailwind + shadcn/ui)
- **`backend/`**: FastAPI app managed by `uv`