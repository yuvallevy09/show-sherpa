# ShowSherpa

Vite + React frontend with a FastAPI backend (in `backend/`).

## Running the app

### Backend (FastAPI)

```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
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