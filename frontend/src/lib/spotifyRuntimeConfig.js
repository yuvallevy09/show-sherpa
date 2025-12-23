const LS_SPOTIFY_CLIENT_ID = "showsherpa_spotify_client_id_v1";

export function getSpotifyClientId() {
  return (import.meta.env.VITE_SPOTIFY_CLIENT_ID || localStorage.getItem(LS_SPOTIFY_CLIENT_ID) || "").trim();
}

export function setSpotifyClientId(clientId) {
  const v = String(clientId || "").trim();
  if (!v) return;
  localStorage.setItem(LS_SPOTIFY_CLIENT_ID, v);
}

export function getSpotifyRedirectUri() {
  // Prefer explicit env value; otherwise default to same-origin callback.
  return (import.meta.env.VITE_SPOTIFY_REDIRECT_URI || `${window.location.origin}/spotify/callback`).trim();
}


