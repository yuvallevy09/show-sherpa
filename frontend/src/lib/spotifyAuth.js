// PKCE helpers for Spotify Authorization Code with PKCE flow.

function base64UrlEncode(bytes) {
  const bin = String.fromCharCode(...bytes);
  const b64 = btoa(bin);
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(len = 64) {
  const bytes = new Uint8Array(len);
  crypto.getRandomValues(bytes);
  // map to unreserved chars
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
  return Array.from(bytes, (b) => chars[b % chars.length]).join("");
}

export async function pkceChallengeFromVerifier(verifier) {
  const data = new TextEncoder().encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

export async function startSpotifyAuth({
  clientId,
  redirectUri,
  scopes = ["user-top-read", "user-read-email", "user-read-private"],
}) {
  if (!clientId) throw new Error("Missing Spotify client id.");
  if (!redirectUri) throw new Error("Missing Spotify redirect uri.");

  const state = randomString(32);
  const codeVerifier = randomString(64);
  const codeChallenge = await pkceChallengeFromVerifier(codeVerifier);

  sessionStorage.setItem("spotify_pkce_state", state);
  sessionStorage.setItem("spotify_pkce_verifier", codeVerifier);

  const params = new URLSearchParams({
    response_type: "code",
    client_id: clientId,
    redirect_uri: redirectUri,
    state,
    scope: scopes.join(" "),
    code_challenge_method: "S256",
    code_challenge: codeChallenge,
  });

  window.location.assign(`https://accounts.spotify.com/authorize?${params.toString()}`);
}

export function readSpotifyPkceState() {
  return {
    state: sessionStorage.getItem("spotify_pkce_state") || "",
    verifier: sessionStorage.getItem("spotify_pkce_verifier") || "",
  };
}

export function clearSpotifyPkceState() {
  sessionStorage.removeItem("spotify_pkce_state");
  sessionStorage.removeItem("spotify_pkce_verifier");
}


