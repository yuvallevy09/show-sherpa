import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { api } from "@/api/apiClient";
import { readSpotifyPkceState, clearSpotifyPkceState } from "@/lib/spotifyAuth";

export default function SpotifyCallbackPage() {
  const navigate = useNavigate();
  const [error, setError] = useState("");

  useEffect(() => {
    const run = async () => {
      const url = new URL(window.location.href);
      const code = url.searchParams.get("code") || "";
      const state = url.searchParams.get("state") || "";
      const err = url.searchParams.get("error") || "";

      if (err) {
        setError(`Spotify authorization failed: ${err}`);
        return;
      }

      const saved = readSpotifyPkceState();
      if (!code || !state || !saved.state || state !== saved.state) {
        setError("Invalid Spotify callback state. Please try connecting again.");
        return;
      }
      if (!saved.verifier) {
        setError("Missing PKCE verifier. Please try connecting again.");
        return;
      }

      try {
        await api.spotify.exchange({ code, code_verifier: saved.verifier });
        clearSpotifyPkceState();
        // Refresh user state
        await api.auth.me();
        navigate("/Chat", { replace: true });
      } catch (e) {
        setError(e?.message || String(e));
      }
    };
    run();
  }, [navigate]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-violet-50/30 p-6">
        <div className="max-w-lg w-full bg-white border border-slate-200/60 rounded-2xl p-6 shadow-sm">
          <h1 className="text-xl font-semibold text-slate-900 mb-2">Spotify connection failed</h1>
          <p className="text-slate-600 mb-4">{error}</p>
          <button
            className="px-4 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800"
            onClick={() => navigate("/Chat")}
          >
            Back to Chat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-violet-50/30">
      <div className="flex items-center gap-3 text-slate-700">
        <Loader2 className="w-6 h-6 text-violet-600 animate-spin" />
        <span>Finishing Spotify connection…</span>
      </div>
    </div>
  );
}


