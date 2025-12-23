import React, { useState } from 'react';
import { Music, CheckCircle2, MapPin, Loader2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '@/api/apiClient';
import { startSpotifyAuth } from "@/lib/spotifyAuth";
import { getSpotifyClientId, getSpotifyRedirectUri, setSpotifyClientId } from "@/lib/spotifyRuntimeConfig";

export default function SpotifyConnect({ onComplete }) {
  const [step, setStep] = useState(1); // 1: welcome, 2: spotify, 3: location
  const [isConnecting, setIsConnecting] = useState(false);
  const [location, setLocation] = useState({
    city: '',
    country: '',
    state: ''
  });

  const handleSpotifyConnect = async () => {
    setIsConnecting(true);

    try {
      // Spotify requires loopback IP literals (not localhost). Also, PKCE state is stored per-origin,
      // so force the app to run on 127.0.0.1 before starting auth.
      if (window.location.hostname === "localhost") {
        const nextUrl = window.location.href.replace("://localhost:", "://127.0.0.1:");
        window.location.replace(nextUrl);
        return;
      }

      let clientId = getSpotifyClientId();
      if (!clientId) {
        const entered = window.prompt("Enter your Spotify Client ID (from Spotify Developer Dashboard):");
        if (entered) setSpotifyClientId(entered);
        clientId = getSpotifyClientId();
      }

      const redirectUri = getSpotifyRedirectUri();
      await startSpotifyAuth({ clientId, redirectUri });
    } catch (e) {
      setIsConnecting(false);
      alert(e?.message || String(e));
    }
  };

  const handleLocationSubmit = async (e) => {
    e.preventDefault();
    
    await api.auth.updateMe({
      location: location
    });
    
    onComplete();
  };

  const handleSkipLocation = async () => {
    await api.auth.updateMe({
      location: { city: 'New York', country: 'US', state: 'NY' }
    });
    onComplete();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-600 via-purple-600 to-indigo-700 flex items-center justify-center p-4">
      <AnimatePresence mode="wait">
        {/* Step 1: Welcome */}
        {step === 1 && (
          <motion.div
            key="welcome"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="max-w-md w-full"
          >
            <div className="bg-white rounded-3xl shadow-2xl p-8 text-center">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-violet-500/50">
                <Music className="w-10 h-10 text-white" />
              </div>
              
              <h1 className="text-3xl font-bold text-slate-900 mb-3">
                Welcome to ShowSherpa
              </h1>
              <p className="text-slate-600 text-lg mb-8 leading-relaxed">
                Your personal live-music concierge powered by your Spotify listening history and real concert data.
              </p>

              <div className="space-y-3 mb-8 text-left">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <CheckCircle2 className="w-4 h-4 text-violet-600" />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900">Personalized Recommendations</p>
                    <p className="text-sm text-slate-600">Based on your actual music taste</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <CheckCircle2 className="w-4 h-4 text-violet-600" />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900">Real Concert Data</p>
                    <p className="text-sm text-slate-600">Live events from Ticketmaster</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-violet-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <CheckCircle2 className="w-4 h-4 text-violet-600" />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900">Conversational Search</p>
                    <p className="text-sm text-slate-600">Just ask naturally, we'll find the perfect shows</p>
                  </div>
                </div>
              </div>

              <Button
                onClick={() => setStep(2)}
                className="w-full h-12 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white rounded-xl font-semibold shadow-lg shadow-violet-500/30"
              >
                Get Started
              </Button>
            </div>
          </motion.div>
        )}

        {/* Step 2: Spotify Connection */}
        {step === 2 && (
          <motion.div
            key="spotify"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="max-w-md w-full"
          >
            <div className="bg-white rounded-3xl shadow-2xl p-8 text-center">
              <div className="w-20 h-20 rounded-2xl bg-[#1DB954] flex items-center justify-center mx-auto mb-6 shadow-xl shadow-green-500/50">
                <svg className="w-10 h-10 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                </svg>
              </div>
              
              <h2 className="text-2xl font-bold text-slate-900 mb-3">
                Connect Your Spotify
              </h2>
              <p className="text-slate-600 mb-8 leading-relaxed">
                We'll analyze your listening history to recommend concerts you'll actually love. Your data stays private and secure.
              </p>

              <div className="bg-slate-50 rounded-xl p-4 mb-8 text-sm text-slate-600 text-left">
                <p className="font-medium text-slate-900 mb-2">We'll access:</p>
                <ul className="space-y-1.5">
                  <li className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                    Your top artists and tracks
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                    Your favorite genres
                  </li>
                  <li className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                    Your listening preferences
                  </li>
                </ul>
              </div>

              <Button
                onClick={handleSpotifyConnect}
                disabled={isConnecting}
                className="w-full h-12 bg-[#1DB954] hover:bg-[#1ed760] text-white rounded-xl font-semibold shadow-lg mb-3"
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Connecting to Spotify...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                    </svg>
                    Connect with Spotify
                  </>
                )}
              </Button>

              <p className="text-xs text-slate-400">
                By connecting, you agree to share your music preferences
              </p>
            </div>
          </motion.div>
        )}

        {/* Step 3: Location */}
        {step === 3 && (
          <motion.div
            key="location"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="max-w-md w-full"
          >
            <div className="bg-white rounded-3xl shadow-2xl p-8">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-violet-500/50">
                <MapPin className="w-10 h-10 text-white" />
              </div>
              
              <h2 className="text-2xl font-bold text-slate-900 mb-3 text-center">
                Where Are You Located?
              </h2>
              <p className="text-slate-600 mb-8 text-center leading-relaxed">
                Help us find concerts near you. You can always change this later.
              </p>

              <form onSubmit={handleLocationSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="city" className="text-sm font-medium text-slate-700">
                      City <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="city"
                      value={location.city}
                      onChange={(e) => setLocation({...location, city: e.target.value})}
                      placeholder="New York"
                      className="mt-1.5 h-11 rounded-xl"
                      required
                    />
                  </div>

                  <div>
                    <Label htmlFor="country" className="text-sm font-medium text-slate-700">
                      Country <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="country"
                      value={location.country}
                      onChange={(e) => setLocation({...location, country: e.target.value})}
                      placeholder="US"
                      className="mt-1.5 h-11 rounded-xl"
                      required
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="state" className="text-sm font-medium text-slate-700">
                    State/Region <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    id="state"
                    value={location.state}
                    onChange={(e) => setLocation({...location, state: e.target.value})}
                    placeholder="NY"
                    className="mt-1.5 h-11 rounded-xl"
                    required
                  />
                </div>

                <div className="pt-4 space-y-3">
                  <Button
                    type="submit"
                    className="w-full h-12 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white rounded-xl font-semibold shadow-lg shadow-violet-500/30"
                  >
                    Continue to Chat
                  </Button>

                  <Button
                    type="button"
                    variant="ghost"
                    onClick={handleSkipLocation}
                    className="w-full h-12 text-slate-600 hover:text-slate-900 rounded-xl"
                  >
                    Skip for Now
                  </Button>
                </div>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress Indicator */}
      <div className="fixed bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2">
        {[1, 2, 3].map((s) => (
          <div
            key={s}
            className={`h-2 rounded-full transition-all duration-300 ${
              s === step 
                ? 'w-8 bg-white' 
                : s < step 
                ? 'w-2 bg-white/60' 
                : 'w-2 bg-white/30'
            }`}
          />
        ))}
      </div>
    </div>
  );
}