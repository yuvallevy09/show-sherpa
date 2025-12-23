import React, { useState, useEffect } from 'react';
import { api } from '@/api/apiClient';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Music, MapPin, User, LogOut, Loader2, Check, Plus, X, Navigation } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link } from 'react-router-dom';
import { createPageUrl } from '../utils';

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const [location, setLocation] = useState({ city: '', country: '', state: '' });
  const [isSaving, setIsSaving] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [newGenre, setNewGenre] = useState('');
  const [newArtist, setNewArtist] = useState('');

  const { data: user, isLoading } = useQuery({
    queryKey: ['currentUser'],
    queryFn: () => api.auth.me(),
  });

  useEffect(() => {
    if (user?.location) {
      setLocation(user.location);
    }
  }, [user]);

  const handleSaveLocation = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    
    await api.auth.updateMe({ location });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    
    setIsSaving(false);
  };

  const handleGetCurrentLocation = () => {
    setIsGettingLocation(true);
    
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const mockLocation = {
            city: 'New York',
            country: 'US',
            state: 'NY'
          };
          
          setLocation(mockLocation);
          await api.auth.updateMe({ location: mockLocation });
          queryClient.invalidateQueries({ queryKey: ['currentUser'] });
          setIsGettingLocation(false);
        },
        (error) => {
          console.error('Error getting location:', error);
          setIsGettingLocation(false);
          alert('Unable to get your location. Please enter it manually.');
        }
      );
    } else {
      setIsGettingLocation(false);
      alert('Geolocation is not supported by your browser.');
    }
  };

  const handleAddGenre = async () => {
    if (!newGenre.trim()) return;
    const updatedGenres = [...(user.custom_genres || []), newGenre.trim()];
    await api.auth.updateMe({ custom_genres: updatedGenres });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    setNewGenre('');
  };

  const handleRemoveGenre = async (genre) => {
    const updatedGenres = (user.custom_genres || []).filter(g => g !== genre);
    await api.auth.updateMe({ custom_genres: updatedGenres });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
  };

  const handleAddArtist = async () => {
    if (!newArtist.trim()) return;
    const updatedArtists = [...(user.custom_artists || []), newArtist.trim()];
    await api.auth.updateMe({ custom_artists: updatedArtists });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    setNewArtist('');
  };

  const handleRemoveArtist = async (artist) => {
    const updatedArtists = (user.custom_artists || []).filter(a => a !== artist);
    await api.auth.updateMe({ custom_artists: updatedArtists });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
  };

  const handleDisconnectSpotify = async () => {
    await api.auth.updateMe({ 
      spotify_connected: false,
      spotify_profile: null 
    });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
  };

  const handleReconnectSpotify = async () => {
    // Simulate reconnection
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockSpotifyData = {
      display_name: user?.full_name || "Music Lover",
      top_artists: ["Tame Impala", "The Midnight", "CHVRCHES", "Japanese Breakfast"],
      top_genres: ["indie rock", "synthwave", "indie pop", "psychedelic rock"]
    };
    
    await api.auth.updateMe({
      spotify_connected: true,
      spotify_profile: mockSpotifyData
    });
    
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-violet-50/30">
        <Loader2 className="w-8 h-8 text-violet-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-violet-50/30">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Settings</h1>
            <p className="text-slate-600 mt-1">Manage your ShowSherpa preferences</p>
          </div>
          <Button asChild variant="outline" className="rounded-xl">
            <Link to={createPageUrl('Chat')}>
              Back to Chat
            </Link>
          </Button>
        </div>

        {/* Profile Card */}
        <Card className="border-slate-200/60 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="w-5 h-5 text-violet-600" />
              Profile
            </CardTitle>
            <CardDescription>Your account information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <Label className="text-sm text-slate-600">Name</Label>
              <p className="font-medium text-slate-900">{user?.full_name}</p>
            </div>
            <div>
              <Label className="text-sm text-slate-600">Email</Label>
              <p className="font-medium text-slate-900">{user?.email}</p>
            </div>
            <Button
              variant="outline"
              className="mt-4 text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200 rounded-xl"
              onClick={() => api.auth.logout()}
            >
              <LogOut className="w-4 h-4 mr-2" />
              Log Out
            </Button>
          </CardContent>
        </Card>

        {/* Spotify Connection Card */}
        <Card className="border-slate-200/60 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Music className="w-5 h-5 text-[#1DB954]" />
              Spotify Connection
            </CardTitle>
            <CardDescription>Manage your Spotify integration</CardDescription>
          </CardHeader>
          <CardContent>
            {user?.spotify_connected ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-green-50 rounded-xl border border-green-200">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-[#1DB954] flex items-center justify-center">
                      <Check className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-900">Connected</p>
                      <p className="text-sm text-slate-600">
                        {user.spotify_profile?.display_name || 'Spotify User'}
                      </p>
                    </div>
                  </div>
                  <Badge className="bg-green-100 text-green-800 border-green-200">
                    Active
                  </Badge>
                </div>

                <div>
                  <Label className="text-sm text-slate-600 mb-2 block">Genres</Label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {user.spotify_profile?.top_genres && user.spotify_profile.top_genres.slice(0, 5).map((genre, idx) => (
                      <Badge key={`spotify-genre-${idx}`} variant="outline" className="bg-violet-50 text-violet-700 border-violet-200">
                        {genre}
                      </Badge>
                    ))}
                    {(user.custom_genres || []).map((genre, idx) => (
                      <Badge key={`custom-genre-${idx}`} variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 pr-1">
                        {genre}
                        <button
                          onClick={() => handleRemoveGenre(genre)}
                          className="ml-1.5 hover:bg-blue-200 rounded-full p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      value={newGenre}
                      onChange={(e) => setNewGenre(e.target.value)}
                      placeholder="Add genre (e.g., jazz, metal)"
                      className="h-9 rounded-xl"
                      onKeyPress={(e) => e.key === 'Enter' && handleAddGenre()}
                    />
                    <Button
                      onClick={handleAddGenre}
                      size="sm"
                      variant="outline"
                      className="rounded-xl"
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                <div>
                  <Label className="text-sm text-slate-600 mb-2 block">Artists</Label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {user.spotify_profile?.top_artists && user.spotify_profile.top_artists.slice(0, 4).map((artist, idx) => (
                      <Badge key={`spotify-artist-${idx}`} variant="outline" className="bg-violet-50 text-violet-700 border-violet-200">
                        {artist}
                      </Badge>
                    ))}
                    {(user.custom_artists || []).map((artist, idx) => (
                      <Badge key={`custom-artist-${idx}`} variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 pr-1">
                        {artist}
                        <button
                          onClick={() => handleRemoveArtist(artist)}
                          className="ml-1.5 hover:bg-blue-200 rounded-full p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      value={newArtist}
                      onChange={(e) => setNewArtist(e.target.value)}
                      placeholder="Add artist (e.g., Miles Davis)"
                      className="h-9 rounded-xl"
                      onKeyPress={(e) => e.key === 'Enter' && handleAddArtist()}
                    />
                    <Button
                      onClick={handleAddArtist}
                      size="sm"
                      variant="outline"
                      className="rounded-xl"
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={handleReconnectSpotify}
                    variant="outline"
                    className="rounded-xl"
                  >
                    Refresh Data
                  </Button>
                  <Button
                    onClick={handleDisconnectSpotify}
                    variant="outline"
                    className="rounded-xl text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                  >
                    Disconnect
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 bg-amber-50 rounded-xl border border-amber-200">
                  <p className="text-sm text-amber-900">
                    Connect your Spotify account to get personalized concert recommendations based on your music taste.
                  </p>
                </div>
                <Button
                  onClick={handleReconnectSpotify}
                  className="bg-[#1DB954] hover:bg-[#1ed760] text-white rounded-xl"
                >
                  <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                  </svg>
                  Connect Spotify
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Location Card */}
        <Card className="border-slate-200/60 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-violet-600" />
              Location
            </CardTitle>
            <CardDescription>Set your location for nearby concert searches</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Button
                onClick={handleGetCurrentLocation}
                disabled={isGettingLocation}
                className="w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white rounded-xl h-11"
              >
                {isGettingLocation ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Getting location...
                  </>
                ) : (
                  <>
                    <Navigation className="w-4 h-4 mr-2" />
                    Use Current Location
                  </>
                )}
              </Button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t border-slate-200" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-white px-2 text-slate-500">Or enter manually</span>
                </div>
              </div>

              <form onSubmit={handleSaveLocation} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="city" className="text-sm font-medium text-slate-700">
                      City <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      id="city"
                      value={location.city || ''}
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
                      value={location.country || ''}
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
                    value={location.state || ''}
                    onChange={(e) => setLocation({...location, state: e.target.value})}
                    placeholder="NY"
                    className="mt-1.5 h-11 rounded-xl"
                    required
                  />
                </div>

                <Button
                  type="submit"
                  disabled={isSaving || !location.city || !location.country || !location.state}
                  className="w-full bg-slate-900 hover:bg-slate-800 text-white rounded-xl"
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Location'
                  )}
                </Button>
              </form>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}