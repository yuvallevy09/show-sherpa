import React, { useState } from 'react';
import { MapPin, Navigation, Loader2, Check } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { api } from '@/api/apiClient';
import { useQueryClient } from '@tanstack/react-query';

export default function LocationSelector({ currentLocation }) {
  const [open, setOpen] = useState(false);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [location, setLocation] = useState({
    city: currentLocation?.city || '',
    country: currentLocation?.country || '',
    state: currentLocation?.state || ''
  });
  const queryClient = useQueryClient();

  const handleGetCurrentLocation = () => {
    setIsGettingLocation(true);
    
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          // In production, you'd reverse geocode these coordinates
          // For now, we'll set a default location
          const mockLocation = {
            city: 'New York',
            country: 'US',
            state: 'NY',
            coordinates: {
              lat: position.coords.latitude,
              lng: position.coords.longitude
            }
          };
          
          setLocation(mockLocation);
          await api.auth.updateMe({ location: mockLocation });
          queryClient.invalidateQueries({ queryKey: ['currentUser'] });
          setIsGettingLocation(false);
          setOpen(false);
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

  const handleSaveLocation = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    
    await api.auth.updateMe({ location });
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    
    setIsSaving(false);
    setOpen(false);
  };

  const displayLocation = currentLocation?.city && currentLocation?.country 
    ? `${currentLocation.city}, ${currentLocation.country}`
    : 'Set location';

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button 
          variant="outline" 
          size="sm"
          className="gap-2 rounded-xl border-slate-200/60 hover:bg-slate-50 hover:border-violet-300 transition-all"
        >
          <MapPin className="w-4 h-4 text-violet-600" />
          <span className="hidden sm:inline text-slate-700">{displayLocation}</span>
        </Button>
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-violet-600" />
            Change Location
          </DialogTitle>
          <DialogDescription>
            Set your location to find concerts near you
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Use Current Location Button */}
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

          {/* Manual Location Form */}
          <form onSubmit={handleSaveLocation} className="space-y-3">
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
                  className="mt-1.5 h-10 rounded-xl"
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
                  className="mt-1.5 h-10 rounded-xl"
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
                className="mt-1.5 h-10 rounded-xl"
                required
              />
            </div>

            <Button
              type="submit"
              disabled={isSaving || !location.city || !location.country || !location.state}
              className="w-full bg-slate-900 hover:bg-slate-800 text-white rounded-xl h-10 mt-4"
            >
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Save Location
                </>
              )}
            </Button>
          </form>
        </div>
      </DialogContent>
    </Dialog>
  );
}