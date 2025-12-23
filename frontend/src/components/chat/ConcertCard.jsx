import React from 'react';
import { Calendar, MapPin, Clock, ExternalLink, Music2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { format } from 'date-fns';
import { motion } from 'framer-motion';

export default function ConcertCard({ concert, index }) {
  const {
    name,
    artist,
    venue,
    date,
    time,
    price,
    image,
    ticketUrl,
    genre
  } = concert;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="group relative overflow-hidden rounded-2xl bg-white border border-slate-200/60 shadow-sm hover:shadow-xl transition-all duration-500"
    >
      {/* Image Section */}
      <div className="relative h-40 overflow-hidden">
        {image ? (
          <img 
            src={image} 
            alt={name}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <Music2 className="w-12 h-12 text-white/50" />
          </div>
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
        
        {/* Genre Badge */}
        {genre && (
          <span className="absolute top-3 left-3 px-2.5 py-1 bg-white/90 backdrop-blur-sm rounded-full text-xs font-medium text-slate-700">
            {genre}
          </span>
        )}
        
        {/* Price Badge */}
        {price && (
          <span className="absolute top-3 right-3 px-2.5 py-1 bg-violet-600 rounded-full text-xs font-semibold text-white">
            {price}
          </span>
        )}
      </div>

      {/* Content Section */}
      <div className="p-4 space-y-3">
        <div>
          <h3 className="font-semibold text-slate-900 text-lg leading-tight line-clamp-1 group-hover:text-violet-600 transition-colors">
            {artist || name}
          </h3>
          {artist && name !== artist && (
            <p className="text-sm text-slate-500 mt-0.5 line-clamp-1">{name}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Calendar className="w-4 h-4 text-violet-500 flex-shrink-0" />
            <span>{date ? format(new Date(date), 'EEE, MMM d, yyyy') : 'Date TBA'}</span>
          </div>
          
          {time && (
            <div className="flex items-center gap-2 text-sm text-slate-600">
              <Clock className="w-4 h-4 text-violet-500 flex-shrink-0" />
              <span>{time}</span>
            </div>
          )}
          
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <MapPin className="w-4 h-4 text-violet-500 flex-shrink-0" />
            <span className="line-clamp-1">{venue || 'Venue TBA'}</span>
          </div>
        </div>

        {ticketUrl && (
          <Button 
            asChild
            className="w-full mt-2 bg-slate-900 hover:bg-violet-600 text-white rounded-xl h-10 font-medium transition-colors duration-300"
          >
            <a href={ticketUrl} target="_blank" rel="noopener noreferrer">
              Get Tickets
              <ExternalLink className="w-4 h-4 ml-2" />
            </a>
          </Button>
        )}
      </div>
    </motion.div>
  );
}