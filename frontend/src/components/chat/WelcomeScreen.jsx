import React from 'react';
import { Music, MapPin, Calendar, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const suggestions = [
  { icon: MapPin, text: "What concerts are happening near me this weekend?" },
  { icon: Music, text: "Find shows from artists similar to my favorites" },
  { icon: Calendar, text: "Any rock concerts in NYC next month?" },
  { icon: Sparkles, text: "Surprise me with something I might like" },
];

export default function WelcomeScreen({ onSuggestionClick }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-violet-500/30">
          <Music className="w-8 h-8 text-white" />
        </div>
        
        <h2 className="text-2xl md:text-3xl font-bold text-slate-900 mb-3">
          Welcome to ShowSherpa
        </h2>
        <p className="text-slate-500 text-lg max-w-md mx-auto leading-relaxed">
          Your personal live-music concierge. Ask me anything about concerts, and I'll find the perfect shows for you.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="w-full max-w-lg"
      >
        <p className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3 text-center">
          Try asking
        </p>
        
        <div className="grid gap-2.5">
          {suggestions.map((suggestion, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
              onClick={() => onSuggestionClick(suggestion.text)}
              className="group flex items-center gap-3 p-4 bg-white border border-slate-200/60 rounded-xl hover:border-violet-300 hover:bg-violet-50/50 transition-all duration-300 text-left shadow-sm hover:shadow-md"
            >
              <div className="w-10 h-10 rounded-lg bg-slate-100 group-hover:bg-violet-100 flex items-center justify-center transition-colors">
                <suggestion.icon className="w-5 h-5 text-slate-500 group-hover:text-violet-600 transition-colors" />
              </div>
              <span className="text-sm text-slate-700 group-hover:text-slate-900 transition-colors">
                {suggestion.text}
              </span>
            </motion.button>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="mt-10 flex items-center gap-6 text-xs text-slate-400"
      >
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded bg-[#1DB954] flex items-center justify-center">
            <svg className="w-2.5 h-2.5 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
            </svg>
          </div>
          <span>Connected to Spotify</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded bg-[#026CDF] flex items-center justify-center">
            <span className="text-white text-[8px] font-bold">TM</span>
          </div>
          <span>Powered by Ticketmaster</span>
        </div>
      </motion.div>
    </div>
  );
}