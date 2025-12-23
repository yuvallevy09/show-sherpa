import React, { useState } from 'react';
import { Plus, ChevronDown, Music, X, History, Settings } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { createPageUrl } from '../../utils';
import ConversationItem from './ConversationItem';

export default function Sidebar({ 
  conversations, 
  activeConversationId, 
  onNewConversation, 
  onSelectConversation,
  onDeleteConversation,
  isOpen,
  onClose
}) {
  const [historyExpanded, setHistoryExpanded] = useState(true);

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside className={cn(
        "fixed lg:relative inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200/60 flex flex-col transition-transform duration-300 ease-out",
        "lg:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        {/* Header */}
        <div className="p-4 border-b border-slate-100">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/30">
                <Music className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-slate-900 text-lg leading-none">ShowSherpa</h1>
                <p className="text-xs text-slate-500 mt-0.5">Live music concierge</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden h-8 w-8 text-slate-500"
              onClick={onClose}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* New Conversation Button */}
          <Button
            onClick={onNewConversation}
            className="w-full bg-slate-900 hover:bg-slate-800 text-white rounded-xl h-11 font-medium shadow-sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Conversation
          </Button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Collapsible Header */}
          <button
            onClick={() => setHistoryExpanded(!historyExpanded)}
            className="flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors"
          >
            <div className="flex items-center gap-2 text-sm font-medium text-slate-600">
              <History className="w-4 h-4" />
              Past Conversations
            </div>
            <ChevronDown className={cn(
              "w-4 h-4 text-slate-400 transition-transform duration-200",
              historyExpanded && "rotate-180"
            )} />
          </button>

          {/* Conversations */}
          <AnimatePresence>
            {historyExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex-1 overflow-hidden"
              >
                <ScrollArea className="h-full px-3 pb-4">
                  {conversations.length === 0 ? (
                    <div className="text-center py-8 px-4">
                      <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-3">
                        <MessageSquareIcon className="w-6 h-6 text-slate-400" />
                      </div>
                      <p className="text-sm text-slate-500">No conversations yet</p>
                      <p className="text-xs text-slate-400 mt-1">Start by asking about concerts!</p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {conversations.map((conv) => (
                        <ConversationItem
                          key={conv.id}
                          conversation={conv}
                          isActive={conv.id === activeConversationId}
                          onClick={() => onSelectConversation(conv.id)}
                          onDelete={onDeleteConversation}
                        />
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 space-y-2">
          <Button
            asChild
            variant="ghost"
            className="w-full justify-start rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-50"
          >
            <Link to={createPageUrl('Settings')}>
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Link>
          </Button>
          <div className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-violet-50 to-purple-50 rounded-xl">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-slate-600">Powered by Spotify & Ticketmaster</span>
          </div>
        </div>
      </aside>
    </>
  );
}

// Inline icon to avoid import issues
function MessageSquareIcon({ className }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  );
}