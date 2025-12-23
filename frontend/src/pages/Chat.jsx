import React, { useState, useEffect, useRef } from 'react';
import { api } from '@/api/apiClient';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Menu, Loader2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { motion, AnimatePresence } from 'framer-motion';

import Sidebar from '@/components/chat/Sidebar';
import MessageBubble from '@/components/chat/MessageBubble';
import ChatInput from '@/components/chat/ChatInput';
import WelcomeScreen from '@/components/chat/WelcomeScreen';
import SpotifyConnect from '@/components/onboarding/SpotifyConnect';
import LocationSelector from '@/components/chat/LocationSelector';

function isConcertQuery(userMessage) {
  const lower = String(userMessage || "").toLowerCase();
  return (
    lower.includes('concert') ||
    lower.includes('show') ||
    lower.includes('gig') ||
    lower.includes('live') ||
    lower.includes('near me') ||
    lower.includes('weekend') ||
    lower.includes('tickets') ||
    lower.includes('events')
  );
}

function isoUtcNoMs(d) {
  const x = new Date(d);
  x.setMilliseconds(0);
  return x.toISOString();
}

async function generateGroundedResponse(userMessage, currentUser) {
  if (!isConcertQuery(userMessage)) {
    return {
      content: "I can help you find **real, bookable concerts** using Ticketmaster.\n\nTry:\n- \"Any concerts near me this month?\"\n- \"Shows in NYC this weekend\"\n- \"Find indie rock concerts in Boston\"",
      concerts: []
    };
  }

  const loc = currentUser?.location || {};
  const city = loc.city;
  const countryCode = loc.country;
  const stateCode = loc.state;

  if (!city || !countryCode || !stateCode) {
    return {
      content: "I can search Ticketmaster, but I need your location first. Please click the location pill in the header (top right) and set **City / Country / State**.",
      concerts: []
    };
  }

  const start = new Date();
  const end = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

  try {
    const result = await api.ticketmaster.searchEvents({
      classificationName: "music",
      city,
      countryCode,
      stateCode,
      startDateTime: isoUtcNoMs(start),
      endDateTime: isoUtcNoMs(end),
      sort: "date,asc",
      size: 10
    });

    const events = result?.events || [];
    if (!events.length) {
      return {
        content: `I searched Ticketmaster for music events near **${city}, ${stateCode}** in the next 30 days, but didn’t find any results. Try a nearby city, or broaden the time window.`,
        concerts: []
      };
    }

    return {
      content: `Here are **${events.length}** upcoming music events near **${city}, ${stateCode}** (Ticketmaster):`,
      concerts: events
    };
  } catch (err) {
    return {
      content: `I couldn't fetch events from Ticketmaster right now.\n\n**Reason**: ${err?.message || String(err)}\n\nDouble-check your backend has \`TICKETMASTER_API_KEY\` set and that the backend is running.`,
      concerts: []
    };
  }
}

async function generateAgentResponse(userMessage, history = []) {
  const res = await api.chat.sendMessage(userMessage, history);
  return {
    content: res?.content || "",
    concerts: res?.concerts || []
  };
}

export default function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [currentMessages, setCurrentMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);
  const messagesEndRef = useRef(null);
  const queryClient = useQueryClient();

  // Fetch current user
  const { data: currentUser } = useQuery({
    queryKey: ['currentUser'],
    queryFn: () => api.auth.me(),
    enabled: !checkingAuth,
  });

  // Check if user has connected Spotify
  useEffect(() => {
    const checkSpotifyConnection = async () => {
      try {
        const user = await api.auth.me();
        if (!user.spotify_connected) {
          setShowOnboarding(true);
        }
      } catch (error) {
        console.error('Error checking Spotify connection:', error);
      } finally {
        setCheckingAuth(false);
      }
    };
    checkSpotifyConnection();
  }, []);

  // Fetch conversations
  const { data: conversations = [], isLoading: conversationsLoading } = useQuery({
    queryKey: ['conversations'],
    queryFn: () => api.entities.Conversation.list('-last_message_at'),
  });

  // Create conversation mutation
  const createConversationMutation = useMutation({
    mutationFn: (data) => api.entities.Conversation.create(data),
    onSuccess: (newConversation) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      setActiveConversationId(newConversation.id);
      setCurrentMessages(newConversation.messages || []);
    },
  });

  // Update conversation mutation
  const updateConversationMutation = useMutation({
    mutationFn: ({ id, data }) => api.entities.Conversation.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });

  // Delete conversation mutation
  const deleteConversationMutation = useMutation({
    mutationFn: (id) => api.entities.Conversation.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      if (activeConversationId) {
        setActiveConversationId(null);
        setCurrentMessages([]);
      }
    },
  });

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentMessages]);

  // Load conversation messages when switching
  useEffect(() => {
    if (activeConversationId) {
      const conversation = conversations.find(c => c.id === activeConversationId);
      if (conversation) {
        setCurrentMessages(conversation.messages || []);
      }
    }
  }, [activeConversationId, conversations]);

  const handleNewConversation = () => {
    setActiveConversationId(null);
    setCurrentMessages([]);
    setSidebarOpen(false);
  };

  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
    setSidebarOpen(false);
  };

  const handleDeleteConversation = (id) => {
    deleteConversationMutation.mutate(id);
  };

  const handleSendMessage = async (content) => {
    const userMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    };

    const newMessages = [...currentMessages, userMessage];
    setCurrentMessages(newMessages);
    setIsLoading(true);

    // Prefer LangGraph agent (LLM + Ticketmaster tools). If not configured, fall back to Ticketmaster-only.
    let response;
    try {
      const history = newMessages.slice(-10).map((m) => ({ role: m.role, content: m.content }));
      response = await generateAgentResponse(content, history);
      if (!response?.content) throw new Error("Empty agent response");
    } catch (e) {
      response = await generateGroundedResponse(content, currentUser);
    }
    
    const assistantMessage = {
      role: 'assistant',
      content: response.content,
      concerts: response.concerts,
      timestamp: new Date().toISOString()
    };

    const finalMessages = [...newMessages, assistantMessage];
    setCurrentMessages(finalMessages);
    setIsLoading(false);

    // Save to database
    if (activeConversationId) {
      updateConversationMutation.mutate({
        id: activeConversationId,
        data: {
          messages: finalMessages,
          last_message_at: new Date().toISOString()
        }
      });
    } else {
      // Create new conversation
      const title = content.length > 50 ? content.substring(0, 50) + '...' : content;
      createConversationMutation.mutate({
        title,
        messages: finalMessages,
        last_message_at: new Date().toISOString()
      });
    }
  };

  const handleSuggestionClick = (suggestion) => {
    handleSendMessage(suggestion);
  };

  const handleOnboardingComplete = () => {
    setShowOnboarding(false);
  };

  // Show onboarding if Spotify not connected
  if (checkingAuth) {
    return (
      <div className="flex h-screen items-center justify-center bg-gradient-to-br from-slate-50 via-white to-violet-50/30">
        <Loader2 className="w-8 h-8 text-violet-600 animate-spin" />
      </div>
    );
  }

  if (showOnboarding) {
    return <SpotifyConnect onComplete={handleOnboardingComplete} />;
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-white to-violet-50/30">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewConversation={handleNewConversation}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-slate-200/60 bg-white/80 backdrop-blur-sm">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden h-9 w-9"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="w-5 h-5 text-slate-600" />
          </Button>
          
          <div className="flex-1">
            <h2 className="font-semibold text-slate-900">
              {activeConversationId 
                ? conversations.find(c => c.id === activeConversationId)?.title || 'Conversation'
                : 'New Conversation'
              }
            </h2>
          </div>

          {/* Location Selector */}
          <LocationSelector currentLocation={currentUser?.location} />
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          {currentMessages.length === 0 ? (
            <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
          ) : (
            <ScrollArea className="h-full">
              <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
                <AnimatePresence mode="popLayout">
                  {currentMessages.map((message, index) => (
                    <MessageBubble key={index} message={message} />
                  ))}
                </AnimatePresence>
                
                {/* Loading indicator */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3"
                  >
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                      <Loader2 className="w-4 h-4 text-white animate-spin" />
                    </div>
                    <div className="bg-white border border-slate-200/60 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </motion.div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gradient-to-t from-white via-white to-transparent">
          <div className="max-w-4xl mx-auto">
            <ChatInput 
              onSend={handleSendMessage} 
              isLoading={isLoading}
              placeholder="Ask about concerts, artists, or events..."
            />
          </div>
        </div>
      </main>
    </div>
  );
}