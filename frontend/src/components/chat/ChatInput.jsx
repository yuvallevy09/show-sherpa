import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function ChatInput({ onSend, isLoading, placeholder }) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px';
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative flex items-end gap-2 p-2 bg-white rounded-2xl border border-slate-200 shadow-lg shadow-slate-200/50 focus-within:border-violet-300 focus-within:ring-4 focus-within:ring-violet-100 transition-all duration-300">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Ask about concerts near you..."}
          disabled={isLoading}
          rows={1}
          className={cn(
            "flex-1 resize-none border-0 bg-transparent px-3 py-2.5 text-sm text-slate-700 placeholder:text-slate-400",
            "focus:outline-none focus:ring-0",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            "min-h-[44px] max-h-[150px]"
          )}
        />
        
        <Button
          type="submit"
          disabled={!message.trim() || isLoading}
          size="icon"
          className={cn(
            "h-10 w-10 rounded-xl flex-shrink-0 transition-all duration-300",
            message.trim() && !isLoading
              ? "bg-gradient-to-br from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 shadow-lg shadow-violet-500/30"
              : "bg-slate-100 text-slate-400"
          )}
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
      
      <p className="text-xs text-slate-400 text-center mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </form>
  );
}