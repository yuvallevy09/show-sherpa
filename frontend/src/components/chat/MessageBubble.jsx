import React from 'react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';
import { cn } from "@/lib/utils";
import { Sparkles } from 'lucide-react';
import ConcertCard from './ConcertCard';

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const hasConcerts = message.concerts && message.concerts.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex gap-3 max-w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
      )}

      <div className={cn(
        "flex flex-col gap-3",
        isUser ? "items-end" : "items-start",
        "max-w-[85%] md:max-w-[75%]"
      )}>
        {/* Message Content */}
        {message.content && (
          <div className={cn(
            "rounded-2xl px-4 py-3",
            isUser 
              ? "bg-gradient-to-br from-violet-600 to-purple-600 text-white rounded-br-md" 
              : "bg-white border border-slate-200/60 text-slate-700 rounded-bl-md shadow-sm"
          )}>
            {isUser ? (
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
            ) : (
              <ReactMarkdown 
                className="text-sm prose prose-sm prose-slate max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                components={{
                  p: ({ children }) => <p className="my-1.5 leading-relaxed">{children}</p>,
                  ul: ({ children }) => <ul className="my-2 ml-4 list-disc space-y-1">{children}</ul>,
                  ol: ({ children }) => <ol className="my-2 ml-4 list-decimal space-y-1">{children}</ol>,
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
                  a: ({ href, children }) => (
                    <a 
                      href={href} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-violet-600 hover:text-violet-700 underline underline-offset-2"
                    >
                      {children}
                    </a>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
        )}

        {/* Concert Cards Grid */}
        {hasConcerts && (
          <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1">
            {message.concerts.map((concert, idx) => (
              <ConcertCard key={idx} concert={concert} index={idx} />
            ))}
          </div>
        )}
      </div>

      {/* User Avatar Placeholder */}
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-slate-200 flex items-center justify-center">
          <span className="text-xs font-semibold text-slate-600">You</span>
        </div>
      )}
    </motion.div>
  );
}