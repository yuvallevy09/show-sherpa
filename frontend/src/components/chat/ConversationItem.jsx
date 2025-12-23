import React from 'react';
import { MessageSquare, Trash2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from 'date-fns';
import { motion } from 'framer-motion';

export default function ConversationItem({ conversation, isActive, onClick, onDelete }) {
  const handleDelete = (e) => {
    e.stopPropagation();
    onDelete(conversation.id);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
      className={cn(
        "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-200",
        isActive 
          ? "bg-violet-50 border border-violet-200" 
          : "hover:bg-slate-50 border border-transparent"
      )}
      onClick={onClick}
    >
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center",
        isActive ? "bg-violet-100" : "bg-slate-100 group-hover:bg-slate-200"
      )}>
        <MessageSquare className={cn(
          "w-4 h-4",
          isActive ? "text-violet-600" : "text-slate-500"
        )} />
      </div>

      <div className="flex-1 min-w-0">
        <p className={cn(
          "text-sm font-medium truncate",
          isActive ? "text-violet-900" : "text-slate-700"
        )}>
          {conversation.title || 'New conversation'}
        </p>
        {conversation.last_message_at && (
          <p className="text-xs text-slate-400 mt-0.5">
            {formatDistanceToNow(new Date(conversation.last_message_at), { addSuffix: true })}
          </p>
        )}
      </div>

      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-500 hover:bg-red-50"
        onClick={handleDelete}
      >
        <Trash2 className="w-3.5 h-3.5" />
      </Button>
    </motion.div>
  );
}