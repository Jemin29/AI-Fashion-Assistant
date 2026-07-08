"use client";
import * as React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Button,
  Badge,
  Input,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  MessageSquare,
  Send,
  Mic,
  Plus,
  Compass,
  CornerDownLeft,
  Sparkles,
  BookOpen,
  Info,
  Clock,
  MoreHorizontal,
  ChevronRight,
  Clipboard,
  Check
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Citation {
  id: string;
  source: string;
  snippet: string;
  score: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  images?: { gradient: string; label: string; emoji: string }[];
  isStreaming?: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  date: string;
}

const CHAT_STARTERS = [
  { label: "🌾 Linen vs Cotton", text: "What makes linen better than cotton in warm climates? Explain fabric weight." },
  { label: "📈 Quiet Luxury style", text: "Provide style guidelines, fabrics, and colors for the quiet luxury trend." },
  { label: "👔 Business Casual", text: "Create a checklist for a minimal business casual capsule wardrobe." },
];

const CITATIONS_DB: Record<string, Citation[]> = {
  linen: [
    { id: "cit-1", source: "Fabric Guidelines (ChromaDB)", snippet: "Linen has high moisture absorption, drying faster than cotton. Perfect for hot climates.", score: 0.96 },
    { id: "cit-2", source: "Summer Styling Rules", snippet: "Earthy linen sets are trending. Recommended weave density: 150-180 GSM.", score: 0.88 },
  ],
  luxury: [
    { id: "cit-3", source: "Luxury Brand Handbook", snippet: "Quiet luxury is characterized by logo-less tailored designs, neutral palettes, and natural fabrics.", score: 0.95 },
    { id: "cit-4", source: "Seasonal Trends Index", snippet: "Camel, ivory, charcoal, and navy remain the core palettes of understated luxury.", score: 0.91 },
  ],
  casual: [
    { id: "cit-5", source: "Capsule Wardrobe Rules", snippet: "A capsule wardrobe contains 12 versatile items: tailored blazer, trousers, linen shirts.", score: 0.93 },
  ],
};

export default function AssistantPage() {
  const [inputValue, setInputValue] = React.useState("");
  const [isTyping, setIsTyping] = React.useState(false);
  const [messages, setMessages] = React.useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Hello! I am your RAG-grounded Fashion Assistant. Ask me about fabric care, trend forecasts, styling advice, or brand details, and I will resolve answers using our vector knowledge index.",
    },
  ]);
  const [chatSessions, setChatSessions] = React.useState<ChatSession[]>([
    { id: "sess-1", title: "Linen vs Cotton Summer Suit", date: "Today" },
    { id: "sess-2", title: "Gucci Color Palettes Analysis", date: "Yesterday" },
    { id: "sess-3", title: "Capsule Wardrobe Rules", date: "3 days ago" },
  ]);
  const [copiedId, setCopiedId] = React.useState<string | null>(null);
  const [isRecording, setIsRecording] = React.useState(false);

  const messagesEndRef = React.useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleCopyCode = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const simulateStreamingResponse = (fullText: string, citations?: Citation[], images?: any[]) => {
    setIsTyping(false);
    const words = fullText.split(" ");
    let currentText = "";
    let wordIdx = 0;

    const messageId = `msg-${Date.now()}`;
    
    // Add empty assistant message
    setMessages((prev) => [
      ...prev,
      { id: messageId, role: "assistant", text: "", citations, images, isStreaming: true },
    ]);

    const interval = setInterval(() => {
      if (wordIdx < words.length) {
        currentText += (wordIdx === 0 ? "" : " ") + words[wordIdx];
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, text: currentText } : msg
          )
        );
        wordIdx++;
      } else {
        clearInterval(interval);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, isStreaming: false } : msg
          )
        );
      }
    }, 45); // Speed of streaming
  };

  const handleSend = (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed) return;

    // 1. Add User Message
    const userMsg: Message = { id: `user-${Date.now()}`, role: "user", text: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsTyping(true);

    // Update session list if new conversation
    if (messages.length === 1) {
      const newSess: ChatSession = {
        id: `sess-${Date.now()}`,
        title: trimmed.slice(0, 24) + (trimmed.length > 24 ? "..." : ""),
        date: "Today",
      };
      setChatSessions((prev) => [newSess, ...prev]);
    }

    // 2. Resolve Mock RAG Answer
    setTimeout(() => {
      const query = trimmed.toLowerCase();
      let responseText = "I ran a vector lookup in our ChromaDB store but found no direct fashion guidelines matches. Based on general fashion principles: natural fibers are always recommended for breathable, editorial styling.";
      let citations: Citation[] = [];
      let images: any[] = [];

      if (query.includes("linen") || query.includes("cotton")) {
        responseText = "Based on Fabric Guidelines matching, **Linen** holds a high moisture absorption rate (up to 20%), drying significantly faster than cotton. In warm climates, linen facilitates immediate heat conductivity. We recommend unstructured suits using earthy colors.\n\nHere is a recommended tailors' setup:\n```css\n.linen-blazer {\n  weight: lightweight-160gsm;\n  lining: unlined-natural-canvas;\n  structure: relaxed-shoulder;\n}\n```";
        citations = CITATIONS_DB.linen;
        images = [
          { label: "Minimal Linen Look", emoji: "🌾", gradient: "from-amber-900 via-stone-900 to-black" },
          { label: "Classic Summer Blazer", emoji: "👔", gradient: "from-neutral-700 via-yellow-950 to-black" },
        ];
      } else if (query.includes("luxury") || query.includes("gucci")) {
        responseText = "Under stated Quiet Luxury trends, visual motifs focus on **camel, ivory, navy, and slate charcoal**. Brand LoRA configurations suggest blending Zara contemporary tailors (60%) with Gucci maximalist texture details (40%) to create understated high-fashion silhouettes without logos.";
        citations = CITATIONS_DB.luxury;
        images = [
          { label: "Camel Coat Concept", emoji: "🧥", gradient: "from-amber-950 via-slate-900 to-black" },
          { label: "Ornate Detail Suit", emoji: "🔱", gradient: "from-indigo-950 via-purple-950 to-black" },
        ];
      } else if (query.includes("casual") || query.includes("capsule")) {
        responseText = "A capsule wardrobe optimizes styling versatility. RAG citations guide a **12-item maximum** grid containing: 1 unstructured wool blazer, 2 tailored stone trousers, 2 linen organic shirts, 1 casual crewneck knit, and 2 luxury sneakers. This supports up to 48 unique style combinations.";
        citations = CITATIONS_DB.casual;
      }

      simulateStreamingResponse(responseText, citations, images);
    }, 1500);
  };

  return (
    <DashboardLayout>
      <div className="h-[calc(100vh-8.5rem)] flex gap-6 relative select-none">
        
        {/* Left Page Sidebar: Conversation Threads */}
        <div className="w-64 glass border border-white/5 rounded-2xl hidden md:flex flex-col overflow-hidden bg-surface-deep/30">
          <div className="p-4 border-b border-white/5 flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full text-xs font-bold"
              leftIcon={<Plus className="w-3.5 h-3.5" />}
              onClick={() => setMessages([{
                id: "welcome",
                role: "assistant",
                text: "Hello! I am your RAG-grounded Fashion Assistant. Ask me about fabric care, trend forecasts, styling advice, or brand details, and I will resolve answers using our vector knowledge index.",
              }])}
            >
              New Thread
            </Button>
          </div>
          <div className="p-2 flex-1 overflow-y-auto flex flex-col gap-1.5">
            <span className="px-2.5 py-1 text-[9px] font-bold text-slate-500 uppercase tracking-wider block">
              Recent Chats
            </span>
            <div className="flex flex-col gap-0.5">
              {chatSessions.map((sess) => (
                <button
                  key={sess.id}
                  className="w-full flex items-center justify-between gap-3 p-2.5 rounded-xl hover:bg-white/5 text-slate-400 hover:text-white transition-colors text-xs text-left"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <MessageSquare className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                    <span className="truncate leading-none">{sess.title}</span>
                  </div>
                  <ChevronRight className="w-3 h-3 text-slate-600 flex-shrink-0" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Central Chat Interface Area */}
        <div className="flex-1 flex flex-col min-w-0 glass border border-white/5 rounded-2xl overflow-hidden bg-black/5 relative">
          
          {/* Top Panel Indicator */}
          <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between bg-black/10">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-bold text-slate-200">Grounded Knowledge Base (ChromaDB)</span>
            </div>
            <Badge variant="new">Fast RAG routing active</Badge>
          </div>

          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
            {messages.map((msg) => {
              const isBot = msg.role === "assistant";
              return (
                <div
                  key={msg.id}
                  className={`flex gap-4 ${isBot ? "items-start" : "items-start flex-row-reverse"}`}
                >
                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm shadow flex-shrink-0 select-none
                    ${isBot ? "bg-gradient-to-tr from-indigo-600 to-purple-600 text-white" : "bg-white/10 text-white"}`}
                  >
                    {isBot ? "AI" : "JD"}
                  </div>

                  {/* Body Balloon */}
                  <div className="flex flex-col gap-2 max-w-[80%]">
                    <div className={`rounded-2xl p-4 border text-xs leading-relaxed
                      ${isBot
                        ? "bg-surface-elevated/40 border-white/5 text-slate-300"
                        : "bg-indigo-600/10 border-indigo-500/20 text-white"
                      }`}
                    >
                      {/* Rendered output - Markdown helper mock */}
                      <div className="whitespace-pre-line">
                        {msg.text}
                      </div>

                      {/* Code Block rendering inside bot message if present */}
                      {msg.text.includes("```css") && (
                        <div className="relative mt-3 rounded-lg overflow-hidden border border-white/5 bg-black/40 font-mono text-[10px]">
                          <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/5 bg-black/20 text-slate-500">
                            <span>CSS Properties</span>
                            <button
                              onClick={() => handleCopyCode(".linen-blazer { weight: 160gsm; }", msg.id)}
                              className="hover:text-white transition-colors"
                            >
                              {copiedId === msg.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Clipboard className="w-3 h-3" />}
                            </button>
                          </div>
                          <pre className="p-3 text-indigo-300 overflow-x-auto">
                            {`.linen-blazer {\n  weight: lightweight-160gsm;\n  lining: unlined-natural-canvas;\n  structure: relaxed-shoulder;\n}`}
                          </pre>
                        </div>
                      )}
                    </div>

                    {/* Grounding Citation cards */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="flex flex-col gap-2 mt-1">
                        <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1">
                          <BookOpen className="w-3 h-3" /> Grounded References
                        </span>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {msg.citations.map((c) => (
                            <div key={c.id} className="p-2 rounded-xl bg-white/2 border border-white/5 text-[10px] flex flex-col gap-0.5">
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-bold text-white truncate">{c.source}</span>
                                <span className="font-mono text-emerald-400 font-bold">{(c.score * 100).toFixed(0)}% match</span>
                              </div>
                              <p className="text-slate-500 leading-relaxed truncate">{c.snippet}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Inline Image Grid responses */}
                    {msg.images && msg.images.length > 0 && (
                      <div className="grid grid-cols-2 gap-3 mt-1 w-72 sm:w-80">
                        {msg.images.map((img, i) => (
                          <div key={i} className="aspect-square rounded-xl relative overflow-hidden border border-white/5 bg-black/10 flex items-center justify-center text-4xl">
                            <div className={`absolute inset-0 bg-gradient-to-br ${img.gradient} opacity-40`} />
                            <span className="z-10 opacity-30 select-none">{img.emoji}</span>
                            <span className="absolute bottom-2 left-2 right-2 text-[9px] font-bold text-white uppercase truncate text-center bg-black/40 py-1 rounded px-1.5">{img.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Bouncing Typing Animation */}
            {isTyping && (
              <div className="flex gap-4 items-start">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white flex items-center justify-center font-bold text-sm shadow flex-shrink-0 select-none animate-pulse">
                  AI
                </div>
                <div className="rounded-2xl p-4 bg-surface-elevated/40 border border-white/5 flex items-center gap-1.5 py-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "200ms" }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "400ms" }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Sticky Bottom input section */}
          <div className="p-4 border-t border-white/5 bg-black/10 flex flex-col gap-3">
            
            {/* Suggested prompts floating chips above */}
            <div className="flex items-center gap-2 flex-wrap">
              <Compass className="w-3.5 h-3.5 text-slate-500" />
              {CHAT_STARTERS.map((starter) => (
                <button
                  key={starter.label}
                  onClick={() => handleSend(starter.text)}
                  className="px-2.5 py-1 rounded-full text-[10px] font-bold text-indigo-400 hover:text-white glass border border-indigo-500/20 hover:border-indigo-500/40 transition-all select-none active:scale-95"
                >
                  {starter.label}
                </button>
              ))}
            </div>

            {/* Main Input Text Box */}
            <div className="relative flex items-center">
              {isRecording && (
                <div className="absolute inset-0 bg-black/90 backdrop-blur-[2px] rounded-2xl flex items-center justify-between px-6 z-30 border border-indigo-500/20">
                  <div className="flex items-center gap-3">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-xs font-bold text-slate-300">Listening to voice input...</span>
                  </div>
                  <div className="flex items-center gap-1.5 h-6">
                    {[0.4, 0.8, 0.5, 0.9, 0.3].map((delay, idx) => (
                      <motion.div
                        key={idx}
                        animate={{ height: ["8px", "20px", "8px"] }}
                        transition={{ duration: 0.8, repeat: Infinity, delay: delay, ease: "easeInOut" }}
                        className="w-1 bg-indigo-500 rounded-full"
                      />
                    ))}
                  </div>
                  <Button
                    variant="outline"
                    size="xs"
                    onClick={() => setIsRecording(false)}
                    className="font-bold text-[10px]"
                  >
                    Cancel
                  </Button>
                </div>
              )}

              <input
                type="text"
                placeholder="Ask about linen fabrics care, streetwear guides, or Gucci styles..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend(inputValue)}
                className="w-full bg-white/5 border border-white/5 hover:border-white/10 focus:border-indigo-500/50 rounded-2xl py-3.5 pl-4 pr-24 text-xs text-white placeholder-slate-500 outline-none transition-all"
              />
              
              <div className="absolute right-2 flex gap-1.5">
                <button
                  onClick={() => setIsRecording(true)}
                  className="p-2 glass rounded-xl border border-white/5 text-slate-400 hover:text-white transition-all active:scale-95"
                  title="Voice input (mock)"
                >
                  <Mic className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => handleSend(inputValue)}
                  disabled={!inputValue.trim()}
                  className="p-2 rounded-xl bg-indigo-600 text-white disabled:opacity-40 hover:bg-indigo-500 transition-all shadow-md shadow-indigo-600/10 flex items-center justify-center active:scale-95"
                  title="Send Message"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

        </div>

      </div>
    </DashboardLayout>
  );
}
