"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  Send,
  Loader2,
  Mic,
  Search,
  MessageSquare,
  BookOpen,
  Trash2,
  Image as ImageIcon,
  CheckCircle2,
  ArrowRight,
  ExternalLink,
  ChevronRight,
  User,
  Plus,
  Compass,
} from "lucide-react";
import axios from "axios";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Label, Field } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/misc";
import { Header } from "@/components/layout/header";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogBody } from "@/components/ui/dialog";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/* ─── Types ────────────────────────────────────────────────────────────────── */
interface CitationDoc {
  document: string;
  metadata?: {
    name?: string;
    brand?: string;
    category?: string;
  };
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  citations?: CitationDoc[];
  inlineImage?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  timestamp: string;
}

const SUGGESTIONS = [
  "Explain linen breathability & drape in detail.",
  "Design parameters for organic cotton techwear.",
  "Which luxury brands align with cyber punk?",
];

const PRE_POPULATED_SESSIONS: ChatSession[] = [
  {
    id: "session-1",
    title: "Linen weave properties",
    timestamp: "10 mins ago",
    messages: [
      { id: "m-1", role: "user", content: "Explain linen weaving structures." },
      {
        id: "m-2",
        role: "assistant",
        content: "Linen typically uses a plain weave coordinate. It is structured with loose spacing to optimize ventilation and thermal dispersion.",
        citations: [
          {
            document: "Linen (Flax) Natural Fiber Technical Specification Sheet.\nStructure: Plain weave 1/1, Thread count: 40x40 per inch.\nThermal conductivity: 0.05 W/m·K. Moisture regain: 12%.",
            metadata: { name: "Linen_Specs_v2.pdf", brand: "Textile Standard", category: "Natural Fibers" },
          },
          {
            document: "Summer Weave Analysis: Breathability metrics across Flax, Hemp, and Organic Cotton weaves in high humidity conditions.",
            metadata: { name: "Weave_Research_2025.pdf", category: "Weave Mechanics" },
          },
        ],
        inlineImage: "/images/minimalist.png",
      },
    ],
  },
  {
    id: "session-2",
    title: "Cyberpunk techwear specs",
    timestamp: "2 hours ago",
    messages: [
      { id: "m-3", role: "user", content: "Suggest fabrics for cyber techwear." },
      {
        id: "m-4",
        role: "assistant",
        content: "We recommend using a 3-layer laminated ripstop nylon with integrated ePTFE membranes, providing a balance between durability and breathability.",
        citations: [
          {
            document: "Techwear Performance Fabrics Compendium.\nMaterial: 3-Layer Ripstop Nylon + ePTFE Membrane.\nWaterproof rating: 20,000mm. Breathability: 15,000g/m²/24h.",
            metadata: { name: "Techwear_Materials_Guide.pdf", brand: "CyberFabric Lab", category: "Performance Tech" },
          },
        ],
        inlineImage: "/images/cyberpunk.png",
      },
    ],
  },
];

export default function QAPage() {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Session & Chat States
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [inputMessage, setInputMessage] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  // UI States
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [citationModalDoc, setCitationModalDoc] = useState<CitationDoc | null>(null);

  // Sync session logs with localStorage
  useEffect(() => {
    const saved = localStorage.getItem("fashion_chat_sessions");
    if (saved) {
      const parsed = JSON.parse(saved);
      setSessions(parsed);
      if (parsed.length > 0) setActiveSessionId(parsed[0].id);
    } else {
      setSessions(PRE_POPULATED_SESSIONS);
      setActiveSessionId(PRE_POPULATED_SESSIONS[0].id);
      localStorage.setItem("fashion_chat_sessions", JSON.stringify(PRE_POPULATED_SESSIONS));
    }
  }, []);

  const saveSessionsToLocal = (updated: ChatSession[]) => {
    setSessions(updated);
    localStorage.setItem("fashion_chat_sessions", JSON.stringify(updated));
  };

  // Scroll to bottom helper
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [sessions, activeSessionId, isThinking]);

  // Active Session helper
  const getActiveSession = () => {
    return sessions.find((s) => s.id === activeSessionId) || sessions[0];
  };

  const handleCreateSession = () => {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      title: "New fashion inquiry",
      timestamp: "Just now",
      messages: [],
    };
    const updated = [newSession, ...sessions];
    saveSessionsToLocal(updated);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter((s) => s.id !== id);
    saveSessionsToLocal(updated);
    if (activeSessionId === id && updated.length > 0) {
      setActiveSessionId(updated[0].id);
    }
  };

  const handleSendMessage = async (textToSend: string) => {
    if (!textToSend.trim() || isThinking) return;

    const currentSession = getActiveSession();
    if (!currentSession) return;

    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: textToSend,
    };

    // Update session title on first message
    const updatedMessages = [...currentSession.messages, userMsg];
    let newTitle = currentSession.title;
    if (currentSession.messages.length === 0) {
      newTitle = textToSend.slice(0, 24) + (textToSend.length > 24 ? "..." : "");
    }

    const updatedSessions = sessions.map((s) =>
      s.id === currentSession.id
        ? { ...s, title: newTitle, messages: updatedMessages, timestamp: "Just now" }
        : s
    );
    saveSessionsToLocal(updatedSessions);
    setInputMessage("");
    setIsThinking(true);

    try {
      // Connect to FastAPI query endpoint
      const response = await axios.post("http://localhost:8000/api/v1/query", {
        query: textToSend,
      });

      const data = response.data;
      const assistantResponse: string = data.response || "No response generated.";
      const docs = data.source_documents || [];

      // Mapped image trigger: if output contains jacket or luxury, render inline premium system images
      let inlineImg: string | undefined;
      if (textToSend.toLowerCase().includes("jacket") || assistantResponse.toLowerCase().includes("jacket")) {
        inlineImg = "/images/cyberpunk.png";
      } else if (textToSend.toLowerCase().includes("luxury") || assistantResponse.toLowerCase().includes("luxury") || assistantResponse.toLowerCase().includes("linen")) {
        inlineImg = "/images/minimalist.png";
      }

      // Prepare streaming response item
      const assistantMsgId = `msg-${Date.now()}-assist`;
      const finalMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        isStreaming: true,
        citations: docs,
        inlineImage: inlineImg,
      };

      // Push initial empty assistant message
      const withAssistMsg = [...updatedMessages, finalMsg];
      const withAssistSessions = sessions.map((s) =>
        s.id === currentSession.id ? { ...s, messages: withAssistMsg } : s
      );
      saveSessionsToLocal(withAssistSessions);

      // Simulate ChatGPT-style typewriter streaming
      const chars = assistantResponse.split("");
      let currentString = "";
      for (let i = 0; i < chars.length; i++) {
        currentString += chars[i];
        await new Promise((resolve) => setTimeout(resolve, 8)); // Typwriter speed
        
        // Update content incrementally
        const streamUpdate = withAssistMsg.map((m) =>
          m.id === assistantMsgId ? { ...m, content: currentString } : m
        );
        const streamSessions = sessions.map((s) =>
          s.id === currentSession.id ? { ...s, messages: streamUpdate } : s
        );
        setSessions(streamSessions);
      }

      // Finish streaming state
      const streamFinished = withAssistMsg.map((m) =>
        m.id === assistantMsgId ? { ...m, content: assistantResponse, isStreaming: false } : m
      );
      const finishedSessions = sessions.map((s) =>
        s.id === currentSession.id ? { ...s, messages: streamFinished } : s
      );
      saveSessionsToLocal(finishedSessions);
    } catch (err) {
      toast.error("RAG connection failed. Rendering mock intelligence.");
      // Fallback mock streaming
      const mockReply = `Neural system failed to reach the FastAPI endpoint directly. 
      Here is a simulated response regarding: *"${textToSend}"*.
      - **Aesthetic Recommendation**: Align with earth tones and organic bamboo textures.
      - **Weave Profile**: High density plain weaves optimized for tensile strength.`;

      const assistantMsgId = `msg-${Date.now()}-assist`;
      const finalMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: mockReply,
        isStreaming: false,
        citations: [
          {
            document: `ChromaDB RAG Knowledge Base Entry:\nSubject: ${textToSend}\nClassification: High-grade Textile Fiber & Aesthetic Specs.\nConfidence Score: 0.94`,
            metadata: { name: "Fashion_RAG_Index_v1.pdf", brand: "AI Fashion Engine", category: "Knowledge Base" },
          },
        ],
        inlineImage: textToSend.toLowerCase().includes("jacket") ? "/images/cyberpunk.png" : "/images/minimalist.png",
      };

      const fallbackMsg = [...updatedMessages, finalMsg];
      const fallbackSessions = sessions.map((s) =>
        s.id === currentSession.id ? { ...s, messages: fallbackMsg } : s
      );
      saveSessionsToLocal(fallbackSessions);
    } finally {
      setIsThinking(false);
    }
  };

  const handleVoiceToggle = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      toast.success("Voice recognition active. Listening...");
    } else {
      toast.success("Voice transcript saved.");
      setInputMessage("Explain cotton quality metrics.");
    }
  };

  // Filter sessions by search term
  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const activeSession = getActiveSession();

  // Custom client-side markdown formatter helper
  const renderMessageContent = (content: string) => {
    // Process bullet points and bold headers
    const lines = content.split("\n");
    return lines.map((line, idx) => {
      let trimmed = line.trim();
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        return (
          <li key={idx} className="list-disc pl-2 ml-4 mt-1 text-xs text-foreground-muted leading-relaxed">
            {trimmed.slice(2)}
          </li>
        );
      }
      // Process bold tags **bold**
      if (trimmed.includes("**")) {
        const parts = trimmed.split("**");
        return (
          <p key={idx} className="text-xs text-foreground leading-relaxed mt-1">
            {parts.map((p, pIdx) => (pIdx % 2 === 1 ? <strong key={pIdx} className="text-primary font-bold">{p}</strong> : p))}
          </p>
        );
      }
      return (
        <p key={idx} className="text-xs text-foreground-muted leading-relaxed mt-1">
          {line}
        </p>
      );
    });
  };

  return (
    <>
      <Header title="RAG Fashion Assistant" description="Context-aware natural language assistant indexed over ChromaDB" />

      <div className="px-6 py-8 max-w-7xl">
        <div className="rounded-2xl border border-border bg-surface-2 overflow-hidden h-[calc(100vh-12rem)] flex">
          {/* ─── LEFT PANEL: CHAT HISTORY & SEARCH (width: 64) ───────────────── */}
          <div className="w-64 border-r border-border bg-surface-1 flex flex-col justify-between shrink-0 hidden md:flex">
            {/* Search and New Session trigger */}
            <div className="p-4 border-b border-border space-y-3">
              <Button onClick={handleCreateSession} className="w-full gap-1.5 h-9" variant="glass">
                <Plus className="h-4 w-4" />
                New Inquiry
              </Button>

              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-foreground-subtle" />
                <Input
                  placeholder="Filter inquiries..."
                  className="pl-8 h-8.5 text-xs rounded-lg"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>

            {/* List log threads */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1.5 scrollbar-thin">
              {filteredSessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => setActiveSessionId(s.id)}
                  className={`group flex items-center justify-between p-2.5 rounded-lg text-xs cursor-pointer transition-all ${
                    activeSession?.id === s.id
                      ? "bg-surface-2 border border-border text-foreground font-semibold"
                      : "text-foreground-muted hover:text-foreground hover:bg-surface-2/40"
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <MessageSquare className="h-3.5 w-3.5 text-foreground-subtle shrink-0" />
                    <span className="truncate">{s.title}</span>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-destructive transition-all shrink-0 ml-1.5"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>

            {/* Ingestion Footer */}
            <div className="p-3 border-t border-border bg-surface-2/30 text-[10px] text-foreground-subtle">
              ChromaDB index crawl: 10.4k docs
            </div>
          </div>

          {/* ─── RIGHT PANEL: MESSAGE DIALOG WINDOW ──────────────────────────── */}
          <div className="flex-1 flex flex-col justify-between bg-surface-2/50 relative">
            {/* Conversations list container */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
              {activeSession && activeSession.messages.length === 0 ? (
                /* Empty state screen with Suggestions */
                <div className="h-full flex flex-col items-center justify-center text-center space-y-6 max-w-lg mx-auto">
                  <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/25">
                    <Sparkles className="h-6 w-6 text-white animate-pulse" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-heading-md text-foreground">Fashion Neural Assistant</h3>
                    <p className="text-body-xs text-foreground-muted leading-relaxed">
                      Ask about fiber coordinates, fabric composition data sheets, seasonal guidelines,
                      or target brand profiles.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-2.5 w-full pt-4">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        onClick={() => handleSendMessage(s)}
                        className="p-3 rounded-xl border border-border bg-surface-2 hover:bg-surface-3 text-xs text-left text-foreground-muted hover:text-foreground flex items-center justify-between transition-all group"
                      >
                        <span>{s}</span>
                        <ChevronRight className="h-3.5 w-3.5 text-foreground-subtle group-hover:translate-x-0.5 transition-transform" />
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                /* Messages rendering thread */
                activeSession && (
                  <div className="space-y-5">
                    {activeSession.messages.map((m) => (
                      <div
                        key={m.id}
                        className={`flex gap-4 max-w-3xl ${
                          m.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                        }`}
                      >
                        {/* Avatar */}
                        <Avatar size="sm" className="shrink-0 mt-1">
                          <AvatarFallback className="text-[10px]">
                            {m.role === "user" ? "JD" : "AI"}
                          </AvatarFallback>
                        </Avatar>

                        {/* Speech Bubble */}
                        <div className="space-y-3 flex-1 min-w-0">
                          <div
                            className={`p-4 rounded-2xl border text-xs leading-relaxed ${
                              m.role === "user"
                                ? "bg-primary border-primary text-white shadow-ds-sm rounded-tr-none"
                                : "bg-surface-2 border-border text-foreground shadow-ds-xs rounded-tl-none"
                            }`}
                          >
                            <div className="space-y-2">{renderMessageContent(m.content)}</div>

                            {/* Render Inline Fashion Image Card */}
                            {m.inlineImage && (
                              <div className="mt-3.5 max-w-sm rounded-lg overflow-hidden border border-white/10 aspect-[4/3] bg-surface-3">
                                <img src={m.inlineImage} alt="Neural Preview" className="object-cover h-full w-full" />
                              </div>
                            )}
                          </div>

                          {/* Citations references list */}
                          {m.role === "assistant" && m.citations && m.citations.length > 0 && (
                            <div className="flex flex-wrap items-center gap-1.5 text-[9px] text-foreground-subtle pl-2">
                              <BookOpen className="h-3.5 w-3.5 text-primary shrink-0" />
                              <span>RAG Sources matched:</span>
                              {m.citations.slice(0, 3).map((cit, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => setCitationModalDoc(cit)}
                                  className="px-2 py-0.5 rounded border border-border bg-surface-3 hover:bg-surface-1 hover:text-foreground text-foreground-muted font-mono transition-all flex items-center gap-1 truncate max-w-[110px]"
                                >
                                  {cit.metadata?.name || cit.metadata?.brand || `Doc-${idx + 1}`}
                                  <ExternalLink className="h-2 w-2" />
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}

                    {/* Loader block for thinking state */}
                    {isThinking && (
                      <div className="flex gap-4 max-w-2xl">
                        <Avatar size="sm" className="shrink-0 mt-1">
                          <AvatarFallback className="text-[10px]">AI</AvatarFallback>
                        </Avatar>
                        <div className="flex items-center gap-2 text-xs text-foreground-subtle">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          <span>Searching fashion knowledge index...</span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              )}
              {/* Ref to anchor bottom scrolling */}
              <div ref={scrollRef} />
            </div>

            {/* Input field wrapper bar */}
            <div className="p-4 border-t border-border bg-surface-1/90 backdrop-blur-xl">
              <div className="max-w-3xl mx-auto flex items-center gap-2 relative">
                {/* Voice toggle trigger */}
                <Button
                  onClick={handleVoiceToggle}
                  variant={isRecording ? "default" : "ghost"}
                  size="icon"
                  className={cn(
                    "h-9 w-9 rounded-xl shrink-0 text-foreground-muted",
                    isRecording ? "animate-pulse border-red-500/25 bg-red-500/10 text-red-500 hover:bg-red-500/15" : ""
                  )}
                  title="Voice dictation placeholder"
                >
                  <Mic className="h-4.5 w-4.5" />
                </Button>

                {/* Main typing Input */}
                <input
                  placeholder="Ask a fashion query (e.g. explain fabric qualities, map brands)..."
                  className="flex-1 h-9 px-4 rounded-xl text-xs bg-surface-2 border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all min-w-0"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSendMessage(inputMessage)}
                  disabled={isThinking}
                />

                {/* Send action */}
                <Button
                  onClick={() => handleSendMessage(inputMessage)}
                  disabled={isThinking || !inputMessage.trim()}
                  size="icon"
                  className="h-9 w-9 rounded-xl shrink-0"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── CITATION SOURCE DETAIL VIEW DIALOG ────────────────────────────────── */}
      <Dialog open={!!citationModalDoc} onOpenChange={(o) => !o && setCitationModalDoc(null)}>
        <DialogContent className="max-w-lg border border-border p-0 overflow-hidden">
          {citationModalDoc && (
            <>
              <DialogHeader className="p-4 border-b border-border">
                <DialogTitle className="text-sm font-bold text-foreground">
                  Reference Source Document
                </DialogTitle>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {citationModalDoc.metadata?.brand && (
                    <Badge variant="primary" size="xs">Brand: {citationModalDoc.metadata.brand}</Badge>
                  )}
                  {citationModalDoc.metadata?.category && (
                    <Badge variant="outline" size="xs">Category: {citationModalDoc.metadata.category}</Badge>
                  )}
                </div>
              </DialogHeader>
              <DialogBody className="p-5 max-h-[300px] overflow-y-auto">
                <p className="text-xs text-foreground leading-relaxed font-mono bg-surface-2 p-3 rounded-lg border border-border whitespace-pre-wrap">
                  {citationModalDoc.document}
                </p>
              </DialogBody>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
