"use client";

import React, { useRef, useState, useEffect } from "react";
import Link from "next/link";
import {
  Sparkles,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Paintbrush,
  Upload,
  Image as ImageIcon,
  Sliders,
  Play,
  RotateCcw,
  Download,
  Share2,
  Eye,
  Settings,
  Trash2,
  RefreshCw,
  Cpu,
  Layers,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Textarea, Label, Field } from "@/components/ui/input";
import { Progress, Separator } from "@/components/ui/misc";
import { Header } from "@/components/layout/header";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogBody, DialogFooter } from "@/components/ui/dialog";
import { toast } from "sonner";

/* ─── Mock Data for Sketch Outputs ─────────────────────────────────────────── */
const MOCK_SKETCH_HISTORY = [
  {
    id: "sk-1",
    sketch: "/images/minimalist.png", // Stand-in for sketch backdrop/canvas preview
    output: "/images/avant_garde.png",
    prompt: "Avant-garde runway coat, structured metallic shoulders, silver lining",
    controlNet: "Canny Edge Detection",
    weight: 0.85,
    time: "3 mins ago",
  },
  {
    id: "sk-2",
    sketch: "/images/cyberpunk.png",
    output: "/images/cyberpunk.png",
    prompt: "Cyberpunk urban combat hoodie, utility straps, waterproof textures",
    controlNet: "Scribble Control",
    weight: 0.95,
    time: "1 hour ago",
  },
];

export default function SketchStudio() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);

  // Drawing states
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushColor, setBrushColor] = useState("#a78bfa"); // Soft violet default
  const [brushSize, setBrushSize] = useState(5);
  const [zoom, setZoom] = useState(1);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);

  // ControlNet settings
  const [controlNetModel, setControlNetModel] = useState("Canny Edge");
  const [controlWeight, setControlWeight] = useState(0.8);
  const [guidanceStart, setGuidanceStart] = useState(0.0);
  const [guidanceEnd, setGuidanceEnd] = useState(1.0);
  const [prompt, setPrompt] = useState("Haute couture drape dress, liquid metal organza texture, futuristic runway catalog --v 6.0");

  // Pipeline states
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [generatedPreview, setGeneratedPreview] = useState<string | null>(null);
  const [history, setHistory] = useState(MOCK_SKETCH_HISTORY);

  // Drag and drop state
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  // Initialize Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Fixed canvas size for high-res drawing coordinates
    canvas.width = 600;
    canvas.height = 450;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    contextRef.current = ctx;

    // Fill background with near-black surface overlay so the drawing is visible
    ctx.fillStyle = "oklch(0.12 0.013 258)"; // Match theme surface-2
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Save initial state to undo stack
    setUndoStack([canvas.toDataURL()]);
  }, []);

  // Update brush specs when states modify
  useEffect(() => {
    if (contextRef.current) {
      contextRef.current.strokeStyle = brushColor;
      contextRef.current.lineWidth = brushSize;
    }
  }, [brushColor, brushSize]);

  // Drawing event handlers
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    contextRef.current.beginPath();
    contextRef.current.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !contextRef.current || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    contextRef.current.lineTo(x, y);
    contextRef.current.stroke();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    // Push new frame to undo stack
    const canvas = canvasRef.current;
    if (canvas) {
      const dataUrl = canvas.toDataURL();
      setUndoStack((prev) => [...prev, dataUrl]);
      setRedoStack([]); // Clear redo stack on new action
    }
  };

  const handleUndo = () => {
    if (undoStack.length <= 1) return; // Keep initial background frame

    const canvas = canvasRef.current;
    const ctx = contextRef.current;
    if (!canvas || !ctx) return;

    const newUndo = [...undoStack];
    const popped = newUndo.pop(); // Remove current frame
    if (popped) setRedoStack((prev) => [popped, ...prev]);

    const prevFrameUrl = newUndo[newUndo.length - 1];
    setUndoStack(newUndo);

    const img = new Image();
    img.src = prevFrameUrl;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    };
  };

  const handleRedo = () => {
    if (redoStack.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = contextRef.current;
    if (!canvas || !ctx) return;

    const nextFrameUrl = redoStack[0];
    setRedoStack((prev) => prev.slice(1));
    setUndoStack((prev) => [...prev, nextFrameUrl]);

    const img = new Image();
    img.src = nextFrameUrl;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    };
  };

  const handleClear = () => {
    const canvas = canvasRef.current;
    const ctx = contextRef.current;
    if (!canvas || !ctx) return;

    ctx.fillStyle = "oklch(0.12 0.013 258)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL();
    setUndoStack((prev) => [...prev, dataUrl]);
    setRedoStack([]);
    toast.success("Canvas reset successfully.");
  };

  // Drag & drop logic
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDragLeave = () => {
    setIsDraggingOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      loadImageToCanvas(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImageToCanvas(file);
  };

  const loadImageToCanvas = (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target?.result as string;
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = contextRef.current;
        if (!canvas || !ctx) return;

        // Draw image keeping aspect ratio
        ctx.fillStyle = "oklch(0.12 0.013 258)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const wrh = img.width / img.height;
        let newWidth = canvas.width;
        let newHeight = newWidth / wrh;
        if (newHeight > canvas.height) {
          newHeight = canvas.height;
          newWidth = newHeight * wrh;
        }
        const xOffset = (canvas.width - newWidth) / 2;
        const yOffset = (canvas.height - newHeight) / 2;

        ctx.drawImage(img, xOffset, yOffset, newWidth, newHeight);

        // Push new action state
        const dataUrl = canvas.toDataURL();
        setUndoStack((prev) => [...prev, dataUrl]);
        setRedoStack([]);
        toast.success("Design catalog image mapped to drawing canvas backdrop.");
      };
    };
    reader.readAsDataURL(file);
  };

  const runSketchGeneration = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenProgress(0);
    setGeneratedPreview(null);

    // Simulate ControlNet RAG generation steps
    for (let progress = 10; progress <= 100; progress += Math.floor(Math.random() * 15) + 5) {
      if (progress > 100) progress = 100;
      setGenProgress(progress);
      await new Promise((resolve) => setTimeout(resolve, 180));
    }

    // Load one of our mock catalog images as rendering output
    const mockOutputs = ["/images/avant_garde.png", "/images/cyberpunk.png", "/images/minimalist.png"];
    const resolvedImage = mockOutputs[Math.floor(Math.random() * mockOutputs.length)];

    setGeneratedPreview(resolvedImage);

    // Add to history stack
    const newHistory = {
      id: `sk-${Date.now()}`,
      sketch: canvasRef.current?.toDataURL() || "/images/minimalist.png",
      output: resolvedImage,
      prompt,
      controlNet: controlNetModel,
      weight: controlWeight,
      time: "Just now",
    };
    setHistory((prev) => [newHistory, ...prev]);

    toast.success("ControlNet rendering solved successfully!");
    setIsGenerating(false);
  };

  const handleDownload = (imgUrl: string) => {
    toast.success("Downloading high-resolution sketch-to-image package...");
    const link = document.createElement("a");
    link.href = imgUrl;
    link.download = "sketch_studio_output.png";
    link.click();
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast.success("Design catalog parameters link copied to clipboard!");
  };

  return (
    <>
      <Header title="Aesthetic Sketch Studio" description="Adobe Firefly-style sketch-to-image ControlNet generation sandbox" />

      <div className="px-6 py-8 space-y-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* ─── LEFT: SKETCHING CANVAS & BRUSHES (lg:col-span-8) ─────────────── */}
          <div className="lg:col-span-8 space-y-4">
            <Card variant="glass" padding="none" className="overflow-hidden border border-border">
              {/* Toolbar */}
              <div className="p-3 border-b border-border bg-surface-1/90 flex flex-wrap items-center justify-between gap-3 text-xs">
                {/* File Upload / Import */}
                <div className="flex items-center gap-1.5">
                  <label className="flex items-center gap-1.5 px-3 py-1.5 h-8 rounded-lg border border-border bg-surface-2 hover:bg-surface-3 transition-all cursor-pointer text-foreground-muted hover:text-foreground">
                    <Upload className="h-3.5 w-3.5" />
                    <span>Upload Image</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                  <Button variant="ghost" size="sm" onClick={handleClear} className="gap-1.5 h-8">
                    <RotateCcw className="h-3.5 w-3.5" />
                    Reset Canvas
                  </Button>
                </div>

                {/* Undo / Redo / Zoom */}
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={handleUndo}
                    disabled={undoStack.length <= 1}
                    title="Undo"
                  >
                    <Undo2 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={handleRedo}
                    disabled={redoStack.length === 0}
                    title="Redo"
                  >
                    <Redo2 className="h-4 w-4" />
                  </Button>
                  <Separator orientation="vertical" className="h-4 mx-1" />
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                    title="Zoom Out"
                  >
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <span className="font-mono text-[10px] w-12 text-center text-foreground-muted">
                    {Math.round(zoom * 100)}%
                  </span>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => setZoom(Math.min(2, zoom + 0.1))}
                    title="Zoom In"
                  >
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </div>

                {/* Brush variables */}
                <div className="flex items-center gap-3">
                  {/* Colors */}
                  <div className="flex gap-1.5">
                    {["#a78bfa", "#f472b6", "#34d399", "#60a5fa", "#ffffff", "#000000"].map((color) => (
                      <button
                        key={color}
                        type="button"
                        onClick={() => setBrushColor(color)}
                        className={`h-5 w-5 rounded-full border transition-all ${
                          brushColor === color
                            ? "border-primary scale-110 ring-2 ring-primary/20"
                            : "border-white/10 hover:scale-105"
                        }`}
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>

                  {/* Size slider */}
                  <div className="flex items-center gap-1.5">
                    <Paintbrush className="h-3.5 w-3.5 text-foreground-muted" />
                    <input
                      type="range"
                      min={1}
                      max={20}
                      value={brushSize}
                      onChange={(e) => setBrushSize(Number(e.target.value))}
                      className="w-16 accent-primary h-1 bg-surface-3 rounded-full cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Drawing Board Canvas Area */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative flex items-center justify-center p-6 bg-surface-3/30 overflow-auto min-h-[480px] transition-colors ${
                  isDraggingOver ? "bg-primary/5 border-2 border-dashed border-primary" : ""
                }`}
              >
                {/* Drag drop help overlay */}
                {isDraggingOver && (
                  <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm gap-2">
                    <ImageIcon className="h-8 w-8 text-primary animate-pulse" />
                    <p className="text-xs font-semibold text-foreground">Drop image coordinates here</p>
                  </div>
                )}

                {/* Main HTML5 draw board */}
                <div
                  style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
                  className="transition-transform duration-75 shadow-ds-xl rounded-xl overflow-hidden border border-border bg-surface-2"
                >
                  <canvas
                    ref={canvasRef}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    className="block cursor-crosshair"
                  />
                </div>
              </div>
            </Card>

            {/* Generated results rendering preview window */}
            {generatedPreview && (
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-sm font-bold text-foreground">Generated Design Preview</CardTitle>
                  <CardDescription>Synthesized outline output matching active ControlNet vectors.</CardDescription>
                </CardHeader>
                <CardContent className="flex justify-center p-6 bg-surface-3/20 rounded-xl border border-border">
                  <div className="rounded-xl overflow-hidden border border-border aspect-[4/3] max-w-lg bg-surface-2">
                    <img src={generatedPreview} alt="Output Design" className="object-cover h-full w-full" />
                  </div>
                </CardContent>
                <CardFooter className="justify-end gap-2 border-t border-border pt-4 mt-2">
                  <Button size="sm" variant="outline" onClick={() => handleDownload(generatedPreview)}>
                    <Download className="h-4 w-4" />
                    Download File
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleShare}>
                    <Share2 className="h-4 w-4" />
                    Share
                  </Button>
                </CardFooter>
              </Card>
            )}
          </div>

          {/* ─── RIGHT: CONTROLNET MODEL SETTINGS (lg:col-span-4) ─────────────── */}
          <div className="lg:col-span-4 space-y-6">
            <Card variant="default">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-primary" />
                  ControlNet Settings
                </CardTitle>
                <CardDescription>Configure conditioning scale and mapping.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                {/* Select ControlNet type */}
                <Field label="Conditioning Model (ControlNet)">
                  <select
                    value={controlNetModel}
                    onChange={(e) => setControlNetModel(e.target.value)}
                    className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all"
                  >
                    <option value="Canny Edge">Canny Edge Detection</option>
                    <option value="Scribble Control">Scribble Control (Freehand)</option>
                    <option value="Depth Mapping">Depth Map Alignment</option>
                    <option value="OpenPose">OpenPose (Pose Detection)</option>
                  </select>
                </Field>

                {/* Scribing Prompt */}
                <Field label="Conditioning Prompt Guidance">
                  <Textarea
                    placeholder="Describe how the sketch outlines map to fabrics, details, lighting..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={4}
                  />
                </Field>

                {/* Control Scale */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Control Weight Scale</Label>
                    <span className="font-mono text-primary font-bold">{controlWeight}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={2.0}
                    step={0.05}
                    value={controlWeight}
                    onChange={(e) => setControlWeight(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {/* Guidance Start */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <Label>Guidance Start</Label>
                      <span className="font-mono text-primary font-bold">{guidanceStart}</span>
                    </div>
                    <input
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.05}
                      value={guidanceStart}
                      onChange={(e) => setGuidanceStart(Number(e.target.value))}
                      className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                    />
                  </div>

                  {/* Guidance End */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <Label>Guidance End</Label>
                      <span className="font-mono text-primary font-bold">{guidanceEnd}</span>
                    </div>
                    <input
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.05}
                      value={guidanceEnd}
                      onChange={(e) => setGuidanceEnd(Number(e.target.value))}
                      className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                    />
                  </div>
                </div>

                {isGenerating && (
                  <div className="space-y-2 pt-2 animate-glow-pulse">
                    <Progress value={genProgress} color="primary" showValue label="Running ControlNet..." />
                  </div>
                )}

                <Button
                  onClick={runSketchGeneration}
                  disabled={isGenerating || !prompt}
                  variant="glow"
                  className="w-full h-10 mt-4"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Running Neural Solver...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      Generate from Sketch
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* History stack list */}
            <div>
              <h3 className="text-overline text-foreground-subtle mb-3">Ingested history logs</h3>
              <div className="space-y-3">
                {history.map((h) => (
                  <Card key={h.id} variant="glass" padding="sm" className="flex items-center gap-3">
                    <div className="h-10 w-10 shrink-0 rounded-lg overflow-hidden border border-border bg-surface-3">
                      <img src={h.sketch} alt="Sketch" className="object-cover h-full w-full" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between text-[9px] text-foreground-subtle">
                        <span>{h.controlNet}</span>
                        <span>{h.time}</span>
                      </div>
                      <h4 className="text-xs font-semibold text-foreground truncate mt-0.5">"{h.prompt}"</h4>
                    </div>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => setGeneratedPreview(h.output)}
                      title="Load render result"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
