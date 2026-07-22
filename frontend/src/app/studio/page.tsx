"use client";

import React, { useState, useEffect } from "react";
import {
  Sparkles,
  Zap,
  Sliders,
  Download,
  Share2,
  Maximize2,
  Minimize2,
  Clock,
  ArrowRight,
  Eye,
  Trash2,
  Grid,
  Settings,
  HelpCircle,
  RefreshCw,
  Image as ImageIcon,
  CheckCircle2,
  Columns,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Textarea, Label, Field } from "@/components/ui/input";
import { Progress, Separator } from "@/components/ui/misc";
import { Header } from "@/components/layout/header";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogBody, DialogFooter } from "@/components/ui/dialog";
import { toast } from "sonner";

/* ─── Mock Fashion Image Catalog ────────────────────────────────────────────── */
const MOCK_GENERATIONS = [
  {
    id: "gen-1",
    title: "Neo-Futurism Runway Concept",
    prompt: "High-end avant-garde fashion design concept, futuristic metallic fabrics, neon lighting, dark sleek background, fashion sketch overlay, ultra-detailed 8k render",
    negativePrompt: "blurry, low quality, deformed, hands, low resolution",
    image: "/images/avant_garde.png",
    aspect: "3:4",
    model: "v6.0 (Latest)",
    stylize: 750,
    chaos: 20,
    weird: 50,
    seed: 8439281,
    time: "2 mins ago",
  },
  {
    id: "gen-2",
    title: "Cyberpunk Techwear Jacket",
    prompt: "Futuristic cyberpunk streetwear design flatlay, glowing LED details, premium techwear jacket, high contrast cyber aesthetics, neon purple and fuchsia lighting, dark carbon fiber background",
    negativePrompt: "blurry, out of focus, low contrast",
    image: "/images/cyberpunk.png",
    aspect: "1:1",
    model: "v6.0 (Latest)",
    stylize: 500,
    chaos: 15,
    weird: 0,
    seed: 3928102,
    time: "1 hour ago",
  },
  {
    id: "gen-3",
    title: "Minimalist Luxury Linen",
    prompt: "Minimalist luxury apparel design moodboard, high-end wool and linen fabrics, beige and cream soft earth tones, architectural clean lines, luxury fashion brand aesthetics, editorial studio catalog shot",
    negativePrompt: "bright colors, neon, saturated, messy layout",
    image: "/images/minimalist.png",
    aspect: "16:9",
    model: "v5.2 (Classic)",
    stylize: 250,
    chaos: 5,
    weird: 100,
    seed: 9283104,
    time: "1 day ago",
  },
];

export default function GenerationStudio() {
  // Input states
  const [prompt, setPrompt] = useState(
    "Futuristic techwear parka jacket, glowing cobalt seams, modular straps, hyper-detailed carbon textures, studio catalog lighting, 8k render, photorealistic --ar 3:4 --v 6.0"
  );
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, deformed, hands, low resolution");
  const [aspect, setAspect] = useState("3:4");
  const [model, setModel] = useState("v6.0 (Latest)");
  const [stylize, setStylize] = useState(600);
  const [chaos, setChaos] = useState(10);
  const [weird, setWeird] = useState(250);

  // Live generation states
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStep, setGenStep] = useState("");
  const [history, setHistory] = useState(MOCK_GENERATIONS);

  // Modal / Viewer states
  const [selectedImage, setSelectedImage] = useState<typeof MOCK_GENERATIONS[0] | null>(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [compLeft, setCompLeft] = useState(MOCK_GENERATIONS[0]);
  const [compRight, setCompRight] = useState(MOCK_GENERATIONS[1]);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);

  // Popular helper chips
  const helperTags = [
    "haute couture",
    "cyberpunk techwear",
    "minimalist luxury",
    "metallic drapery",
    "modular pockets",
    "recycled polymers",
    "photorealistic 8k",
    "studio catalog lighting",
  ];

  const handleAddTag = (tag: string) => {
    setPrompt((prev) => {
      const base = prev.split(" --")[0];
      const params = prev.includes(" --") ? " --" + prev.split(" --").slice(1).join(" --") : "";
      return `${base}, ${tag}${params}`;
    });
  };

  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenProgress(0);

    const steps = [
      { text: "Parsing prompt tokens...", val: 15 },
      { text: "Crawling vector DB weights...", val: 40 },
      { text: "Initializing latent space diffusion...", val: 65 },
      { text: "De-noising render passes...", val: 90 },
      { text: "Compiling upscale matrix...", val: 100 },
    ];

    for (const step of steps) {
      setGenStep(step.text);
      // Fast incremental progression
      let currentVal = genProgress;
      while (currentVal < step.val) {
        currentVal += Math.floor(Math.random() * 4) + 1;
        if (currentVal > step.val) currentVal = step.val;
        setGenProgress(currentVal);
        await new Promise((resolve) => setTimeout(resolve, 80));
      }
    }

    // Pick a random mock catalog image to simulate successful render
    const randMock = MOCK_GENERATIONS[Math.floor(Math.random() * MOCK_GENERATIONS.length)];
    const newGen = {
      id: `gen-${Date.now()}`,
      title: prompt.split(",")[0].slice(0, 30) || "Neural Studio Render",
      prompt,
      negativePrompt,
      image: randMock.image,
      aspect,
      model,
      stylize,
      chaos,
      weird,
      seed: Math.floor(Math.random() * 9000000) + 1000000,
      time: "Just now",
    };

    setHistory((prev) => [newGen, ...prev]);
    toast.success("Fashion specification rendering successfully resolved!");
    setIsGenerating(false);
  };

  const handleDownload = (imgUrl: string) => {
    toast.success("Downloading high-resolution design file package...");
    // Mock download trigger
    const link = document.createElement("a");
    link.href = imgUrl;
    link.download = "fashion_design_studio.png";
    link.click();
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast.success("Studio coordinate link copied to clipboard!");
  };

  // Drag handlers for the comparison slider
  const handleTouchMove = (e: React.TouchEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const pos = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pos);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingSlider) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pos = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pos);
  };

  return (
    <>
      <Header title="Neural Generation Studio" description="Midjourney-style high-fidelity fashion rendering sandbox" />

      <div className="px-6 py-8 space-y-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* ─── LEFT: CONTROLS & PROMPT EDITOR (lg:col-span-5) ────────────────── */}
          <div className="lg:col-span-5 space-y-6">
            <Card variant="default">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-primary" />
                  Studio Specifications
                </CardTitle>
                <CardDescription>Configure neural generation tokens and weights.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                {/* Large Prompt Editor */}
                <Field label="Aesthetic Prompt Template">
                  <Textarea
                    placeholder="Type raw aesthetic variables..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={4}
                    className="font-mono text-xs leading-normal scrollbar-thin"
                  />
                </Field>

                {/* Popular helpers helper chips */}
                <div className="space-y-1.5">
                  <span className="text-[10px] text-overline text-foreground-subtle block">Aesthetic modifiers</span>
                  <div className="flex flex-wrap gap-1.5">
                    {helperTags.map((tag) => (
                      <button
                        key={tag}
                        type="button"
                        onClick={() => handleAddTag(tag)}
                        className="px-2 py-1 rounded-lg border border-border bg-surface-3 hover:bg-surface-2 text-[10px] text-foreground-muted hover:text-foreground transition-all"
                      >
                        +{tag}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Negative Prompt */}
                <Field label="Negative Prompts (Exclude)">
                  <Input
                    placeholder="blurry, low contrast, messy folds..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                  />
                </Field>

                {/* Advanced sliders/parameters */}
                <Separator gradient />

                <div className="grid grid-cols-2 gap-4">
                  <Field label="Aspect Ratio">
                    <select
                      value={aspect}
                      onChange={(e) => setAspect(e.target.value)}
                      className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all"
                    >
                      <option value="1:1">1:1 Square</option>
                      <option value="3:4">3:4 Portrait</option>
                      <option value="9:16">9:16 Story</option>
                      <option value="16:9">16:9 Cinema</option>
                    </select>
                  </Field>

                  <Field label="Model Version">
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all"
                    >
                      <option value="v6.0 (Latest)">v6.0 (Latest)</option>
                      <option value="v5.2 (Classic)">v5.2 (Classic)</option>
                      <option value="Niji v6 (Anime)">Niji v6 (Illustrative)</option>
                    </select>
                  </Field>
                </div>

                <div className="space-y-4 pt-2">
                  {/* Stylize */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <Label>Stylize Weight</Label>
                      <span className="font-mono text-primary tabular-nums">{stylize}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1000}
                      value={stylize}
                      onChange={(e) => setStylize(Number(e.target.value))}
                      className="w-full accent-primary bg-surface-3 rounded-full h-1"
                    />
                  </div>

                  {/* Chaos */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <Label>Chaos Weight</Label>
                      <span className="font-mono text-primary tabular-nums">{chaos}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={chaos}
                      onChange={(e) => setChaos(Number(e.target.value))}
                      className="w-full accent-primary bg-surface-3 rounded-full h-1"
                    />
                  </div>

                  {/* Weirdness */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <Label>Weirdness Weight</Label>
                      <span className="font-mono text-primary tabular-nums">{weird}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={3000}
                      value={weird}
                      onChange={(e) => setWeird(Number(e.target.value))}
                      className="w-full accent-primary bg-surface-3 rounded-full h-1"
                    />
                  </div>
                </div>

                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating || !prompt}
                  variant="glow"
                  className="w-full h-10 mt-4"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Running Neural Ingestion...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      Generate Design Mockup
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* ─── RIGHT: PREVIEW & GENERATION WINDOW (lg:col-span-7) ─────────────── */}
          <div className="lg:col-span-7 space-y-6">
            {/* Live Progress Overlay */}
            {isGenerating && (
              <Card variant="glass" className="border-primary/30 animate-glow-pulse">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-bold text-foreground">
                      Synthesizing Neural Layers...
                    </CardTitle>
                    <Badge variant="primary" className="animate-pulse">
                      Active GPU Job
                    </Badge>
                  </div>
                  <CardDescription className="font-mono text-[10px] text-primary mt-1">
                    {genStep}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-2">
                  <Progress value={genProgress} color="primary" showValue label="Upscaling layout matrix" />
                </CardContent>
              </Card>
            )}

            {/* Split Screen Image Comparison Option */}
            <div className="flex items-center justify-between">
              <h3 className="text-overline text-foreground-subtle">Studio workspace & history</h3>
              <Button
                variant={comparisonMode ? "default" : "outline"}
                size="sm"
                onClick={() => setComparisonMode(!comparisonMode)}
                className="gap-1.5"
              >
                <Columns className="h-4 w-4" />
                {comparisonMode ? "Grid Mode" : "Comparison Slider"}
              </Button>
            </div>

            {/* Comparison Slider Mode */}
            {comparisonMode ? (
              <Card variant="glass" padding="none" className="overflow-hidden border border-border">
                <div className="p-4 border-b border-border bg-surface-1 flex justify-between items-center gap-4 text-xs">
                  <div>
                    <span className="text-overline text-foreground-subtle block">Compare parameters</span>
                    <span className="font-bold text-foreground">Slide to inspect style differences</span>
                  </div>
                  <div className="flex gap-2">
                    <select
                      className="bg-surface-2 border border-border p-1.5 rounded-lg text-xs"
                      onChange={(e) => setCompLeft(history.find(h => h.id === e.target.value) || history[0])}
                      value={compLeft.id}
                    >
                      {history.map(h => <option key={h.id} value={h.id}>{h.title}</option>)}
                    </select>
                    <select
                      className="bg-surface-2 border border-border p-1.5 rounded-lg text-xs"
                      onChange={(e) => setCompRight(history.find(h => h.id === e.target.value) || history[1])}
                      value={compRight.id}
                    >
                      {history.map(h => <option key={h.id} value={h.id}>{h.title}</option>)}
                    </select>
                  </div>
                </div>

                <div
                  className="relative aspect-[4/3] w-full select-none overflow-hidden bg-surface-3 cursor-ew-resize"
                  onMouseMove={handleMouseMove}
                  onTouchMove={handleTouchMove}
                  onMouseDown={() => setIsDraggingSlider(true)}
                  onMouseUp={() => setIsDraggingSlider(false)}
                  onMouseLeave={() => setIsDraggingSlider(false)}
                >
                  {/* Right Image (Background) */}
                  <img
                    src={compRight.image}
                    alt="Right Design"
                    className="absolute inset-0 h-full w-full object-cover pointer-events-none"
                  />
                  <div className="absolute right-4 top-4 z-10 glass px-2.5 py-1 rounded text-[10px] text-white">
                    {compRight.title}
                  </div>

                  {/* Left Image (Foregound clipped) */}
                  <div
                    className="absolute inset-y-0 left-0 overflow-hidden pointer-events-none"
                    style={{ width: `${sliderPosition}%` }}
                  >
                    <img
                      src={compLeft.image}
                      alt="Left Design"
                      className="absolute inset-y-0 left-0 h-full w-full object-cover max-w-none pointer-events-none"
                      style={{ width: "100%", height: "100%" }} // Fits parent clip bounds
                    />
                  </div>
                  <div className="absolute left-4 top-4 z-10 glass px-2.5 py-1 rounded text-[10px] text-white pointer-events-none">
                    {compLeft.title}
                  </div>

                  {/* Divider Line & Slider button handle */}
                  <div
                    className="absolute inset-y-0 w-0.5 bg-primary/80 z-20 pointer-events-none"
                    style={{ left: `${sliderPosition}%` }}
                  >
                    <div className="absolute top-1/2 -left-3.5 -translate-y-1/2 h-7 w-7 rounded-full bg-primary border-2 border-background flex items-center justify-center shadow-lg pointer-events-none">
                      <Columns className="h-3.5 w-3.5 text-white" />
                    </div>
                  </div>
                </div>
              </Card>
            ) : (
              /* Grid Generations History Mode */
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {history.map((item) => (
                  <Card key={item.id} variant="glass" padding="none" className="overflow-hidden group hover:border-primary/45 transition-all">
                    <div className="relative aspect-[3/4] bg-surface-3 overflow-hidden">
                      <img src={item.image} alt={item.title} className="object-cover w-full h-full transition-transform duration-500 group-hover:scale-102" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-90" />

                      {/* Header details overlays */}
                      <div className="absolute top-3 left-3 right-3 flex justify-between items-center z-10">
                        <Badge variant="gradient" size="xs">
                          {item.aspect} Ratio
                        </Badge>
                        <span className="text-[10px] text-white/50 font-mono">{item.time}</span>
                      </div>

                      {/* Footer text content overlays */}
                      <div className="absolute bottom-3 left-3 right-3 space-y-2">
                        <div>
                          <span className="text-[10px] text-primary font-bold">{item.model}</span>
                          <h4 className="text-xs font-bold text-white truncate mt-0.5">{item.title}</h4>
                          <p className="text-[10px] text-white/60 italic truncate mt-1">"{item.prompt}"</p>
                        </div>

                        {/* Interactive operations buttons */}
                        <div className="flex gap-1.5 border-t border-white/10 pt-2.5 mt-1">
                          <Button
                            size="icon-xs"
                            variant="glass"
                            onClick={() => setSelectedImage(item)}
                            title="Fullscreen inspect"
                          >
                            <Maximize2 className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            size="icon-xs"
                            variant="glass"
                            onClick={() => handleDownload(item.image)}
                            title="Download package"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            size="icon-xs"
                            variant="glass"
                            onClick={handleShare}
                            title="Share coordinate link"
                          >
                            <Share2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── FULLSCREEN COMPONENT VIEWER DIALOG ────────────────────────────────── */}
      <Dialog open={!!selectedImage} onOpenChange={(o) => !o && setSelectedImage(null)}>
        <DialogContent className="max-w-2xl border border-border p-0 overflow-hidden">
          {selectedImage && (
            <>
              <DialogHeader className="p-4 border-b border-border">
                <DialogTitle className="text-sm font-bold text-foreground truncate">
                  {selectedImage.title}
                </DialogTitle>
                <DialogDescription className="text-xs">
                  Detailed neural configuration sheet logs.
                </DialogDescription>
              </DialogHeader>
              <DialogBody className="p-0">
                <div className="grid grid-cols-1 md:grid-cols-12">
                  {/* Image container */}
                  <div className="md:col-span-6 bg-surface-3 aspect-[3/4]">
                    <img src={selectedImage.image} alt={selectedImage.title} className="object-cover h-full w-full" />
                  </div>

                  {/* Telemetry details panel */}
                  <div className="md:col-span-6 p-5 space-y-4 max-h-[400px] overflow-y-auto">
                    <div className="space-y-1">
                      <span className="text-overline text-foreground-subtle block">Aesthetic Prompt</span>
                      <p className="text-xs font-mono text-foreground leading-normal p-2.5 bg-surface-2 border border-border rounded-lg">
                        "{selectedImage.prompt}"
                      </p>
                    </div>

                    <div className="space-y-1">
                      <span className="text-overline text-foreground-subtle block">Negative tokens</span>
                      <p className="text-xs font-mono text-foreground-muted leading-normal p-2.5 bg-surface-2 border border-border rounded-lg">
                        "{selectedImage.negativePrompt}"
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="p-2.5 bg-surface-2 border border-border rounded-lg">
                        <span className="text-[10px] text-foreground-subtle block">Model Version</span>
                        <span className="font-semibold text-foreground mt-0.5 block">{selectedImage.model}</span>
                      </div>
                      <div className="p-2.5 bg-surface-2 border border-border rounded-lg">
                        <span className="text-[10px] text-foreground-subtle block">Seed coordinate</span>
                        <span className="font-mono text-foreground mt-0.5 block">{selectedImage.seed}</span>
                      </div>
                      <div className="p-2.5 bg-surface-2 border border-border rounded-lg">
                        <span className="text-[10px] text-foreground-subtle block">Stylize weight</span>
                        <span className="font-mono text-foreground mt-0.5 block">{selectedImage.stylize}</span>
                      </div>
                      <div className="p-2.5 bg-surface-2 border border-border rounded-lg">
                        <span className="text-[10px] text-foreground-subtle block">Weirdness weight</span>
                        <span className="font-mono text-foreground mt-0.5 block">{selectedImage.weird}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </DialogBody>
              <DialogFooter className="p-4 border-t border-border">
                <Button variant="ghost" onClick={() => setSelectedImage(null)}>
                  Close
                </Button>
                <Button onClick={() => handleDownload(selectedImage.image)}>
                  <Download className="h-4 w-4" />
                  Download package
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
