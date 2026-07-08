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
  Textarea,
  Select,
  Dialog,
  Slider,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Sparkles,
  Wand2,
  Download,
  Share2,
  Maximize2,
  Trash2,
  Sliders,
  History,
  Info,
  Check,
  ChevronRight,
  Eye
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const STYLE_PRESETS = [
  { name: "Luxury Runway", prompt: "Haute couture silk evening gown with aurora gradient embroidery, chanel style, fashion week runway, dramatic spotlighting, high fashion" },
  { name: "Urban Streetwear", prompt: "Oversized graphic hoodie, cargo pants, chunky sneakers, urban city background, neon signage, retro film grain, editorial photography" },
  { name: "Minimalist Studio", prompt: "Monochromatic ivory tailored suit, clean lines, Celine aesthetic, beige studio background, soft natural lighting, editorial look" },
  { name: "Cyber Techwear", prompt: "Technical jacket with reflective straps, utility chest rig, dark background, futuristic rain-slicked city streets, neon accents, cyberpunk look" },
  { name: "Bohemian Summer", prompt: "Flowing linen dress, floral prints, straw hat, sun-drenched organic textures, golden hour photography, natural shadows" },
];

const AUTOCOMPLETE_MAP: Record<string, string> = {
  "lux": "urious silk evening gown with gold embroidery, runway lighting, chanel style",
  "street": "wear editorial outfit, oversized graphic hoodie, cargo pants, urban photography",
  "mini": "malist tailored suit, clean monochromatic lines, Celine aesthetic, studio background",
  "tech": "wear utility jacket, reflective straps, dark futuristic city streets, cyberpunk",
  "boh": "emian summer linen dress, flowing organic fabrics, golden hour lighting, natural warm tones",
};

interface GenRun {
  id: string;
  prompt: string;
  negativePrompt: string;
  steps: number;
  cfg: number;
  seed: number;
  width: number;
  height: number;
  bgGradient: string;
  emoji: string;
  score: number;
  date: string;
}

const INITIAL_HISTORY: GenRun[] = [
  {
    id: "run-01",
    prompt: "Haute couture silk evening gown with aurora gradient embroidery",
    negativePrompt: "blurry, low quality, text, watermark",
    steps: 30,
    cfg: 7.5,
    seed: 881294,
    width: 1024,
    height: 1024,
    bgGradient: "from-purple-900 via-indigo-900 to-black",
    emoji: "👗",
    score: 0.94,
    date: "10 mins ago",
  },
  {
    id: "run-02",
    prompt: "Oversized graphic hoodie, cargo pants, chunky sneakers",
    negativePrompt: "blurry, low quality, text, watermark",
    steps: 30,
    cfg: 7.5,
    seed: 441029,
    width: 1024,
    height: 1024,
    bgGradient: "from-orange-900 via-red-900 to-black",
    emoji: "🏀",
    score: 0.91,
    date: "1 hour ago",
  },
];

export default function TextToFashionPage() {
  const [prompt, setPrompt] = React.useState("");
  const [negativePrompt, setNegativePrompt] = React.useState("low quality, blurry, watermark, bad proportions");
  const [autocomplete, setAutocomplete] = React.useState("");
  const [isGenerating, setIsGenerating] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [steps, setSteps] = React.useState(30);
  const [cfg, setCfg] = React.useState(7.5);
  const [seed, setSeed] = React.useState(-1);
  const [width, setWidth] = React.useState("1024");
  const [height, setHeight] = React.useState("1024");
  const [history, setHistory] = React.useState<GenRun[]>(INITIAL_HISTORY);
  const [activeOutput, setActiveOutput] = React.useState<GenRun | null>(INITIAL_HISTORY[0]);
  const [isFullscreen, setIsFullscreen] = React.useState(false);
  const [showShareToast, setShowShareToast] = React.useState(false);

  // Autocomplete matching logic
  React.useEffect(() => {
    const term = prompt.toLowerCase();
    if (!term) {
      setAutocomplete("");
      return;
    }
    const matchedKey = Object.keys(AUTOCOMPLETE_MAP).find((key) => term.endsWith(key));
    if (matchedKey) {
      setAutocomplete(AUTOCOMPLETE_MAP[matchedKey]);
    } else {
      setAutocomplete("");
    }
  }, [prompt]);

  const handleApplyAutocomplete = () => {
    if (autocomplete) {
      setPrompt((prev) => prev + autocomplete);
      setAutocomplete("");
    }
  };

  const handleApplyPreset = (p: string) => {
    setPrompt(p);
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setProgress(0);

    // Simulate diffusion steps
    for (let i = 0; i <= 100; i += 10) {
      setProgress(i);
      await new Promise((r) => setTimeout(r, 200));
    }

    const gradients = [
      "from-purple-900 via-indigo-900 to-black",
      "from-orange-900 via-red-900 to-black",
      "from-slate-800 via-gray-900 to-black",
      "from-teal-900 via-cyan-900 to-black",
      "from-amber-900 via-yellow-900 to-black",
    ];

    const emojis = ["👗", "🏀", "🤍", "🥷", "🌾"];

    const newRun: GenRun = {
      id: `run-${Date.now().toString().slice(-4)}`,
      prompt,
      negativePrompt,
      steps,
      cfg,
      seed: seed === -1 ? Math.floor(Math.random() * 1000000) : seed,
      width: parseInt(width),
      height: parseInt(height),
      bgGradient: gradients[Math.floor(Math.random() * gradients.length)],
      emoji: emojis[Math.floor(Math.random() * emojis.length)],
      score: 0.85 + Math.random() * 0.12,
      date: "Just now",
    };

    setHistory((prev) => [newRun, ...prev]);
    setActiveOutput(newRun);
    setIsGenerating(false);
  };

  const handleShare = () => {
    setShowShareToast(true);
    setTimeout(() => setShowShareToast(false), 2000);
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="🎨 Text-to-Fashion Studio"
        badge="SDXL Pipeline"
        description="Convert detailed textual prompts into high-fidelity editorial fashion concepts."
      />
      <Section className="relative">

        {/* Share Toast */}
        <AnimatePresence>
          {showShareToast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-16 left-1/2 -translate-x-1/2 z-50 glass-strong border border-indigo-500/30 px-5 py-2.5 rounded-xl text-xs text-white flex items-center gap-2 shadow-2xl"
            >
              <Check className="w-4 h-4 text-emerald-400" /> Share link copied to clipboard!
            </motion.div>
          )}
        </AnimatePresence>

        {/* Side-by-Side Creative Studio Canvas */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Left Column: Input Prompt Builders */}
          <div className="flex flex-col gap-6">
            
            {/* Prompt presets */}
            <div className="flex flex-col gap-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                Prompt Style Presets
              </label>
              <div className="flex flex-wrap gap-2">
                {STYLE_PRESETS.map((style) => (
                  <button
                    key={style.name}
                    onClick={() => handleApplyPreset(style.prompt)}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold glass border border-white/5 hover:border-indigo-500/30 hover:text-white transition-all text-slate-400 active:scale-95"
                  >
                    {style.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Large prompt editor */}
            <div className="relative">
              <Textarea
                label="Fashion Prompt Description"
                placeholder="Oversized streetwear hoodie, cargo pants, high-top sneakers, editorial styling..."
                rows={5}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="font-mono text-sm leading-relaxed"
                helperText="Type 'lux', 'street', 'mini', 'tech', or 'boh' to trigger autocomplete."
              />
              
              {/* Autocomplete suggestion overlay */}
              <AnimatePresence>
                {autocomplete && (
                  <motion.button
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 5 }}
                    onClick={handleApplyAutocomplete}
                    className="absolute bottom-10 right-4 px-3 py-1.5 rounded-lg bg-indigo-600/90 text-white text-xs font-bold shadow-lg border border-indigo-500/40 hover:bg-indigo-500 flex items-center gap-1.5 select-none"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    Autocomplete: Tab / Click <span className="font-mono text-[10px] bg-black/25 px-1 py-0.5 rounded">{autocomplete.slice(0, 10)}...</span>
                  </motion.button>
                )}
              </AnimatePresence>
            </div>

            {/* Negative prompt panel */}
            <Card className="border-white/5">
              <CardHeader className="p-4 border-b border-white/5">
                <CardTitle className="text-sm font-bold">Negative Prompt Options</CardTitle>
                <CardDescription className="text-[10px]">Visual attributes to exclude from the generation</CardDescription>
              </CardHeader>
              <CardContent className="p-4">
                <Textarea
                  placeholder="Low quality, blurry, text, watermark..."
                  rows={2}
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  className="font-mono text-xs"
                />
              </CardContent>
            </Card>

            {/* Hyper-Parameters details */}
            <Card className="border-white/5">
              <CardHeader className="p-4 border-b border-white/5 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-bold">Inference Parameters</CardTitle>
                  <CardDescription className="text-[10px]">Tweak steps, scales, and resolutions</CardDescription>
                </div>
                <Sliders className="w-4 h-4 text-slate-500" />
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Inference Steps"
                    min={10}
                    max={50}
                    value={steps}
                    onChange={(e: any) => setSteps(parseInt(e.target.value))}
                  />
                  <Slider
                    label="Guidance CFG"
                    min={1}
                    max={15}
                    step={0.5}
                    value={cfg}
                    onChange={(e: any) => setCfg(parseFloat(e.target.value))}
                  />
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <Select
                    label="Width"
                    options={[
                      { value: "512", label: "512px" },
                      { value: "768", label: "768px" },
                      { value: "1024", label: "1024px" },
                    ]}
                    value={width}
                    onChange={setWidth}
                  />
                  <Select
                    label="Height"
                    options={[
                      { value: "512", label: "512px" },
                      { value: "768", label: "768px" },
                      { value: "1024", label: "1024px" },
                    ]}
                    value={height}
                    onChange={setHeight}
                  />
                  <Input
                    label="Seed"
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value))}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Launch CTA */}
            <Button
              variant="primary"
              size="lg"
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              leftIcon={<Wand2 className="w-4 h-4" />}
            >
              {isGenerating ? `Generating Outfit (${progress}%)...` : "Generate Outfit Concepts"}
            </Button>
          </div>

          {/* Right Column: Generation canvas output */}
          <div className="flex flex-col gap-6">
            
            {/* Canvas Output Display */}
            <Card className="border-white/5 relative aspect-square w-full rounded-2xl overflow-hidden flex flex-col items-center justify-center bg-black/10">
              <AnimatePresence mode="wait">
                {isGenerating ? (
                  <motion.div
                    key="generating"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-black/30 backdrop-blur-sm relative"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.15, 1], rotate: [0, 180, 360] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="text-6xl mb-4"
                    >
                      🎨
                    </motion.div>
                    <h3 className="text-white font-bold mb-2">Rendering Design Concept</h3>
                    <p className="text-slate-500 text-xs max-w-xs mb-4">SDXL is executing diffusion processing in Mock Mode.</p>
                    
                    {/* Progress bar overlay */}
                    <div className="w-64 h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-600 transition-all duration-200" style={{ width: `${progress}%` }} />
                    </div>
                  </motion.div>
                ) : activeOutput ? (
                  <motion.div
                    key={activeOutput.id}
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full flex flex-col relative"
                  >
                    {/* Visual canvas grid */}
                    <div className={`flex-1 bg-gradient-to-br ${activeOutput.bgGradient} flex items-center justify-center text-8xl relative overflow-hidden`}>
                      <span className="opacity-25 select-none">{activeOutput.emoji}</span>
                      
                      {/* Action overlays top right */}
                      <div className="absolute top-4 right-4 flex gap-2">
                        <button
                          onClick={() => setIsFullscreen(true)}
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white transition-all"
                          title="Fullscreen View"
                        >
                          <Maximize2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={handleShare}
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white transition-all"
                          title="Share Link"
                        >
                          <Share2 className="w-4 h-4" />
                        </button>
                        <button
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white transition-all"
                          title="Download Design"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                      
                      {/* CLIP score left bottom overlay */}
                      <div className="absolute bottom-4 left-4 glass rounded-xl px-3 py-1.5 border border-white/5 flex items-center gap-1.5 text-xs text-white">
                        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                        CLIP score: <span className="font-mono text-emerald-400 font-bold">{(activeOutput.score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="text-center p-8 text-slate-500">
                    <Info className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                    <p className="text-sm font-semibold">Generate a look to display canvas output.</p>
                  </div>
                )}
              </AnimatePresence>
            </Card>

            {/* Session Generation History */}
            <div className="flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 px-1">
                <History className="w-3.5 h-3.5" /> Session Generation History
              </h4>
              
              <div className="flex flex-col gap-2">
                {history.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => setActiveOutput(run)}
                    className={`w-full flex items-center gap-4 p-3 rounded-xl border text-left transition-all ${
                      activeOutput?.id === run.id
                        ? "border-indigo-500/40 bg-indigo-500/5"
                        : "border-white/5 hover:border-white/10 hover:bg-white/2"
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${run.bgGradient} flex items-center justify-center text-xl flex-shrink-0`}>
                      {run.emoji}
                    </div>
                    <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                      <div className="text-white font-bold text-xs truncate">{run.prompt}</div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-medium">
                        <span>Seed: {run.seed}</span>
                        <span>·</span>
                        <span>CLIP: {(run.score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                  </button>
                ))}
              </div>
            </div>

          </div>

        </div>

      </Section>

      {/* Fullscreen image viewer modal */}
      {activeOutput && (
        <Dialog
          isOpen={isFullscreen}
          onClose={() => setIsFullscreen(false)}
          title="Fullscreen Outfit View"
          size="lg"
          footer={
            <>
              <Button variant="outline" onClick={() => setIsFullscreen(false)}>Close Viewer</Button>
              <Button variant="primary" leftIcon={<Download className="w-4 h-4" />}>Download PNG</Button>
            </>
          }
        >
          <div className="flex flex-col md:flex-row gap-6 items-stretch">
            {/* Visual Canvas left */}
            <div className={`flex-1 min-h-[300px] bg-gradient-to-br ${activeOutput.bgGradient} rounded-2xl flex items-center justify-center text-9xl relative border border-white/5`}>
              <span className="opacity-25 select-none">{activeOutput.emoji}</span>
            </div>

            {/* Metadata right */}
            <div className="w-full md:w-80 flex flex-col gap-4">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Prompt</span>
                <p className="text-white text-xs font-semibold leading-relaxed font-mono">{activeOutput.prompt}</p>
              </div>
              
              <div className="flex flex-col gap-1 border-t border-white/5 pt-3">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Negative Prompt</span>
                <p className="text-slate-400 text-xs font-mono">{activeOutput.negativePrompt}</p>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-white/5 pt-3 text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block font-semibold">Seed</span>
                  <span className="font-mono text-white mt-0.5 block">{activeOutput.seed}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block font-semibold">CLIP Match</span>
                  <span className="font-mono text-emerald-400 font-bold mt-0.5 block">{(activeOutput.score * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block font-semibold">Steps</span>
                  <span className="font-mono text-white mt-0.5 block">{activeOutput.steps}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block font-semibold">Dimensions</span>
                  <span className="font-mono text-white mt-0.5 block">{activeOutput.width}x{activeOutput.height}</span>
                </div>
              </div>
            </div>
          </div>
        </Dialog>
      )}
    </DashboardLayout>
  );
}
