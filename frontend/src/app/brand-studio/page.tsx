"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Sparkles,
  Zap,
  Sliders,
  RefreshCw,
  Layers,
  ArrowRight,
  TrendingUp,
  Award,
  Maximize2,
  Download,
  Share2,
  ChevronDown,
  Activity,
  Cpu,
  Globe,
  SlidersHorizontal,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Label, Field } from "@/components/ui/input";
import { Progress, Separator } from "@/components/ui/misc";
import { Header } from "@/components/layout/header";
import { toast } from "sonner";

/* ─── Mock Luxury Brand Specifications ─────────────────────────────────────── */
const BRAND_PROFILES = [
  {
    id: "brand-balenciaga",
    name: "Balenciaga",
    aesthetic: "Oversized silhouettes, cyber street couture, distressed denim, heavy proportions",
    image: "/images/cyberpunk.png",
    defaultWeights: { silhouette: 1.2, draping: 0.4, texture: 0.8, hardware: 0.9 },
  },
  {
    id: "brand-chanel",
    name: "Chanel",
    aesthetic: "Classic tweed, relaxed elegance, pearl/metal hardware accents, structured tailoring",
    image: "/images/minimalist.png",
    defaultWeights: { silhouette: 0.5, draping: 1.1, texture: 0.9, hardware: 0.7 },
  },
  {
    id: "brand-westwood",
    name: "Vivienne Westwood",
    aesthetic: "Neo-Victorian punk tailoring, heavy corsetry, metallic draping, safety hardware",
    image: "/images/avant_garde.png",
    defaultWeights: { silhouette: 0.9, draping: 1.3, texture: 0.6, hardware: 1.2 },
  },
];

const STYLE_CATEGORIES = [
  { name: "Haute Couture", desc: "Fine-art custom collections" },
  { name: "Cyber Streetwear", desc: "Techwear & urban modular items" },
  { name: "Minimalist Classic", desc: "Clean silhouettes, organic fabrics" },
  { name: "Baroque Retro", desc: "Heavy texture, classic patterns" },
];

export default function LuxuryBrandStudio() {
  const [selectedBrandA, setSelectedBrandA] = useState(BRAND_PROFILES[0]);
  const [selectedBrandB, setSelectedBrandB] = useState(BRAND_PROFILES[1]);
  const [mixRatio, setMixRatio] = useState(50); // Slider: 0 to 100 (% Brand A)

  // LoRA Weights
  const [weightSilhouette, setWeightSilhouette] = useState(1.0);
  const [weightDraping, setWeightDraping] = useState(1.0);
  const [weightTexture, setWeightTexture] = useState(1.0);
  const [weightHardware, setWeightHardware] = useState(1.0);

  // Switch Categories
  const [activeCategory, setActiveCategory] = useState(STYLE_CATEGORIES[0].name);

  // Generation status
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStep, setGenStep] = useState("");
  const [mixedPreview, setMixedPreview] = useState<string | null>(null);

  const handleApplyPreset = (brand: typeof BRAND_PROFILES[0], type: "A" | "B") => {
    if (type === "A") {
      setSelectedBrandA(brand);
      toast.success(`Applied ${brand.name} weights to Model A.`);
    } else {
      setSelectedBrandB(brand);
      toast.success(`Applied ${brand.name} weights to Model B.`);
    }
  };

  const handleBlendBrands = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenProgress(0);
    setMixedPreview(null);

    const steps = [
      { text: `Loading LoRA model A (${selectedBrandA.name}) weights...`, val: 20 },
      { text: `Loading LoRA model B (${selectedBrandB.name}) weights...`, val: 45 },
      { text: "Merging weight vectors inside neural cache layers...", val: 70 },
      { text: "Running latent diffusion synthesis steps...", val: 95 },
      { text: "Compiling output design mockup sheets...", val: 100 },
    ];

    for (const step of steps) {
      setGenStep(step.text);
      let currentVal = genProgress;
      while (currentVal < step.val) {
        currentVal += Math.floor(Math.random() * 5) + 1;
        if (currentVal > step.val) currentVal = step.val;
        setGenProgress(currentVal);
        await new Promise((resolve) => setTimeout(resolve, 90));
      }
    }

    // Output is determined by mixing ratio skew:
    // If skew is towards Brand A (>60%), show A's image, if B (>60%), show B's, else show C's (Westwood)
    let outputImage = BRAND_PROFILES[2].image; // default mixed
    if (mixRatio > 60) {
      outputImage = selectedBrandA.image;
    } else if (mixRatio < 40) {
      outputImage = selectedBrandB.image;
    }

    setMixedPreview(outputImage);
    toast.success("LoRA weights merged successfully!");
    setIsGenerating(false);
  };

  const handleDownload = (imgUrl: string) => {
    toast.success("Downloading design specs pack...");
    const link = document.createElement("a");
    link.href = imgUrl;
    link.download = "lora_blend_spec.png";
    link.click();
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast.success("LoRA blend configurations link copied!");
  };

  return (
    <>
      <Header title="Luxury Brand Studio" description="LoRA fine-tuning and brand aesthetic weight mixing sandbox" />

      <div className="px-6 py-8 space-y-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* ─── LEFT: CONTROLS & BRAND MERGING (lg:col-span-5) ────────────────── */}
          <div className="lg:col-span-5 space-y-6">
            {/* Style Switching (Quick Select) */}
            <Card variant="glass">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-bold uppercase tracking-wider text-primary">
                  Style Profile
                </CardTitle>
                <CardDescription>Select target collection theme category.</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="grid grid-cols-2 gap-2">
                  {STYLE_CATEGORIES.map((cat) => (
                    <button
                      key={cat.name}
                      onClick={() => setActiveCategory(cat.name)}
                      className={`p-3 rounded-xl border text-left transition-all ${
                        activeCategory === cat.name
                          ? "bg-primary/8 border-primary text-foreground"
                          : "bg-surface-2 border-border text-foreground-muted hover:bg-surface-3/50"
                      }`}
                    >
                      <h4 className="text-xs font-bold">{cat.name}</h4>
                      <p className="text-[9px] text-foreground-subtle mt-0.5 leading-normal">{cat.desc}</p>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* LoRA Controls */}
            <Card variant="default">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <SlidersHorizontal className="h-4 w-4 text-violet-400" />
                  LoRA Hyperparameters
                </CardTitle>
                <CardDescription>Adjust weights for specific brand stylistic layers.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                {/* Silhouette */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Silhouette Proportions (LoRA)</Label>
                    <span className="font-mono text-primary font-bold">{weightSilhouette.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.5}
                    step={0.05}
                    value={weightSilhouette}
                    onChange={(e) => setWeightSilhouette(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>

                {/* Draping */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Draping & Flow (LoRA)</Label>
                    <span className="font-mono text-primary font-bold">{weightDraping.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.5}
                    step={0.05}
                    value={weightDraping}
                    onChange={(e) => setWeightDraping(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>

                {/* Texture */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Fabric Texture Reflectivity (LoRA)</Label>
                    <span className="font-mono text-primary font-bold">{weightTexture.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.5}
                    step={0.05}
                    value={weightTexture}
                    onChange={(e) => setWeightTexture(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>

                {/* Hardware */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Accents & Hardware (LoRA)</Label>
                    <span className="font-mono text-primary font-bold">{weightHardware.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.5}
                    step={0.05}
                    value={weightHardware}
                    onChange={(e) => setWeightHardware(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Brand Comparison & Style Mixing */}
            <Card variant="default">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-bold flex items-center gap-2">
                  <Layers className="h-4 w-4 text-fuchsia-400" />
                  Style Mixing Console
                </CardTitle>
                <CardDescription>Configure target mix parameters.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Model A (Skew Right)">
                    <select
                      value={selectedBrandA.id}
                      onChange={(e) => setSelectedBrandA(BRAND_PROFILES.find(b => b.id === e.target.value) || BRAND_PROFILES[0])}
                      className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all"
                    >
                      {BRAND_PROFILES.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                    </select>
                  </Field>

                  <Field label="Model B (Skew Left)">
                    <select
                      value={selectedBrandB.id}
                      onChange={(e) => setSelectedBrandB(BRAND_PROFILES.find(b => b.id === e.target.value) || BRAND_PROFILES[1])}
                      className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all"
                    >
                      {BRAND_PROFILES.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                    </select>
                  </Field>
                </div>

                <div className="space-y-2 pt-2">
                  <div className="flex justify-between text-xs text-foreground-muted">
                    <span>{selectedBrandA.name} ({mixRatio}%)</span>
                    <span>{selectedBrandB.name} ({100 - mixRatio}%)</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={mixRatio}
                    onChange={(e) => setMixRatio(Number(e.target.value))}
                    className="w-full accent-primary bg-surface-3 rounded-full h-1 cursor-pointer"
                  />
                </div>

                <Button
                  onClick={handleBlendBrands}
                  disabled={isGenerating || selectedBrandA.id === selectedBrandB.id}
                  variant="glow"
                  className="w-full h-10 mt-2"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Blending Models...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      Blend Brand Models
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* ─── RIGHT: PREVIEW & BRAND CARDS (lg:col-span-7) ──────────────────── */}
          <div className="lg:col-span-7 space-y-6">
            {/* Live progress details */}
            {isGenerating && (
              <Card variant="glass" className="border-primary/30 animate-glow-pulse">
                <CardHeader>
                  <CardTitle className="text-sm font-bold text-foreground">Blending Active LoRAs...</CardTitle>
                  <CardDescription className="font-mono text-[10px] text-primary mt-1">{genStep}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Progress value={genProgress} color="primary" showValue label="Merging weights matrices" />
                </CardContent>
              </Card>
            )}

            {/* Live Preview Display Card */}
            {mixedPreview ? (
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-sm font-bold text-foreground">Mixed Output Preview</CardTitle>
                  <CardDescription>Synthesized design aligned with category: {activeCategory}</CardDescription>
                </CardHeader>
                <CardContent className="flex justify-center p-6 bg-surface-3/15 rounded-xl border border-border">
                  <div className="rounded-xl overflow-hidden border border-border aspect-[4/3] max-w-lg bg-surface-2">
                    <img src={mixedPreview} alt="Mixed Design Output" className="object-cover h-full w-full" />
                  </div>
                </CardContent>
                <CardFooter className="justify-end gap-2 border-t border-border pt-4 mt-2">
                  <Button size="sm" variant="outline" onClick={() => handleDownload(mixedPreview)}>
                    <Download className="h-4 w-4" />
                    Download Spec pack
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleShare}>
                    <Share2 className="h-4 w-4" />
                    Share Config
                  </Button>
                </CardFooter>
              </Card>
            ) : (
              /* Brand Cards list */
              <div className="space-y-4">
                <h3 className="text-overline text-foreground-subtle">Luxury brand model files</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {BRAND_PROFILES.map((brand) => (
                    <Card key={brand.id} variant="glass" className="hover:border-primary/30 transition-all">
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-center">
                          <CardTitle>{brand.name}</CardTitle>
                          <Award className="h-4 w-4 text-amber-400" />
                        </div>
                        <CardDescription className="text-body-xs leading-relaxed mt-1">
                          {brand.aesthetic}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="pt-2 flex flex-col gap-3">
                        {/* Static weights preview badges */}
                        <div className="grid grid-cols-2 gap-2 text-[10px]">
                          <div className="flex justify-between border-b border-border pb-1">
                            <span className="text-foreground-subtle">Silhouette</span>
                            <span className="font-mono font-bold text-primary">{brand.defaultWeights.silhouette}</span>
                          </div>
                          <div className="flex justify-between border-b border-border pb-1">
                            <span className="text-foreground-subtle">Draping</span>
                            <span className="font-mono font-bold text-primary">{brand.defaultWeights.draping}</span>
                          </div>
                          <div className="flex justify-between border-b border-border pb-1">
                            <span className="text-foreground-subtle">Reflectivity</span>
                            <span className="font-mono font-bold text-primary">{brand.defaultWeights.texture}</span>
                          </div>
                          <div className="flex justify-between border-b border-border pb-1">
                            <span className="text-foreground-subtle">Hardware</span>
                            <span className="font-mono font-bold text-primary">{brand.defaultWeights.hardware}</span>
                          </div>
                        </div>
                      </CardContent>
                      <CardFooter className="border-t border-border pt-3 mt-2 justify-end gap-2">
                        <Button size="xs" variant="outline" onClick={() => handleApplyPreset(brand, "A")}>
                          Apply Model A
                        </Button>
                        <Button size="xs" variant="outline" onClick={() => handleApplyPreset(brand, "B")}>
                          Apply Model B
                        </Button>
                      </CardFooter>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
