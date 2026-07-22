"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shirt,
  Wand2,
  Loader2,
  Sparkles,
  ChevronRight,
  Star,
  ArrowRight,
  Zap,
} from "lucide-react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { useStyleRecommendations } from "@/lib/queries";
import { styleRecommendationSchema, type StyleRecommendationFormValues } from "@/schemas";
import { formatLabel } from "@/lib/utils";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const GENDER_OPTIONS = ["men", "women", "unisex"] as const;
const STYLE_OPTIONS = [
  "streetwear", "luxury", "formal", "business_casual",
  "techwear", "minimalist", "vintage", "athleisure",
] as const;
const OCCASION_OPTIONS = [
  "casual", "business_casual", "formal", "party",
  "sport", "outdoor", "beach", "lounge",
] as const;
const FIT_OPTIONS = [
  "slim_fit", "regular_fit", "relaxed_fit", "oversized",
  "cropped", "skinny", "straight", "athletic_fit",
] as const;

const STYLE_PRESET_CARDS = [
  {
    label: "Cyber Techwear",
    icon: "⚡",
    gradient: "from-blue-500/20 to-violet-500/20",
    border: "border-violet-500/30",
    values: { gender: "men", style: "techwear", occasion: "casual", fit: "regular_fit" },
  },
  {
    label: "Luxury Minimal",
    icon: "✦",
    gradient: "from-amber-500/20 to-orange-500/20",
    border: "border-amber-500/30",
    values: { gender: "women", style: "luxury", occasion: "formal", fit: "slim_fit" },
  },
  {
    label: "Streetwear Drop",
    icon: "🔥",
    gradient: "from-fuchsia-500/20 to-pink-500/20",
    border: "border-fuchsia-500/30",
    values: { gender: "unisex", style: "streetwear", occasion: "casual", fit: "oversized" },
  },
];

function SelectChip({
  id, label, options, value, onChange,
}: {
  id: string;
  label: string;
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-xs font-medium text-foreground-muted uppercase tracking-wider">
        {label}
      </label>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              value === opt
                ? "bg-primary/20 border-primary text-primary shadow-sm"
                : "bg-surface-2 border-border text-foreground-muted hover:text-foreground hover:border-border-strong"
            }`}
          >
            {formatLabel(opt)}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function StylesPage() {
  const mutation = useStyleRecommendations();
  const [results, setResults] = useState<string[]>([]);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const { control, handleSubmit, reset, setValue } = useForm<StyleRecommendationFormValues>({
    resolver: zodResolver(styleRecommendationSchema),
    defaultValues: { gender: "unisex", style: "streetwear", occasion: "casual", fit: "regular_fit" },
  });

  const applyPreset = (preset: (typeof STYLE_PRESET_CARDS)[0]) => {
    setActivePreset(preset.label);
    Object.entries(preset.values).forEach(([k, v]) => {
      setValue(k as keyof StyleRecommendationFormValues, v);
    });
  };

  const onSubmit = async (values: StyleRecommendationFormValues) => {
    try {
      const data = await mutation.mutateAsync(values);
      setResults(data.recommendations);
      toast.success(`${data.recommendations.length} style profiles generated`);
    } catch {
      toast.error("Failed to generate styles — check backend connection.");
    }
  };

  return (
    <>
      <Header title="Style Recommendations" description="AI-powered personalized style combinations" />
      <div className="px-6 py-8 space-y-8 max-w-4xl">

        {/* Hero Banner */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-blue-500/10 via-violet-500/10 to-fuchsia-500/10 p-6"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-violet-500/5 to-transparent animate-pulse" />
          <div className="relative flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-violet-500 shadow-lg shadow-blue-500/25 shrink-0">
              <Shirt className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-heading-lg text-foreground font-bold">Style Intelligence Engine</h1>
              <p className="text-sm text-foreground-muted mt-0.5">
                Generate personalized fashion combinations using RAG-enhanced AI models.
              </p>
            </div>
            <div className="ml-auto hidden md:flex items-center gap-2">
              <Badge variant="outline" className="text-xs border-violet-500/40 text-violet-400">
                <Zap className="h-3 w-3 mr-1" />
                RAG Enhanced
              </Badge>
            </div>
          </div>
        </motion.div>

        {/* Quick Preset Cards */}
        <div className="space-y-3">
          <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Quick Presets</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {STYLE_PRESET_CARDS.map((preset) => (
              <motion.button
                key={preset.label}
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => applyPreset(preset)}
                className={`text-left p-4 rounded-xl border bg-gradient-to-br ${preset.gradient} ${preset.border} transition-all ${
                  activePreset === preset.label ? "ring-2 ring-primary/40 shadow-lg" : "hover:shadow-md"
                }`}
              >
                <div className="text-2xl mb-2">{preset.icon}</div>
                <p className="font-semibold text-sm text-foreground">{preset.label}</p>
                <p className="text-xs text-foreground-muted mt-0.5">{formatLabel(preset.values.style)} · {formatLabel(preset.values.fit)}</p>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Configuration Panel */}
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              Style Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                <Controller name="gender" control={control} render={({ field }) => (
                  <SelectChip id="style-gender" label="Gender" options={GENDER_OPTIONS} value={field.value} onChange={field.onChange} />
                )} />
                <Controller name="style" control={control} render={({ field }) => (
                  <SelectChip id="style-category" label="Style Category" options={STYLE_OPTIONS} value={field.value} onChange={field.onChange} />
                )} />
                <Controller name="occasion" control={control} render={({ field }) => (
                  <SelectChip id="style-occasion" label="Occasion" options={OCCASION_OPTIONS} value={field.value} onChange={field.onChange} />
                )} />
                <Controller name="fit" control={control} render={({ field }) => (
                  <SelectChip id="style-fit" label="Fit Profile" options={FIT_OPTIONS} value={field.value} onChange={field.onChange} />
                )} />
              </div>

              <div className="flex items-center gap-3 pt-2">
                <Button
                  id="style-submit"
                  type="submit"
                  variant="default"
                  size="lg"
                  className="gap-2"
                  disabled={mutation.isPending}
                >
                  {mutation.isPending ? (
                    <><Loader2 className="h-4 w-4 animate-spin" /> Generating Styles…</>
                  ) : (
                    <><Wand2 className="h-4 w-4" /> Generate Style Profiles</>
                  )}
                </Button>
                {results.length > 0 && (
                  <Button type="button" variant="ghost" size="sm" onClick={() => { setResults([]); reset(); setActivePreset(null); }}>
                    Clear Results
                  </Button>
                )}
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        <AnimatePresence>
          {results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider flex items-center gap-2">
                  <Star className="h-3.5 w-3.5 text-amber-400" />
                  {results.length} Style Profiles Generated
                </p>
              </div>
              <div className="grid gap-4">
                {results.map((rec, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -16 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="group rounded-xl border border-border bg-surface-1 p-5 hover:border-border-strong hover:bg-surface-2 transition-all"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary font-bold text-sm shrink-0">
                          {i + 1}
                        </div>
                        <p className="text-xs font-semibold text-primary uppercase tracking-wider">
                          Style Profile #{i + 1}
                        </p>
                      </div>
                      <ChevronRight className="h-4 w-4 text-foreground-subtle group-hover:text-foreground-muted transition-colors shrink-0" />
                    </div>
                    <p className="mt-3 text-sm text-foreground-muted leading-relaxed whitespace-pre-wrap pl-11">
                      {rec}
                    </p>
                  </motion.div>
                ))}
              </div>

              <div className="flex items-center justify-center pt-2">
                <button className="flex items-center gap-2 text-xs text-foreground-muted hover:text-foreground transition-colors">
                  <ArrowRight className="h-3.5 w-3.5" />
                  Refine & Generate More
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
