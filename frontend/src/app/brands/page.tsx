"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Tag,
  Loader2,
  Sparkles,
  Globe,
  Award,
  TrendingUp,
  ArrowUpRight,
  Zap,
} from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { useBrandRecommendations } from "@/lib/queries";
import { brandRecommendationSchema, type BrandRecommendationFormValues } from "@/schemas";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea, Field } from "@/components/ui/input";

const STYLE_QUICK_TAGS = [
  "streetwear", "minimalist", "luxury", "techwear",
  "avant-garde", "sustainable", "athletic", "vintage",
];

const BRAND_TIERS = [
  { name: "Ultra Luxury", color: "text-amber-400", bg: "bg-amber-400/10 border-amber-400/30", icon: "◈" },
  { name: "Contemporary", color: "text-blue-400", bg: "bg-blue-400/10 border-blue-400/30", icon: "◆" },
  { name: "Streetwear", color: "text-fuchsia-400", bg: "bg-fuchsia-400/10 border-fuchsia-400/30", icon: "◉" },
  { name: "Performance", color: "text-emerald-400", bg: "bg-emerald-400/10 border-emerald-400/30", icon: "◎" },
];

export default function BrandsPage() {
  const mutation = useBrandRecommendations();
  const [results, setResults] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>(["streetwear", "minimalist"]);
  const [activeTier, setActiveTier] = useState(BRAND_TIERS[0].name);

  const {
    register,
    handleSubmit,
    setValue,
    getValues,
    formState: { errors },
  } = useForm<BrandRecommendationFormValues>({
    resolver: zodResolver(brandRecommendationSchema),
    defaultValues: {
      preferred_styles: "streetwear, minimalist",
      target_aesthetic: "",
    },
  });

  const toggleTag = (tag: string) => {
    const next = selectedTags.includes(tag)
      ? selectedTags.filter((t) => t !== tag)
      : [...selectedTags, tag];
    setSelectedTags(next);
    setValue("preferred_styles", next.join(", "));
  };

  const onSubmit = async (values: BrandRecommendationFormValues) => {
    try {
      const payload = {
        preferred_styles: values.preferred_styles.split(",").map((s) => s.trim()).filter(Boolean),
        target_aesthetic: values.target_aesthetic,
        user_id: values.user_id,
      };
      const data = await mutation.mutateAsync(payload);
      setResults(data.recommendations);
      toast.success(`${data.recommendations.length} brand profiles matched`);
    } catch {
      toast.error("Brand matching failed — check backend connection.");
    }
  };

  return (
    <>
      <Header title="Brand Recommendations" description="Match brands to your aesthetic profile" />
      <div className="px-6 py-8 space-y-8 max-w-4xl">

        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-fuchsia-500/10 via-pink-500/10 to-rose-500/10 p-6"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-fuchsia-500/5 to-transparent animate-pulse" />
          <div className="relative flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-fuchsia-500 to-pink-500 shadow-lg shadow-fuchsia-500/25 shrink-0">
              <Tag className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-heading-lg text-foreground font-bold">Brand Matching Intelligence</h1>
              <p className="text-sm text-foreground-muted mt-0.5">
                Identify luxury and streetwear brands aligned with your aesthetic DNA.
              </p>
            </div>
            <div className="ml-auto hidden md:flex items-center gap-2">
              <Badge variant="outline" className="text-xs border-fuchsia-500/40 text-fuchsia-400">
                <Globe className="h-3 w-3 mr-1" />
                Global Index
              </Badge>
            </div>
          </div>
        </motion.div>

        {/* Brand Tier Selector */}
        <div className="space-y-3">
          <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Market Tier Filter</p>
          <div className="flex flex-wrap gap-2">
            {BRAND_TIERS.map((tier) => (
              <button
                key={tier.name}
                type="button"
                onClick={() => setActiveTier(tier.name)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border transition-all ${
                  activeTier === tier.name
                    ? `${tier.bg} ${tier.color} shadow-sm ring-1 ring-current/30`
                    : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
                }`}
              >
                <span>{tier.icon}</span>
                {tier.name}
              </button>
            ))}
          </div>
        </div>

        {/* Quick Style Tags */}
        <div className="space-y-3">
          <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Style DNA Tags</p>
          <div className="flex flex-wrap gap-2">
            {STYLE_QUICK_TAGS.map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => toggleTag(tag)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all capitalize ${
                  selectedTags.includes(tag)
                    ? "bg-fuchsia-500/20 border-fuchsia-500/40 text-fuchsia-300"
                    : "bg-surface-2 border-border text-foreground-muted hover:text-foreground hover:border-border-strong"
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        {/* Form Panel */}
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Award className="h-4 w-4 text-fuchsia-400" />
              Aesthetic Profile
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
              <Field label="Target Aesthetic Description" error={errors.target_aesthetic?.message}>
                <Textarea
                  id="brand-aesthetic"
                  {...register("target_aesthetic")}
                  rows={4}
                  placeholder="Techwear and functional activewear with high-durability fabrics and a futuristic, cyberpunk-influenced aesthetic..."
                  error={!!errors.target_aesthetic}
                />
              </Field>

              <div className="flex items-center gap-3">
                <Button
                  id="brand-submit"
                  type="submit"
                  variant="default"
                  size="lg"
                  className="gap-2"
                  disabled={mutation.isPending}
                >
                  {mutation.isPending ? (
                    <><Loader2 className="h-4 w-4 animate-spin" /> Matching Brands…</>
                  ) : (
                    <><Tag className="h-4 w-4" /> Recommend Brands</>
                  )}
                </Button>
                {results.length > 0 && (
                  <Button type="button" variant="ghost" size="sm" onClick={() => setResults([])}>
                    Clear
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
              <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="h-3.5 w-3.5 text-fuchsia-400" />
                {results.length} Brand Matches
              </p>
              <div className="grid gap-4 sm:grid-cols-2">
                {results.map((rec, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.08 }}
                    className="group relative rounded-xl border border-border bg-surface-1 p-5 hover:border-fuchsia-500/30 hover:shadow-md transition-all overflow-hidden"
                  >
                    <div className="absolute top-0 right-0 h-16 w-16 rounded-bl-full bg-gradient-to-bl from-fuchsia-500/10 to-transparent" />
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-fuchsia-500/10 text-fuchsia-400 font-bold text-xs shrink-0">
                          {i + 1}
                        </div>
                        <p className="text-xs font-semibold text-fuchsia-400 uppercase tracking-wider">Brand #{i + 1}</p>
                      </div>
                      <ArrowUpRight className="h-3.5 w-3.5 text-foreground-subtle group-hover:text-fuchsia-400 transition-colors" />
                    </div>
                    <p className="text-sm text-foreground-muted leading-relaxed whitespace-pre-wrap">
                      {rec}
                    </p>
                    <div className="mt-3 flex items-center gap-2">
                      <TrendingUp className="h-3 w-3 text-emerald-400" />
                      <span className="text-[10px] text-emerald-400">High aesthetic match</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
