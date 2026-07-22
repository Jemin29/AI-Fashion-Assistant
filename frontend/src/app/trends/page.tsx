"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  TrendingUp,
  BarChart3,
  Loader2,
  Sparkles,
  Activity,
  Clock,
  ChevronRight,
  Zap,
  Globe,
} from "lucide-react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { useTrendExplainMutation, useTrendForecastMutation } from "@/lib/queries";
import { trendExplainSchema, type TrendExplainFormValues, trendForecastSchema, type TrendForecastFormValues } from "@/schemas";
import type { TrendExplainResponse, TrendForecast } from "@/types";
import { formatLabel } from "@/lib/utils";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Field } from "@/components/ui/input";
import { Progress } from "@/components/ui/misc";

type TrendMode = "explain" | "forecast";
type TrendResult =
  | { mode: "explain"; data: TrendExplainResponse }
  | { mode: "forecast"; data: TrendForecast[] };

const TRENDING_NOW = [
  "Quiet Luxury", "Gorpcore", "Coastal Grandmother",
  "Y2K Revival", "Dark Academia", "Neo-Brutalism",
];

const SEASON_OPTIONS = [
  {
    value: "spring_summer",
    label: "Spring / Summer",
    icon: "☀️",
    desc: "SS26 Collection Forecast",
    color: "text-amber-400 border-amber-400/30 bg-amber-400/10",
  },
  {
    value: "autumn_winter",
    label: "Autumn / Winter",
    icon: "🍂",
    desc: "AW26 Collection Forecast",
    color: "text-orange-400 border-orange-400/30 bg-orange-400/10",
  },
];

export default function TrendsPage() {
  const [mode, setMode] = useState<TrendMode>("explain");
  const [result, setResult] = useState<TrendResult | null>(null);
  const [selectedSeason, setSelectedSeason] = useState("spring_summer");

  const explainMutation = useTrendExplainMutation();
  const forecastMutation = useTrendForecastMutation();

  const explainForm = useForm<TrendExplainFormValues>({
    resolver: zodResolver(trendExplainSchema),
    defaultValues: { trend_name: "" },
  });

  const forecastForm = useForm<TrendForecastFormValues>({
    resolver: zodResolver(trendForecastSchema),
    defaultValues: { season: "spring_summer" },
  });

  const onExplain = async (values: TrendExplainFormValues) => {
    try {
      const data = await explainMutation.mutateAsync(values.trend_name);
      setResult({ mode: "explain", data });
      toast.success("Trend analysis complete");
    } catch {
      toast.error("Failed to analyze trend — check backend connection.");
    }
  };

  const onForecast = async (values: TrendForecastFormValues) => {
    try {
      const data = await forecastMutation.mutateAsync(values.season);
      setResult({ mode: "forecast", data: data.forecasts });
      toast.success(`${data.forecasts.length} trend forecasts generated`);
    } catch {
      toast.error("Failed to forecast trends — check backend connection.");
    }
  };

  const isPending = explainMutation.isPending || forecastMutation.isPending;

  return (
    <>
      <Header title="Trend Forecasting" description="AI-powered fashion trend analysis & season forecasts" />
      <div className="px-6 py-8 space-y-8 max-w-4xl">

        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-emerald-500/10 via-cyan-500/10 to-teal-500/10 p-6"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500/5 to-transparent animate-pulse" />
          <div className="relative flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-cyan-500 shadow-lg shadow-emerald-500/25 shrink-0">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-heading-lg text-foreground font-bold">Trend Intelligence System</h1>
              <p className="text-sm text-foreground-muted mt-0.5">
                Analyze active trends and generate AI-powered season forecasts with confidence metrics.
              </p>
            </div>
            <div className="ml-auto hidden md:flex items-center gap-2">
              <Badge variant="outline" className="text-xs border-emerald-500/40 text-emerald-400">
                <Globe className="h-3 w-3 mr-1" />
                Live Index
              </Badge>
            </div>
          </div>
        </motion.div>

        {/* Mode Switch */}
        <div className="flex gap-1 rounded-xl border border-border bg-surface-2 p-1 w-fit">
          {([
            { id: "explain" as TrendMode, label: "Analyze Trend", icon: TrendingUp },
            { id: "forecast" as TrendMode, label: "Season Forecast", icon: BarChart3 },
          ] as { id: TrendMode; label: string; icon: React.ElementType }[]).map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setMode(id)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                mode === id
                  ? "bg-gradient-to-r from-emerald-600 to-cyan-600 text-white shadow"
                  : "text-foreground-muted hover:text-foreground"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* Trending Now Bar */}
        {mode === "explain" && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Trending Now</p>
            <div className="flex flex-wrap gap-2">
              {TRENDING_NOW.map((trend) => (
                <button
                  key={trend}
                  type="button"
                  onClick={() => explainForm.setValue("trend_name", trend)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-surface-2 border border-border text-foreground-muted hover:text-foreground hover:border-border-strong transition-all"
                >
                  <Activity className="h-3 w-3 text-emerald-400" />
                  {trend}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Forms */}
        <AnimatePresence mode="wait">
          {mode === "explain" ? (
            <motion.div
              key="explain"
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 16 }}
            >
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-emerald-400" />
                    Trend Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <form onSubmit={explainForm.handleSubmit(onExplain)} className="space-y-4">
                    <Field label="Trend Name to Analyze" error={explainForm.formState.errors.trend_name?.message}>
                      <Input
                        id="trend-name"
                        {...explainForm.register("trend_name")}
                        type="text"
                        placeholder="e.g. Velvet Gown Opulence, Gorpcore, Quiet Luxury"
                        error={!!explainForm.formState.errors.trend_name}
                      />
                    </Field>
                    <Button
                      id="trend-explain-submit"
                      type="submit"
                      variant="default"
                      size="lg"
                      disabled={isPending}
                      className="gap-2"
                    >
                      {isPending ? (
                        <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing…</>
                      ) : (
                        <><TrendingUp className="h-4 w-4" /> Analyze Trend</>
                      )}
                    </Button>
                  </form>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              key="forecast"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
            >
              <Card variant="glass">
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-cyan-400" />
                    Season Forecast
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <form onSubmit={forecastForm.handleSubmit(onForecast)} className="space-y-5">
                    <div className="space-y-3">
                      <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Target Season</p>
                      <div className="grid grid-cols-2 gap-3">
                        {SEASON_OPTIONS.map((s) => (
                          <button
                            key={s.value}
                            type="button"
                            onClick={() => {
                              setSelectedSeason(s.value);
                              forecastForm.setValue("season", s.value as "spring_summer" | "autumn_winter");
                            }}
                            className={`p-4 rounded-xl border text-left transition-all ${
                              selectedSeason === s.value
                                ? `${s.color} ring-2 ring-current/30 shadow-sm`
                                : "bg-surface-2 border-border hover:border-border-strong"
                            }`}
                          >
                            <div className="text-2xl mb-1">{s.icon}</div>
                            <p className="font-semibold text-sm text-foreground">{s.label}</p>
                            <p className="text-xs text-foreground-muted mt-0.5">{s.desc}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                    <Button
                      id="trend-forecast-submit"
                      type="submit"
                      variant="default"
                      size="lg"
                      disabled={isPending}
                      className="gap-2"
                    >
                      {isPending ? (
                        <><Loader2 className="h-4 w-4 animate-spin" /> Forecasting…</>
                      ) : (
                        <><BarChart3 className="h-4 w-4" /> Generate Forecast</>
                      )}
                    </Button>
                  </form>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              {result.mode === "explain" && (
                <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-cyan-500/5 p-6 space-y-5">
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="font-bold text-foreground text-lg flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-emerald-400" />
                      {result.data.trend}
                    </h3>
                    <Badge variant="outline" className="text-emerald-400 border-emerald-400/40 shrink-0">
                      {(result.data.confidence * 100).toFixed(0)}% Confidence
                    </Badge>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Confidence Score</p>
                    <Progress value={result.data.confidence * 100} className="h-1.5" />
                  </div>
                  <p className="text-sm text-foreground-muted leading-relaxed">{result.data.explanation}</p>
                  <div className="rounded-xl bg-surface-2 border border-border p-4">
                    <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider mb-2">AI Reasoning</p>
                    <p className="text-xs text-foreground-subtle leading-relaxed">{result.data.reasoning}</p>
                  </div>
                </div>
              )}

              {result.mode === "forecast" && (
                <div className="space-y-4">
                  <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider flex items-center gap-2">
                    <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
                    {result.data.length} Season Forecasts
                  </p>
                  {result.data.map((fc, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -16 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.08 }}
                      className="rounded-xl border border-border bg-surface-1 p-5 space-y-3 hover:border-border-strong transition-all"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <h4 className="font-semibold text-foreground flex items-center gap-2">
                          <ChevronRight className="h-4 w-4 text-cyan-400 shrink-0" />
                          {fc.trend}
                        </h4>
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400">
                            <Zap className="h-2.5 w-2.5" />
                            {fc.growth_rate}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-foreground-muted">
                        <Clock className="h-3 w-3" />
                        Season: {formatLabel(fc.season)}
                      </div>
                      <p className="text-sm text-foreground-muted leading-relaxed">{fc.explanation}</p>
                      <p className="text-xs text-foreground-subtle border-t border-border pt-3 leading-relaxed">{fc.reasoning}</p>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
