"use client";

import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, BarChart3, Loader2, Sparkles } from "lucide-react";
import { useState } from "react";
import { useTrendExplainMutation, useTrendForecastMutation } from "@/lib/queries";
import {
  trendExplainSchema,
  type TrendExplainFormValues,
  trendForecastSchema,
  type TrendForecastFormValues,
} from "@/schemas";
import type { TrendExplainResponse, TrendForecast } from "@/types";
import { formatLabel } from "@/lib/utils";

type TrendMode = "explain" | "forecast";
type TrendResult =
  | { mode: "explain"; data: TrendExplainResponse }
  | { mode: "forecast"; data: TrendForecast[] };

export function TrendForm() {
  const [mode, setMode] = useState<TrendMode>("explain");
  const [result, setResult] = useState<TrendResult | null>(null);

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
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  const onForecast = async (values: TrendForecastFormValues) => {
    try {
      const data = await forecastMutation.mutateAsync(values.season);
      setResult({ mode: "forecast", data: data.forecasts });
      toast.success(`${data.forecasts.length} trend forecasts generated`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  const isPending = explainMutation.isPending || forecastMutation.isPending;

  return (
    <div className="space-y-6">
      {/* Mode Tabs */}
      <div className="flex gap-2 rounded-xl border border-white/10 bg-white/5 p-1 w-fit">
        <button
          id="trend-mode-explain"
          type="button"
          onClick={() => setMode("explain")}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            mode === "explain"
              ? "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow"
              : "text-white/50 hover:text-white/80"
          }`}
        >
          <TrendingUp className="h-3.5 w-3.5" />
          Analyze Trend
        </button>
        <button
          id="trend-mode-forecast"
          type="button"
          onClick={() => setMode("forecast")}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            mode === "forecast"
              ? "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow"
              : "text-white/50 hover:text-white/80"
          }`}
        >
          <BarChart3 className="h-3.5 w-3.5" />
          Forecast Season
        </button>
      </div>

      <AnimatePresence mode="wait">
        {mode === "explain" ? (
          <motion.form
            key="explain"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            onSubmit={explainForm.handleSubmit(onExplain)}
            className="space-y-4"
          >
            <div className="space-y-2">
              <label
                htmlFor="trend-name"
                className="text-sm font-medium text-white/80"
              >
                Trend Name
              </label>
              <input
                id="trend-name"
                {...explainForm.register("trend_name")}
                type="text"
                placeholder="Velvet Gown Opulence"
                className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all"
              />
              {explainForm.formState.errors.trend_name && (
                <p className="text-xs text-red-400">
                  {explainForm.formState.errors.trend_name.message}
                </p>
              )}
            </div>
            <button
              id="trend-explain-submit"
              type="submit"
              disabled={isPending}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
            >
              {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
              {isPending ? "Analyzing…" : "Analyze Trend"}
            </button>
          </motion.form>
        ) : (
          <motion.form
            key="forecast"
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            onSubmit={forecastForm.handleSubmit(onForecast)}
            className="space-y-4"
          >
            <div className="space-y-2">
              <label
                htmlFor="forecast-season"
                className="text-sm font-medium text-white/80"
              >
                Target Season
              </label>
              <Controller
                name="season"
                control={forecastForm.control}
                render={({ field }) => (
                  <select
                    id="forecast-season"
                    value={field.value}
                    onChange={(e) => field.onChange(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-white focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all appearance-none"
                  >
                    <option value="spring_summer" className="bg-gray-900">Spring / Summer</option>
                    <option value="autumn_winter" className="bg-gray-900">Autumn / Winter</option>
                  </select>
                )}
              />
            </div>
            <button
              id="trend-forecast-submit"
              type="submit"
              disabled={isPending}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
            >
              {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
              {isPending ? "Forecasting…" : "Forecast Trends"}
            </button>
          </motion.form>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {result && result.mode === "explain" && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-white/10 bg-white/5 p-5 space-y-3"
          >
            <h3 className="font-semibold text-white flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-violet-400" />
              {result.data.trend}
            </h3>
            <p className="text-sm text-white/75 leading-relaxed">
              {result.data.explanation}
            </p>
            <div className="flex gap-4 text-xs text-white/40">
              <span>
                Confidence:{" "}
                <span className="text-emerald-400 font-medium">
                  {(result.data.confidence * 100).toFixed(0)}%
                </span>
              </span>
            </div>
            <p className="text-xs text-white/50 leading-relaxed border-t border-white/10 pt-3">
              {result.data.reasoning}
            </p>
          </motion.div>
        )}

        {result && result.mode === "forecast" && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <p className="text-xs font-medium text-violet-400 uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="h-3.5 w-3.5" />
              {result.data.length} Trend Forecasts
            </p>
            {result.data.map((fc, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07 }}
                className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <h4 className="font-semibold text-white text-sm">{fc.trend}</h4>
                  <span className="shrink-0 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400">
                    {fc.growth_rate}
                  </span>
                </div>
                <p className="text-xs text-white/50">
                  Season: {formatLabel(fc.season)}
                </p>
                <p className="text-sm text-white/75 leading-relaxed">
                  {fc.explanation}
                </p>
                <p className="text-xs text-white/40 border-t border-white/10 pt-2">
                  {fc.reasoning}
                </p>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
