"use client";

import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Wand2, Loader2 } from "lucide-react";
import { useState } from "react";
import { useStyleRecommendations } from "@/lib/queries";
import {
  styleRecommendationSchema,
  type StyleRecommendationFormValues,
} from "@/schemas";
import { formatLabel } from "@/lib/utils";

const GENDER_OPTIONS = ["men", "women", "unisex"] as const;
const STYLE_OPTIONS = [
  "streetwear",
  "luxury",
  "formal",
  "business_casual",
  "techwear",
  "minimalist",
  "vintage",
  "athleisure",
] as const;
const OCCASION_OPTIONS = [
  "casual",
  "business_casual",
  "formal",
  "party",
  "sport",
  "outdoor",
  "beach",
  "lounge",
] as const;
const FIT_OPTIONS = [
  "slim_fit",
  "regular_fit",
  "relaxed_fit",
  "oversized",
  "cropped",
  "skinny",
  "straight",
  "athletic_fit",
] as const;

function SelectField({
  id,
  label,
  options,
  value,
  onChange,
}: {
  id: string;
  label: string;
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm font-medium text-white/80">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-white focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all appearance-none cursor-pointer"
      >
        {options.map((opt) => (
          <option key={opt} value={opt} className="bg-gray-900 text-white">
            {formatLabel(opt)}
          </option>
        ))}
      </select>
    </div>
  );
}

export function StyleForm() {
  const mutation = useStyleRecommendations();
  const [results, setResults] = useState<string[]>([]);

  const { control, handleSubmit } = useForm<StyleRecommendationFormValues>({
    resolver: zodResolver(styleRecommendationSchema),
    defaultValues: {
      gender: "unisex",
      style: "streetwear",
      occasion: "casual",
      fit: "regular_fit",
    },
  });

  const onSubmit = async (values: StyleRecommendationFormValues) => {
    try {
      const data = await mutation.mutateAsync(values);
      setResults(data.recommendations);
      toast.success(`${data.recommendations.length} style recommendations found`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Controller
            name="gender"
            control={control}
            render={({ field }) => (
              <SelectField
                id="style-gender"
                label="Gender"
                options={GENDER_OPTIONS}
                value={field.value}
                onChange={field.onChange}
              />
            )}
          />
          <Controller
            name="style"
            control={control}
            render={({ field }) => (
              <SelectField
                id="style-category"
                label="Style Category"
                options={STYLE_OPTIONS}
                value={field.value}
                onChange={field.onChange}
              />
            )}
          />
          <Controller
            name="occasion"
            control={control}
            render={({ field }) => (
              <SelectField
                id="style-occasion"
                label="Occasion"
                options={OCCASION_OPTIONS}
                value={field.value}
                onChange={field.onChange}
              />
            )}
          />
          <Controller
            name="fit"
            control={control}
            render={({ field }) => (
              <SelectField
                id="style-fit"
                label="Fit Profile"
                options={FIT_OPTIONS}
                value={field.value}
                onChange={field.onChange}
              />
            )}
          />
        </div>

        <button
          id="style-submit"
          type="submit"
          disabled={mutation.isPending}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
        >
          {mutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Wand2 className="h-4 w-4" />
          )}
          {mutation.isPending ? "Generating…" : "Recommend Styles"}
        </button>
      </form>

      <AnimatePresence>
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <p className="text-xs font-medium text-violet-400 uppercase tracking-wider flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-violet-400 inline-block" />
              {results.length} Style Profiles
            </p>
            {results.map((rec, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className="rounded-xl border border-white/10 bg-white/5 p-4"
              >
                <p className="text-xs font-semibold text-violet-300 mb-1">
                  Style {i + 1}
                </p>
                <p className="text-sm text-white/80 leading-relaxed whitespace-pre-wrap">
                  {rec}
                </p>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
