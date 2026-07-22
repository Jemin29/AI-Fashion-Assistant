"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Tag, Loader2 } from "lucide-react";
import { useState } from "react";
import { useBrandRecommendations } from "@/lib/queries";
import {
  brandRecommendationSchema,
  type BrandRecommendationFormValues,
} from "@/schemas";

export function BrandForm() {
  const mutation = useBrandRecommendations();
  const [results, setResults] = useState<string[]>([]);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<BrandRecommendationFormValues>({
    resolver: zodResolver(brandRecommendationSchema),
    defaultValues: {
      preferred_styles: "streetwear, minimalist",
      target_aesthetic: "",
    },
  });

  const onSubmit = async (values: BrandRecommendationFormValues) => {
    try {
      const payload = {
        preferred_styles: values.preferred_styles
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        target_aesthetic: values.target_aesthetic,
        user_id: values.user_id,
      };
      const data = await mutation.mutateAsync(payload);
      setResults(data.recommendations);
      toast.success(`${data.recommendations.length} brand recommendations found`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-2">
          <label
            htmlFor="brand-styles"
            className="text-sm font-medium text-white/80"
          >
            Preferred Style Themes
            <span className="ml-1 text-white/30 font-normal">(comma-separated)</span>
          </label>
          <input
            id="brand-styles"
            {...register("preferred_styles")}
            type="text"
            placeholder="streetwear, minimalist, athletic"
            className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all"
          />
          {errors.preferred_styles && (
            <p className="text-xs text-red-400">
              {errors.preferred_styles.message}
            </p>
          )}
        </div>

        <div className="space-y-2">
          <label
            htmlFor="brand-aesthetic"
            className="text-sm font-medium text-white/80"
          >
            Target Brand Aesthetic Profile
          </label>
          <textarea
            id="brand-aesthetic"
            {...register("target_aesthetic")}
            rows={4}
            placeholder="Techwear and functional activewear with high-durability fabrics and a futuristic aesthetic."
            className="w-full resize-none rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/25 focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all"
          />
          {errors.target_aesthetic && (
            <p className="text-xs text-red-400">
              {errors.target_aesthetic.message}
            </p>
          )}
        </div>

        <button
          id="brand-submit"
          type="submit"
          disabled={mutation.isPending}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
        >
          {mutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Tag className="h-4 w-4" />
          )}
          {mutation.isPending ? "Matching…" : "Recommend Brands"}
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
              {results.length} Brand Profiles
            </p>
            {results.map((rec, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className="rounded-xl border border-white/10 bg-white/5 p-4"
              >
                <p className="text-xs font-semibold text-fuchsia-300 mb-1">
                  Brand {i + 1}
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
