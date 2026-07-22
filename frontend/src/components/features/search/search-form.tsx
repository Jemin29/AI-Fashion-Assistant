"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Loader2, Database } from "lucide-react";
import { useState } from "react";
import { useSemanticSearchMutation } from "@/lib/queries";
import { semanticSearchSchema, type SemanticSearchFormValues } from "@/schemas";
import type { SemanticSearchResult } from "@/types";
import { truncate } from "@/lib/utils";

export function SearchForm() {
  const mutation = useSemanticSearchMutation();
  const [results, setResults] = useState<SemanticSearchResult[]>([]);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SemanticSearchFormValues>({
    resolver: zodResolver(semanticSearchSchema),
  });

  const onSubmit = async (values: SemanticSearchFormValues) => {
    try {
      const data = await mutation.mutateAsync(values.query);
      setResults(data.results);
      if (data.results.length === 0) {
        toast.info("No matching documents found");
      } else {
        toast.success(`${data.results.length} vector matches found`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-2">
          <label
            htmlFor="search-query"
            className="text-sm font-medium text-white/80"
          >
            Vector Search Query
          </label>
          <div className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/30" />
            <input
              id="search-query"
              {...register("query")}
              type="text"
              placeholder="oversized black utility hoodies for skateboarding"
              className="w-full rounded-xl border border-white/10 bg-white/5 py-2.5 pl-11 pr-4 text-sm text-white placeholder:text-white/25 focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all"
            />
          </div>
          {errors.query && (
            <p className="text-xs text-red-400">{errors.query.message}</p>
          )}
        </div>

        <button
          id="search-submit"
          type="submit"
          disabled={mutation.isPending}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
        >
          {mutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Search className="h-4 w-4" />
          )}
          {mutation.isPending ? "Searching…" : "Execute Vector Search"}
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
              <Database className="h-3.5 w-3.5" />
              {results.length} ChromaDB Matches
            </p>
            {results.map((r, i) => {
              const distance =
                typeof r.distance === "number" ? r.distance.toFixed(4) : "—";
              const metaEntries = Object.entries(r.metadata ?? {}).slice(0, 4);

              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.07 }}
                  className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-xs text-white/40">
                      <span className="font-medium text-violet-300">
                        Match {i + 1}
                      </span>
                      {r.collection && (
                        <span className="rounded-full border border-white/10 px-2 py-0.5">
                          {r.collection}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-white/30">
                      Distance: {distance}
                    </span>
                  </div>

                  <p className="text-sm text-white/80 leading-relaxed">
                    {truncate(r.document, 200)}
                  </p>

                  {metaEntries.length > 0 && (
                    <div className="flex flex-wrap gap-2 pt-1">
                      {metaEntries.map(([k, v]) => (
                        <span
                          key={k}
                          className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/40"
                        >
                          {k}: {String(v)}
                        </span>
                      ))}
                    </div>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
