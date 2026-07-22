"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { useQueryFashion } from "@/lib/queries";
import { querySchema, type QueryFormValues } from "@/schemas";
import type { QueryResponse } from "@/types";

export function QAForm() {
  const mutation = useQueryFashion();
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [showDocs, setShowDocs] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<QueryFormValues>({
    resolver: zodResolver(querySchema),
  });

  const onSubmit = async (values: QueryFormValues) => {
    try {
      const data = await mutation.mutateAsync(values);
      setResult(data);
      toast.success("Answer generated successfully");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Request failed");
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-2">
          <label
            htmlFor="qa-query"
            className="text-sm font-medium text-white/80"
          >
            Your Fashion Question
          </label>
          <textarea
            id="qa-query"
            {...register("query")}
            rows={4}
            placeholder="Why is linen popular in Spring/Summer? Explain its drape and weave characteristics."
            className="w-full resize-none rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-white/25 focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/20 transition-all"
          />
          {errors.query && (
            <p className="text-xs text-red-400">{errors.query.message}</p>
          )}
        </div>

        <button
          id="qa-submit"
          type="submit"
          disabled={mutation.isPending}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
        >
          {mutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
          {mutation.isPending ? "Thinking…" : "Ask Question"}
        </button>
      </form>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="rounded-xl border border-white/10 bg-white/5 p-5 space-y-4"
          >
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs font-medium text-violet-400 uppercase tracking-wider">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-400" />
                AI Response
              </div>
              <p className="text-sm text-white/85 leading-relaxed whitespace-pre-wrap">
                {result.response}
              </p>
            </div>

            {result.source_documents.length > 0 && (
              <div>
                <button
                  id="qa-toggle-docs"
                  type="button"
                  onClick={() => setShowDocs((v) => !v)}
                  className="flex items-center gap-2 text-xs text-white/40 hover:text-white/60 transition-colors"
                >
                  <BookOpen className="h-3.5 w-3.5" />
                  {result.source_documents.length} source documents
                  {showDocs ? (
                    <ChevronUp className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5" />
                  )}
                </button>

                <AnimatePresence>
                  {showDocs && (
                    <motion.ul
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="mt-3 space-y-2 overflow-hidden"
                    >
                      {result.source_documents.slice(0, 3).map((doc, i) => {
                        const name =
                          doc.metadata?.name ??
                          doc.metadata?.brand ??
                          `Document ${i + 1}`;
                        return (
                          <li
                            key={i}
                            className="rounded-lg border border-white/5 bg-white/3 px-3 py-2"
                          >
                            <p className="text-xs font-semibold text-violet-300">
                              {name}
                            </p>
                            <p className="text-xs text-white/40 mt-0.5 line-clamp-2">
                              {doc.document}
                            </p>
                          </li>
                        );
                      })}
                    </motion.ul>
                  )}
                </AnimatePresence>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
