"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Loader2,
  Database,
  Sparkles,
  FileText,
  Clock,
  ChevronRight,
  Zap,
  Globe,
  Filter,
} from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { useSemanticSearchMutation } from "@/lib/queries";
import { semanticSearchSchema, type SemanticSearchFormValues } from "@/schemas";
import type { SemanticSearchResult } from "@/types";
import { truncate } from "@/lib/utils";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input, Field } from "@/components/ui/input";

const EXAMPLE_QUERIES = [
  "oversized black utility hoodies for skateboarding",
  "linen breathable summer dresses with natural dyes",
  "waterproof techwear jackets with modular pockets",
  "minimalist luxury cashmere knitwear editorial",
  "avant-garde metallic runway corset with structured shoulders",
];

const COLLECTION_FILTERS = [
  { label: "All Collections", value: null },
  { label: "Fabrics", value: "fabrics" },
  { label: "Brands", value: "brands" },
  { label: "Trends", value: "trends" },
  { label: "Garments", value: "garments" },
];

function DistanceBar({ value }: { value: number }) {
  // Distance 0 = perfect match, ~2 = far. Convert to similarity %
  const similarity = Math.max(0, Math.min(100, Math.round((1 - value / 2) * 100)));
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-foreground-subtle">
        <span>Similarity</span>
        <span className="text-emerald-400 font-medium">{similarity}%</span>
      </div>
      <div className="h-1 rounded-full bg-surface-3 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all"
          style={{ width: `${similarity}%` }}
        />
      </div>
    </div>
  );
}

export default function SearchPage() {
  const mutation = useSemanticSearchMutation();
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<SemanticSearchFormValues>({
    resolver: zodResolver(semanticSearchSchema),
  });

  const filteredResults = activeFilter
    ? results.filter((r) => r.collection === activeFilter)
    : results;

  const onSubmit = async (values: SemanticSearchFormValues) => {
    try {
      setSearchQuery(values.query);
      const data = await mutation.mutateAsync(values.query);
      setResults(data.results);
      if (data.results.length === 0) {
        toast.info("No matching documents found in ChromaDB index.");
      } else {
        toast.success(`${data.results.length} vector matches found`);
      }
    } catch {
      toast.error("Vector search failed — check backend connection.");
    }
  };

  return (
    <>
      <Header title="Semantic Search" description="High-dimensional vector search over ChromaDB" />
      <div className="px-6 py-8 space-y-8 max-w-4xl">

        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-orange-500/10 via-amber-500/10 to-yellow-500/10 p-6"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/5 to-transparent animate-pulse" />
          <div className="relative flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 shadow-lg shadow-orange-500/25 shrink-0">
              <Database className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-heading-lg text-foreground font-bold">Vector Search Engine</h1>
              <p className="text-sm text-foreground-muted mt-0.5">
                High-dimensional similarity search over the ChromaDB fashion knowledge index.
              </p>
            </div>
            <div className="ml-auto hidden md:flex flex-col items-end gap-1">
              <Badge variant="outline" className="text-xs border-orange-500/40 text-orange-400">
                <Globe className="h-3 w-3 mr-1" />
                ChromaDB
              </Badge>
              <span className="text-[10px] text-foreground-subtle">cosine similarity</span>
            </div>
          </div>
        </motion.div>

        {/* Example Queries */}
        <div className="space-y-3">
          <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Example Queries</p>
          <div className="flex flex-col gap-2">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => setValue("query", q)}
                className="flex items-center gap-3 rounded-xl border border-border bg-surface-2 px-4 py-3 text-left text-sm text-foreground-muted hover:text-foreground hover:border-border-strong hover:bg-surface-1 transition-all group"
              >
                <Search className="h-3.5 w-3.5 shrink-0 text-orange-400" />
                <span className="flex-1 truncate">{q}</span>
                <ChevronRight className="h-3.5 w-3.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
            ))}
          </div>
        </div>

        {/* Search Form */}
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Search className="h-4 w-4 text-orange-400" />
              Vector Query
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <Field label="Natural Language Search Query" error={errors.query?.message}>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground-subtle" />
                  <Input
                    id="search-query"
                    {...register("query")}
                    type="text"
                    placeholder="oversized black utility hoodies for skateboarding..."
                    className="pl-10"
                    error={!!errors.query}
                  />
                </div>
              </Field>
              <div className="flex items-center gap-3">
                <Button
                  id="search-submit"
                  type="submit"
                  variant="default"
                  size="lg"
                  disabled={mutation.isPending}
                  className="gap-2"
                >
                  {mutation.isPending ? (
                    <><Loader2 className="h-4 w-4 animate-spin" /> Searching…</>
                  ) : (
                    <><Zap className="h-4 w-4" /> Execute Vector Search</>
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
              {/* Results header + collection filter */}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider flex items-center gap-2">
                  <Database className="h-3.5 w-3.5 text-orange-400" />
                  {filteredResults.length} ChromaDB Matches
                  {searchQuery && (
                    <span className="text-foreground-subtle">for &ldquo;{truncate(searchQuery, 40)}&rdquo;</span>
                  )}
                </p>
                <div className="flex items-center gap-2">
                  <Filter className="h-3.5 w-3.5 text-foreground-subtle" />
                  {COLLECTION_FILTERS.map((f) => (
                    <button
                      key={f.label}
                      type="button"
                      onClick={() => setActiveFilter(f.value)}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-all ${
                        activeFilter === f.value
                          ? "bg-orange-500/20 border-orange-500/40 text-orange-300"
                          : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Result Cards */}
              <div className="space-y-3">
                {filteredResults.map((r, i) => {
                  const distance = typeof r.distance === "number" ? r.distance : 1;
                  const metaEntries = Object.entries(r.metadata ?? {}).slice(0, 4);

                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -16 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.07 }}
                      className="rounded-xl border border-border bg-surface-1 p-5 space-y-3 hover:border-border-strong transition-all group"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/10 shrink-0">
                            <FileText className="h-4 w-4 text-orange-400" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="text-xs font-semibold text-orange-400 uppercase tracking-wider">
                                Match #{i + 1}
                              </p>
                              {r.collection && (
                                <span className="rounded-full border border-border bg-surface-2 px-2 py-0.5 text-[10px] text-foreground-subtle">
                                  {r.collection}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <Clock className="h-3 w-3 text-foreground-subtle" />
                          <span className="text-[10px] text-foreground-subtle">dist: {distance.toFixed(4)}</span>
                        </div>
                      </div>

                      <p className="text-sm text-foreground-muted leading-relaxed">
                        {truncate(r.document, 250)}
                      </p>

                      <DistanceBar value={distance} />

                      {metaEntries.length > 0 && (
                        <div className="flex flex-wrap gap-2 pt-1 border-t border-border">
                          {metaEntries.map(([k, v]) => (
                            <span
                              key={k}
                              className="rounded-full border border-border bg-surface-2 px-2.5 py-0.5 text-[10px] text-foreground-subtle"
                            >
                              <span className="text-foreground-muted">{k}:</span> {String(v)}
                            </span>
                          ))}
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </div>

              {filteredResults.length === 0 && activeFilter && (
                <div className="rounded-xl border border-border bg-surface-2 p-8 text-center">
                  <Sparkles className="h-8 w-8 text-foreground-subtle mx-auto mb-3" />
                  <p className="text-sm text-foreground-muted">No results in &ldquo;{activeFilter}&rdquo; collection.</p>
                  <Button type="button" variant="ghost" size="sm" className="mt-3" onClick={() => setActiveFilter(null)}>
                    Show All Results
                  </Button>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
