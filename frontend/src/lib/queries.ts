import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getHealth,
  postQuery,
  postStyleRecommendations,
  postBrandRecommendations,
  getTrendExplain,
  getTrendForecast,
  getSemanticSearch,
} from "@/lib/api";
import type {
  QueryRequest,
  StyleRecommendationRequest,
  BrandRecommendationRequest,
} from "@/types";

// ─── Query Keys ───────────────────────────────────────────────────────────────

export const queryKeys = {
  health: ["health"] as const,
  trendExplain: (name: string) => ["trends", "explain", name] as const,
  trendForecast: (season: string) => ["trends", "forecast", season] as const,
  semanticSearch: (query: string) => ["search", query] as const,
};

// ─── Health Check ─────────────────────────────────────────────────────────────

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    staleTime: 30_000,
    retry: 1,
  });
}

// ─── Fashion Q&A ─────────────────────────────────────────────────────────────

export function useQueryFashion() {
  return useMutation({
    mutationFn: (payload: QueryRequest) => postQuery(payload),
  });
}

// ─── Style Recommendations ────────────────────────────────────────────────────

export function useStyleRecommendations() {
  return useMutation({
    mutationFn: (payload: StyleRecommendationRequest) =>
      postStyleRecommendations(payload),
  });
}

// ─── Brand Recommendations ────────────────────────────────────────────────────

export function useBrandRecommendations() {
  return useMutation({
    mutationFn: (payload: BrandRecommendationRequest) =>
      postBrandRecommendations(payload),
  });
}

// ─── Trend Explain ────────────────────────────────────────────────────────────

export function useTrendExplainMutation() {
  return useMutation({
    mutationFn: (trend_name: string) => getTrendExplain(trend_name),
  });
}

// ─── Trend Forecast ───────────────────────────────────────────────────────────

export function useTrendForecastMutation() {
  return useMutation({
    mutationFn: (season: string) => getTrendForecast(season),
  });
}

// ─── Semantic Search ─────────────────────────────────────────────────────────

export function useSemanticSearchMutation() {
  return useMutation({
    mutationFn: (query: string) => getSemanticSearch(query),
  });
}
