import { z } from "zod";

// ─── Q&A Schema ───────────────────────────────────────────────────────────────

export const querySchema = z.object({
  query: z
    .string()
    .min(3, "Question must be at least 3 characters")
    .max(1000, "Question is too long"),
  user_id: z.string().optional(),
});

export type QueryFormValues = z.infer<typeof querySchema>;

// ─── Style Recommendation Schema ─────────────────────────────────────────────

export const styleRecommendationSchema = z.object({
  gender: z.enum(["men", "women", "unisex"]),
  style: z.enum([
    "streetwear",
    "luxury",
    "formal",
    "business_casual",
    "techwear",
    "minimalist",
    "vintage",
    "athleisure",
  ]),
  occasion: z.enum([
    "casual",
    "business_casual",
    "formal",
    "party",
    "sport",
    "outdoor",
    "beach",
    "lounge",
  ]),
  fit: z.enum([
    "slim_fit",
    "regular_fit",
    "relaxed_fit",
    "oversized",
    "cropped",
    "skinny",
    "straight",
    "athletic_fit",
  ]),
  user_id: z.string().optional(),
});

export type StyleRecommendationFormValues = z.infer<
  typeof styleRecommendationSchema
>;

// ─── Brand Recommendation Schema ─────────────────────────────────────────────

export const brandRecommendationSchema = z.object({
  preferred_styles: z
    .string()
    .min(2, "Enter at least one style (comma-separated)"),
  target_aesthetic: z
    .string()
    .min(10, "Please describe the target aesthetic in more detail"),
  user_id: z.string().optional(),
});

export type BrandRecommendationFormValues = z.infer<
  typeof brandRecommendationSchema
>;

// ─── Trend Schemas ────────────────────────────────────────────────────────────

export const trendExplainSchema = z.object({
  trend_name: z
    .string()
    .min(2, "Trend name must be at least 2 characters")
    .max(200, "Trend name is too long"),
});

export type TrendExplainFormValues = z.infer<typeof trendExplainSchema>;

export const trendForecastSchema = z.object({
  season: z.enum(["spring_summer", "autumn_winter"]),
});

export type TrendForecastFormValues = z.infer<typeof trendForecastSchema>;

// ─── Semantic Search Schema ───────────────────────────────────────────────────

export const semanticSearchSchema = z.object({
  query: z
    .string()
    .min(3, "Search query must be at least 3 characters")
    .max(500, "Query is too long"),
});

export type SemanticSearchFormValues = z.infer<typeof semanticSearchSchema>;
