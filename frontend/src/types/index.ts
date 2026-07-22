// ─── FastAPI Request Models ───────────────────────────────────────────────────

export interface QueryRequest {
  query: string;
  user_id?: string;
}

export interface StyleRecommendationRequest {
  gender: "men" | "women" | "unisex";
  style:
    | "streetwear"
    | "luxury"
    | "formal"
    | "business_casual"
    | "techwear"
    | "minimalist"
    | "vintage"
    | "athleisure";
  occasion:
    | "casual"
    | "business_casual"
    | "formal"
    | "party"
    | "sport"
    | "outdoor"
    | "beach"
    | "lounge";
  fit:
    | "slim_fit"
    | "regular_fit"
    | "relaxed_fit"
    | "oversized"
    | "cropped"
    | "skinny"
    | "straight"
    | "athletic_fit";
  user_id?: string;
}

export interface BrandRecommendationRequest {
  preferred_styles: string[];
  target_aesthetic: string;
  user_id?: string;
}

// ─── FastAPI Response Models ──────────────────────────────────────────────────

export interface SourceDocument {
  document: string;
  metadata: Record<string, string>;
  collection?: string;
  distance?: number;
}

export interface QueryResponse {
  response: string;
  source_documents: SourceDocument[];
}

export interface StyleRecommendationResponse {
  recommendations: string[];
}

export interface BrandRecommendationResponse {
  recommendations: string[];
}

export interface TrendExplainResponse {
  trend: string;
  explanation: string;
  confidence: number;
  reasoning: string;
}

export interface TrendForecast {
  trend: string;
  season: string;
  growth_rate: string;
  explanation: string;
  reasoning: string;
}

export interface TrendForecastResponse {
  forecasts: TrendForecast[];
}

export interface SemanticSearchResult {
  document: string;
  metadata: Record<string, string>;
  collection: string;
  distance: number;
}

export interface SemanticSearchResponse {
  results: SemanticSearchResult[];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

// ─── UI Types ────────────────────────────────────────────────────────────────

export type Season = "spring_summer" | "autumn_winter";

export interface NavItem {
  href: string;
  label: string;
  icon: string;
  description: string;
}
