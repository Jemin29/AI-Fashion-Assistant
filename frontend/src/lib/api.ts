import axios from "axios";
import type {
  QueryRequest,
  QueryResponse,
  StyleRecommendationRequest,
  StyleRecommendationResponse,
  BrandRecommendationRequest,
  BrandRecommendationResponse,
  TrendExplainResponse,
  TrendForecastResponse,
  SemanticSearchResponse,
  HealthResponse,
} from "@/types";

// ─── Axios Instance ───────────────────────────────────────────────────────────

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15000,
});

// Request Interceptor: Inject JWT Bearer auth token if present
api.interceptors.request.use(
  (config) => {
    if (typeof window !== "undefined") {
      const savedUser = localStorage.getItem("fashion_auth_user");
      if (savedUser) {
        try {
          const user = JSON.parse(savedUser);
          if (user.token) {
            config.headers.Authorization = `Bearer ${user.token}`;
          }
        } catch {
          // ignore parsing issues
        }
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response Interceptor: Retries + Token Refresh Fallback + Error Normalization
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (!originalRequest) return Promise.reject(error);

    // 1. Retry Logic for 5xx or Network/Timeout errors (up to 3 times)
    originalRequest._retryCount = originalRequest._retryCount ?? 0;
    const shouldRetry =
      originalRequest._retryCount < 3 &&
      (!error.response || (error.response.status >= 500 && error.response.status <= 504));

    if (shouldRetry) {
      originalRequest._retryCount += 1;
      const delay = Math.pow(2, originalRequest._retryCount) * 1000;
      console.warn(`[API Connection] Retry attempt #${originalRequest._retryCount} in ${delay}ms for URL: ${originalRequest.url}`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return api(originalRequest);
    }

    // 2. Token Refresh Interceptor (for 401 Unauthorized errors)
    if (error.response?.status === 401 && !originalRequest._retryRefresh) {
      originalRequest._retryRefresh = true;
      try {
        const savedUserStr = localStorage.getItem("fashion_auth_user");
        if (savedUserStr) {
          const user = JSON.parse(savedUserStr);
          let newToken = "mock_refreshed_jwt_token_" + Date.now();
          
          try {
            // Attempt to hit FastAPI refresh token endpoint if it exists
            const refreshRes = await axios.post(`${api.defaults.baseURL}/api/v1/auth/refresh`, {
              refresh_token: user.refreshToken ?? "mock_refresh_token"
            });
            newToken = refreshRes.data.access_token;
          } catch {
            // Simulated token refresh fallback since FastAPI main.py is focused on AI endpoints
            console.info("[API Auth] FastAPI OAuth endpoint offline, fallback to session simulation refresh.");
          }

          user.token = newToken;
          localStorage.setItem("fashion_auth_user", JSON.stringify(user));
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        if (typeof window !== "undefined") {
          localStorage.removeItem("fashion_auth_user");
          window.location.href = "/login";
        }
        return Promise.reject(refreshError);
      }
    }

    // 3. Error Normalization
    const errorMsg =
      error.response?.data?.detail ??
      error.response?.data?.message ??
      error.message ??
      "An unexpected connection error occurred.";

    console.error(`[API Network Error] URL: ${originalRequest.url} | Message: ${errorMsg}`);
    return Promise.reject(new Error(errorMsg));
  }
);

// ─── API Functions ────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  const res = await api.get<HealthResponse>("/");
  return res.data;
}

export async function postQuery(payload: QueryRequest): Promise<QueryResponse> {
  const res = await api.post<QueryResponse>("/api/v1/query", payload);
  return res.data;
}

export async function postStyleRecommendations(
  payload: StyleRecommendationRequest
): Promise<StyleRecommendationResponse> {
  const res = await api.post<StyleRecommendationResponse>(
    "/api/v1/recommendations/styles",
    payload
  );
  return res.data;
}

export async function postBrandRecommendations(
  payload: BrandRecommendationRequest
): Promise<BrandRecommendationResponse> {
  const res = await api.post<BrandRecommendationResponse>(
    "/api/v1/recommendations/brands",
    payload
  );
  return res.data;
}

export async function getTrendExplain(
  trend_name: string
): Promise<TrendExplainResponse> {
  const res = await api.get<TrendExplainResponse>(
    "/api/v1/trends/explain",
    { params: { trend_name } }
  );
  return res.data;
}

export async function getTrendForecast(
  season: string
): Promise<TrendForecastResponse> {
  const res = await api.get<TrendForecastResponse>(
    "/api/v1/trends/forecast",
    { params: { season } }
  );
  return res.data;
}

export async function getSemanticSearch(
  query: string
): Promise<SemanticSearchResponse> {
  const res = await api.get<SemanticSearchResponse>("/api/v1/search", {
    params: { query },
  });
  return res.data;
}

export default api;
