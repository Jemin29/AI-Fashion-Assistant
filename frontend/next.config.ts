import type { NextConfig } from "next";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // Proxy /api/* requests to FastAPI backend during development
  // This avoids CORS issues when developing locally
  async rewrites() {
    return [
      {
        source: "/proxy/:path*",
        destination: `${API_URL}/:path*`,
      },
    ];
  },

  // Security & performance headers
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },

  // Turbopack is now the default in Next.js 15
  // Uncomment below only if you need Webpack-specific config
  // webpack: (config) => { return config; },
};

export default nextConfig;
