import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    fontSize: {
      "xs": ["0.75rem", { lineHeight: "1rem" }],
      "sm": ["0.875rem", { lineHeight: "1.25rem" }],
      "base": ["1rem", { lineHeight: "1.5rem" }],
      "lg": ["1.125rem", { lineHeight: "1.75rem" }],
      "xl": ["1.25rem", { lineHeight: "1.75rem" }],
      "2xl": ["1.5rem", { lineHeight: "2.25rem" }],
      "3xl": ["1.875rem", { lineHeight: "2.25rem" }],
      "4xl": ["2.25rem", { lineHeight: "2.5rem" }],
      "5xl": ["3rem", { lineHeight: "1" }],
      "6xl": ["3.75rem", { lineHeight: "1" }],
      "7xl": ["4.5rem", { lineHeight: "1" }],
    },
    spacing: {
      // Standard Tailwind scale keys strictly mapped to closest spacing tokens
      "0": "0px",
      "0.5": "8px",
      "1": "8px",
      "1.5": "8px",
      "2": "8px",      // Token: 8
      "2.5": "8px",
      "3": "16px",
      "3.5": "16px",
      "4": "16px",     // Token: 16
      "5": "24px",
      "6": "24px",     // Token: 24
      "7": "32px",
      "8": "32px",     // Token: 32
      "9": "40px",
      "10": "40px",    // Token: 40
      "11": "48px",
      "12": "48px",    // Token: 48
      "14": "64px",
      "16": "64px",    // Token: 64
      "20": "80px",    // Token: 80
      "24": "96px",    // Token: 96
      "px": "0px",
    },
    extend: {
      fontFamily: {
        sans: ["var(--font-plus-jakarta)", "sans-serif"],
        body: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      colors: {
        brand: {
          indigo: "var(--accent-indigo)",
          coral: "var(--accent-coral)",
          teal: "var(--accent-teal)",
        },
        surface: {
          deep: "var(--bg-deep)",
          card: "var(--bg-card)",
          elevated: "var(--bg-elevated)",
        },
        border: "var(--border-color)",
        input: "var(--input-color)",
        status: {
          success: "var(--status-success)",
          warning: "var(--status-warning)",
          danger: "var(--status-error)",
          info: "var(--status-info)",
        },
      },
      keyframes: {
        blob: {
          "0%, 100%": { transform: "translate(0, 0) scale(1)" },
          "33%": { transform: "translate(30px, -50px) scale(1.1)" },
          "66%": { transform: "translate(-20px, 20px) scale(0.9)" },
        },
        gradientShift: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        scroll: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        shimmer: {
          from: { backgroundPosition: "-200% 0" },
          to: { backgroundPosition: "200% 0" },
        },
        fadeIn: {
          from: { opacity: "0", transform: "translateY(10px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        blob: "blob 7s infinite",
        gradientShift: "gradientShift 6s ease infinite",
        scroll: "scroll 20s linear infinite",
        shimmer: "shimmer 2s infinite",
        fadeIn: "fadeIn 0.5s ease-out",
      },
      backgroundSize: {
        "300%": "300% 300%",
        "200%": "200% 100%",
      },
    },
  },
  plugins: [],
};

export default config;
