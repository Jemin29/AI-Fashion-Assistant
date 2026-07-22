import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trend Forecasting",
  description: "Forecast future design element and seasonal trend momentum using AI metrics.",
  openGraph: {
    title: "Trend Forecasting | AI Design Studio",
    description: "Forecast future design element and seasonal trend momentum using AI metrics.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
