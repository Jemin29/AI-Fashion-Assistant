import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Brand Recommendations",
  description: "Identify and match clothing brands that match your target aesthetic profile.",
  openGraph: {
    title: "Brand Recommendations | AI Design Studio",
    description: "Identify and match clothing brands that match your target aesthetic profile.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
