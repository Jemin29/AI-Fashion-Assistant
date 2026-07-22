import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Recommendations",
  description: "Explore personalized AI recommendations tailored to your aesthetic fashion DNA.",
  openGraph: {
    title: "AI Recommendations | AI Design Studio",
    description: "Explore personalized AI recommendations tailored to your aesthetic fashion DNA.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
