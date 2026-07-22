import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Style Recommendations",
  description: "Personalized style and fashion combinations recommended by AI.",
  openGraph: {
    title: "Style Recommendations | AI Design Studio",
    description: "Personalized style and fashion combinations recommended by AI.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
