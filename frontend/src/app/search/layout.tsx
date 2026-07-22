import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Semantic Vector Search",
  description: "High-dimensional similarity matches directly over the ChromaDB vector collections.",
  openGraph: {
    title: "Semantic Vector Search | AI Design Studio",
    description: "High-dimensional similarity matches directly over the ChromaDB vector collections.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
