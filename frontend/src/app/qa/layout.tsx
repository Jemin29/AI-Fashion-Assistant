import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fashion Q&A",
  description: "Ask fashion and styling questions powered by RAG-enhanced language models.",
  openGraph: {
    title: "Fashion Q&A | AI Design Studio",
    description: "Ask fashion and styling questions powered by RAG-enhanced language models.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
