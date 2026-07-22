import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Executive Analytics",
  description: "Monitor AI fashion model performance, RAG retrieval scores, FID image accuracy, and GPU efficiency KPIs.",
  openGraph: {
    title: "Executive Analytics | AI Design Studio",
    description: "Monitor AI fashion model performance, RAG retrieval scores, FID image accuracy, and GPU efficiency KPIs.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
