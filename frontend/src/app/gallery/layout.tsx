import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fashion Gallery",
  description: "Browse the Pinterest-style fashion masonry archive in the AI Design Studio.",
  openGraph: {
    title: "Fashion Gallery | AI Design Studio",
    description: "Browse the Pinterest-style fashion masonry archive.",
  },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
