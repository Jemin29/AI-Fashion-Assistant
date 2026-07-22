import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Generation Studio | AI Fashion",
  description:
    "Midjourney-style generation studio with advanced weights, aspect toggles, slider comparison modes, and fullscreen logs.",
};

export default function GenerationStudioLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
