import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Brand Studio | AI Fashion",
  description:
    "Luxury Brand Studio for fine-tuning LoRA weights, comparing design elements, and blending brand models.",
};

export default function LuxuryBrandStudioLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
