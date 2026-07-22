import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sketch Studio | AI Fashion",
  description:
    "Adobe Firefly-quality Sketch Studio with brush size controls, image drag & drop triggers, and ControlNet parameters.",
};

export default function SketchStudioLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
