import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Design System | AI Fashion",
  description:
    "Premium enterprise design system — tokens, typography, and components inspired by Apple, Vercel, Linear, Midjourney, and Adobe Firefly.",
};

export default function DesignSystemLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
