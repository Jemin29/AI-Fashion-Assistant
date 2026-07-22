import type { Metadata } from "next";
import { Inter, Geist } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/layout/providers";
import { LayoutWrapper } from "@/components/layout/layout-wrapper";
import { cn } from "@/lib/utils";

const geist = Geist({subsets:['latin'],variable:'--font-sans'}) as any;

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  using: "swap",
} as any);

export const metadata: Metadata = {
  title: {
    default: "AI Fashion Design Assistant",
    template: "%s | AI Fashion",
  },
  description:
    "Production-grade AI-powered fashion assistant with RAG, trend forecasting, style & brand recommendations, and semantic vector search.",
  keywords: ["fashion", "AI", "RAG", "trends", "style", "recommendations"],
  authors: [{ name: "AI Fashion Team" }],
  openGraph: {
    type: "website",
    title: "AI Fashion Design Assistant",
    description:
      "Context-aware fashion recommendations powered by RAG and ChromaDB.",
    siteName: "AI Fashion",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className={cn("font-sans", (geist as any).variable)}>
      <body className={`${(inter as any).variable} font-sans antialiased`}>
        <Providers>
          {/* Background gradient blobs */}
          <div className="pointer-events-none fixed inset-0 overflow-hidden">
            <div className="absolute -top-40 -left-40 h-96 w-96 rounded-full bg-violet-600/10 blur-3xl" />
            <div className="absolute top-1/3 -right-40 h-80 w-80 rounded-full bg-fuchsia-600/8 blur-3xl" />
            <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-indigo-600/8 blur-3xl" />
          </div>

          <LayoutWrapper>
            {children}
          </LayoutWrapper>
        </Providers>
      </body>
    </html>
  );
}
