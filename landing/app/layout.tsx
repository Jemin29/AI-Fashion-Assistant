import type { Metadata } from "next";
import { Inter, Plus_Jakarta_Sans, Geist_Mono } from "next/font/google";

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-plus-jakarta",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Fashion Creative Studio — Generate. Style. Create.",
  description:
    "The world's most advanced AI-powered fashion design platform. Generate haute couture looks, apply brand styles with LoRA, and explore trends with SDXL + ControlNet.",
  keywords: [
    "AI fashion",
    "fashion design AI",
    "SDXL fashion",
    "stable diffusion fashion",
    "AI clothes generator",
    "fashion assistant",
  ],
  openGraph: {
    title: "AI Fashion Creative Studio",
    description: "Generate stunning fashion designs with AI",
    type: "website",
  },
};

import { SplashScreen } from "@/components/SplashScreen";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${plusJakarta.variable} ${inter.variable} ${geistMono.variable}`}>
      <body className="antialiased">
        <SplashScreen />
        {children}
      </body>
    </html>
  );
}
