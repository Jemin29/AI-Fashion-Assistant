import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dashboard | AI Fashion",
  description:
    "Premium enterprise dashboard with hardware telemetry, recent outputs logs, usage charts, and system alerts.",
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
