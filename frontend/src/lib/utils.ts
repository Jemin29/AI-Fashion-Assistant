import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a label string from snake_case to Title Case */
export function formatLabel(str: string): string {
  return str
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/** Truncate text to a given length */
export function truncate(str: string, max = 120): string {
  return str.length > max ? str.slice(0, max) + "…" : str;
}
