import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Adapted from shadcndashboard/src/lib/utils.ts (MIT); centralizes class merging.
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
