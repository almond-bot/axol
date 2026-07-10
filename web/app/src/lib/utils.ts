import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Words that stay in their canonical casing wherever they appear in a label.
const ACRONYMS: Record<string, string> = {
  id: "ID",
  ip: "IP",
  url: "URL",
  vr: "VR",
  usb: "USB",
  fps: "FPS",
  hz: "Hz",
  ik: "IK",
  api: "API",
  tls: "TLS",
  com: "CoM",
}

/**
 * The app-wide label casing: sentence case (first word capitalized, the rest
 * lowercase) with acronyms in their canonical form — "Repo ID", "Teleop Hz",
 * "CoM x", "Max step rad". Use this instead of the CSS `capitalize` utility,
 * which Title-Cases Every Word.
 */
export function sentenceCase(label: string): string {
  const words = label.trim().split(/\s+/)
  return words
    .map((word, i) => {
      const canonical = ACRONYMS[word.toLowerCase()]
      if (canonical) return canonical
      return i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word
    })
    .join(" ")
}
