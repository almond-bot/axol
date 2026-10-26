import type { DataCollectionPrice, DataCollectionRequest } from "@/lib/data-collection"

/**
 * Calls to Almond's Data Collection functions (web/api), which hold the
 * Stripe key. They always run on axol.almond.bot, even when this panel is
 * served by `axol serve` from the robot. A Vercel deployment (production or
 * preview) calls its own functions so a preview can be tested end to end.
 */
export function almondApiOrigin(hostname = window.location.hostname): string {
  if (hostname === "axol.almond.bot" || hostname.endsWith(".vercel.app")) return ""
  return import.meta.env.VITE_ALMOND_API_ORIGIN || "https://axol.almond.bot"
}

const OFFLINE = "Can't reach almond.bot. Connect this computer to the internet, then try again."

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${almondApiOrigin()}${path}`, init)
  } catch {
    throw new Error(OFFLINE)
  }
  const body = (await response.json().catch(() => null)) as (T & { error?: string }) | null
  if (!response.ok || body == null) {
    throw new Error(body?.error ?? `Almond's server answered HTTP ${response.status}.`)
  }
  return body
}

export function fetchDataCollectionPrice(): Promise<DataCollectionPrice> {
  return call<DataCollectionPrice>("/api/data-collection/price")
}

/** Creates the Stripe Checkout session and returns its page URL. */
export async function startDataCollectionCheckout(request: DataCollectionRequest): Promise<string> {
  const { url } = await call<{ url: string }>("/api/data-collection/checkout", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  })
  return url
}
