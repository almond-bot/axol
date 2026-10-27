import Stripe from "stripe"

/**
 * Shared plumbing for the Teleop Data Collection functions. They run on Vercel with
 * axol.almond.bot, never on a customer's robot: the Stripe key must stay on
 * Almond's side. Files under `_lib` are not routes.
 *
 * Environment (set as Vercel sensitive variables, per environment):
 * - STRIPE_SECRET_KEY: a restricted key (`rk_…`) with write access to
 *   Checkout Sessions and read access to Prices.
 * - STRIPE_DATA_COLLECTION_PRICE_ID: the hourly Price of the Teleop Data Collection
 *   product. The panel shows whatever amount it has.
 *
 * Fulfillment is not here: Beaver (beaver.almond.bot) already receives every
 * checkout on the account through its own Stripe webhook, and opens a Teleop
 * Data Collection request for a session tagged `metadata.service =
 * "data_collection"`.
 */

export class MissingConfigError extends Error {
  constructor(name: string) {
    super(`${name} is not set`)
    this.name = "MissingConfigError"
  }
}

export function requiredEnv(name: string): string {
  const value = process.env[name]
  if (!value) throw new MissingConfigError(name)
  return value
}

let client: Stripe | null = null

export function stripeClient(): Stripe {
  client ??= new Stripe(requiredEnv("STRIPE_SECRET_KEY"))
  return client
}

/** The control panel calls these functions from the robot's own address
 *  (any LAN host) as well as from axol.almond.bot. No cookies are involved,
 *  so any origin may call them. */
const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type",
}

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...CORS_HEADERS, ...headers },
  })
}

export function preflight(): Response {
  return new Response(null, {
    status: 204,
    headers: { ...CORS_HEADERS, "access-control-max-age": "86400" },
  })
}

/** Never put a key or an env dump in a log line: only the message. */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Shown to the customer when Stripe or the configuration fails. */
export const PAYMENT_UNAVAILABLE =
  "Payment isn't available right now. Try again later, or email hello@almond.bot."
