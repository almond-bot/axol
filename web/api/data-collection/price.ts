import type { DataCollectionPrice } from "../../app/src/lib/data-collection"
import {
  PAYMENT_UNAVAILABLE,
  errorMessage,
  json,
  preflight,
  requiredEnv,
  stripeClient,
} from "../_lib/stripe"

/** GET /api/data-collection/price: the hourly price the panel shows. */
export async function GET(): Promise<Response> {
  try {
    const price = await stripeClient().prices.retrieve(
      requiredEnv("STRIPE_DATA_COLLECTION_PRICE_ID")
    )
    if (!price.active || price.unit_amount == null) {
      throw new Error(`price ${price.id} is inactive or has no unit amount`)
    }
    const body: DataCollectionPrice = { unitAmount: price.unit_amount, currency: price.currency }
    // A price change shows up within five minutes.
    return json(body, 200, { "cache-control": "public, s-maxage=300, stale-while-revalidate=3600" })
  } catch (error) {
    console.error(`data-collection price: ${errorMessage(error)}`)
    return json({ error: PAYMENT_UNAVAILABLE }, 502)
  }
}

export const OPTIONS = preflight
