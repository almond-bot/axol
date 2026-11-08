import {
  PRICE_KEYS,
  type DataCollectionPrice,
  type PriceKey,
} from "../../app/src/lib/data-collection.js"
import {
  PAYMENT_UNAVAILABLE,
  errorMessage,
  json,
  preflight,
  priceIdFor,
  stripeClient,
} from "../_lib/stripe.js"

/** GET /api/data-collection/price?price=teleoperator|almond_axol|your_axol:
 *  an hourly price the panel shows. Without `price` (panels that predate
 *  Request teleoperator) it is the Almond's Axol data price. */
export async function GET(request?: Request): Promise<Response> {
  const asked = request ? new URL(request.url).searchParams.get("price") : null
  const key: PriceKey = (PRICE_KEYS as readonly (string | null)[]).includes(asked)
    ? (asked as PriceKey)
    : "almond_axol"
  try {
    const price = await stripeClient().prices.retrieve(priceIdFor(key))
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
