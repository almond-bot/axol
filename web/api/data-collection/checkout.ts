import { validateDataCollectionRequest } from "../../app/src/lib/data-collection"
import { returnTo } from "../_lib/requests"
import {
  PAYMENT_UNAVAILABLE,
  errorMessage,
  json,
  preflight,
  requiredEnv,
  stripeClient,
} from "../_lib/stripe"

/** Tags these sessions in the Stripe Dashboard. */
const INTEGRATION_ID = "axol-data-collection-rkqzmwtb"

/** Shown above the Pay button on Stripe's checkout page. */
const REFUND_NOTE =
  "You pay for the hours you request. If Almond delivers fewer usable hours, " +
  "we refund the difference."

/**
 * POST /api/data-collection/checkout: start a Stripe Checkout session for a
 * Teleop Data Collection request. The customer pays hours × the hourly price up
 * front. The request itself (task, example) travels as session and payment
 * metadata. Beaver's Stripe webhook reads it from there once the payment
 * succeeds, records the request and tells #sales.
 */
export async function POST(request: Request): Promise<Response> {
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return json({ error: "The request was not valid JSON." }, 400)
  }
  const result = validateDataCollectionRequest(body)
  if (!result.ok) return json({ error: result.error }, 400)
  const req = result.value

  const metadata = {
    service: "data_collection",
    hours: String(req.hours),
    task: req.task,
    example_kind: req.example?.kind ?? "",
    example: req.example?.value ?? "",
    operation: req.operation,
  }

  try {
    const session = await stripeClient().checkout.sessions.create({
      mode: "payment",
      line_items: [{ price: requiredEnv("STRIPE_DATA_COLLECTION_PRICE_ID"), quantity: req.hours }],
      success_url: returnTo(req.returnUrl, "paid"),
      cancel_url: returnTo(req.returnUrl, "cancelled"),
      customer_creation: "always",
      metadata,
      payment_intent_data: {
        description: `Teleop Data Collection, ${req.hours} h`,
        metadata,
      },
      custom_text: { submit: { message: REFUND_NOTE } },
      integration_identifier: INTEGRATION_ID,
    })
    if (!session.url) throw new Error(`session ${session.id} has no URL`)
    return json({ url: session.url })
  } catch (error) {
    console.error(`data-collection checkout: ${errorMessage(error)}`)
    return json({ error: PAYMENT_UNAVAILABLE }, 502)
  }
}

export const OPTIONS = preflight
