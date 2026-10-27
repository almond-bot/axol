import {
  priceKey,
  validateDataCollectionRequest,
  type CollectionMethod,
  type DataCollectionRequest,
  type DataCollectionService,
} from "../../app/src/lib/data-collection.js"
import { returnTo } from "../_lib/requests.js"
import {
  PAYMENT_UNAVAILABLE,
  errorMessage,
  json,
  preflight,
  priceIdFor,
  stripeClient,
} from "../_lib/stripe.js"

/** Tags these sessions in the Stripe Dashboard. */
const INTEGRATION_ID = "axol-data-collection-rkqzmwtb"

/** Shown above the Pay button on Stripe's checkout page. */
const REFUND_NOTE: Record<DataCollectionService, string> = {
  task:
    "You prepay the hours you request. Almond bills by the minute and refunds " +
    "any time you don't use.",
  dataset:
    "You pay for the hours you request. If Almond delivers fewer usable hours, " +
    "we refund the difference.",
}

const METHOD: Record<CollectionMethod, string> = {
  almond_axol: "collected by Almond with Axol",
  your_axol: "collected on your Axol",
}

/** The payment's description in Stripe and on the receipt. */
function description(req: DataCollectionRequest): string {
  if (req.method == null) return `Teleoperator, one-off task, ${req.hours} h`
  return `Teleop Data Collection, training dataset, ${METHOD[req.method]}, ${req.hours} h`
}

/**
 * POST /api/data-collection/checkout: start a Stripe Checkout session for a
 * Teleop Data Collection request: a one-off task done on the customer's Axol,
 * or a training dataset. Each charges hours × its own hourly price up front.
 * The request itself (task, method, example) travels as session and payment
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

  // Beaver keys on `service`. `request_type` says which of the two it is, and
  // `collection_method` how a dataset is collected (empty for a task).
  const metadata = {
    service: "data_collection",
    request_type: req.service,
    collection_method: req.method ?? "",
    hours: String(req.hours),
    task: req.task,
    example_kind: req.example?.kind ?? "",
    example: req.example?.value ?? "",
    operation: req.operation,
  }

  try {
    const session = await stripeClient().checkout.sessions.create({
      mode: "payment",
      line_items: [{ price: priceIdFor(priceKey(req.service, req.method)), quantity: req.hours }],
      success_url: returnTo(req.returnUrl, "paid"),
      cancel_url: returnTo(req.returnUrl, "cancelled"),
      customer_creation: "always",
      metadata,
      payment_intent_data: {
        description: description(req),
        metadata,
      },
      custom_text: { submit: { message: REFUND_NOTE[req.service] } },
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
