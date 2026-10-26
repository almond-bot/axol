import type Stripe from "stripe"
import { requestSummary } from "../_lib/requests"
import { MissingConfigError, errorMessage, requiredEnv, stripeClient } from "../_lib/stripe"

/** Log the request and, when configured, post it to the team's channel. A
 *  failed post throws, so Stripe retries the event. */
async function announce(session: Stripe.Checkout.Session, outcome: "paid" | "failed") {
  const text = requestSummary(session, outcome)
  console.log(text)
  const notifyUrl = process.env.DATA_COLLECTION_NOTIFY_URL
  if (!notifyUrl) return
  const response = await fetch(notifyUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!response.ok) throw new Error(`notify returned HTTP ${response.status}`)
}

/**
 * POST /api/stripe/webhook: Stripe's events for Data Collection checkouts.
 * This is where a request is fulfilled (announced to the team), not on the
 * customer's return page, which they may never reach.
 *
 * Subscribe the endpoint to checkout.session.completed,
 * checkout.session.async_payment_succeeded and
 * checkout.session.async_payment_failed.
 */
export async function POST(request: Request): Promise<Response> {
  const payload = await request.text()
  let event: Stripe.Event
  try {
    event = await stripeClient().webhooks.constructEventAsync(
      payload,
      request.headers.get("stripe-signature") ?? "",
      requiredEnv("STRIPE_WEBHOOK_SECRET")
    )
  } catch (error) {
    if (error instanceof MissingConfigError) {
      // A deploy without its secrets: fail loudly so Stripe keeps retrying.
      console.error(`stripe webhook: ${error.message}`)
      return new Response("Webhook not configured", { status: 500 })
    }
    return new Response("Invalid signature", { status: 400 })
  }

  try {
    switch (event.type) {
      case "checkout.session.completed":
      case "checkout.session.async_payment_succeeded": {
        const session = event.data.object
        if (session.metadata?.service !== "data_collection") break
        // A delayed payment method completes the session unpaid; its
        // async_payment_succeeded event follows once the money arrives.
        if (session.payment_status === "unpaid") break
        await announce(session, "paid")
        break
      }
      case "checkout.session.async_payment_failed": {
        const session = event.data.object
        if (session.metadata?.service !== "data_collection") break
        await announce(session, "failed")
        break
      }
    }
  } catch (error) {
    console.error(`stripe webhook ${event.id}: ${errorMessage(error)}`)
    return new Response("Handler failed", { status: 500 })
  }
  return new Response(null, { status: 200 })
}
