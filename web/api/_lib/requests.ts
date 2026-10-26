import type Stripe from "stripe"
import { formatMoney } from "../../app/src/lib/data-collection"

/** Helpers for the Teleop Data Collection routes, kept out of the route files so
 *  each route exports only its HTTP handlers. */

/** `returnUrl` with `data_collection=<outcome>`, replacing an earlier one. */
export function returnTo(returnUrl: string, outcome: "paid" | "cancelled"): string {
  const url = new URL(returnUrl)
  url.searchParams.set("data_collection", outcome)
  return url.toString()
}

const OPERATION_LABELS: Record<string, string> = {
  teleop: "Teleoperation",
  "collect-data": "Collect data",
}

/** The message Almond's team receives for one Teleop Data Collection request. */
export function requestSummary(
  session: Stripe.Checkout.Session,
  outcome: "paid" | "failed"
): string {
  const m = session.metadata ?? {}
  const total =
    session.amount_total != null && session.currency
      ? formatMoney(session.amount_total, session.currency)
      : "unknown amount"
  const paymentIntent =
    typeof session.payment_intent === "string" ? session.payment_intent : session.payment_intent?.id
  const dashboard = paymentIntent
    ? `https://dashboard.stripe.com/${session.livemode ? "" : "test/"}payments/${paymentIntent}`
    : `checkout session ${session.id}`
  return [
    outcome === "paid"
      ? `New Teleop Data Collection request: ${m.hours} h, ${total} paid`
      : `Teleop Data Collection payment failed: ${m.hours} h, ${total}`,
    `Customer: ${session.customer_details?.email ?? "no email"}`,
    `Task: ${m.task}`,
    `Example (${m.example_kind === "dataset" ? "Hugging Face dataset" : "video"}): ${m.example}`,
    `Requested from: ${OPERATION_LABELS[m.operation] ?? m.operation} page`,
    `Stripe: ${dashboard}`,
  ].join("\n")
}
