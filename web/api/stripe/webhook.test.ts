// @vitest-environment node
import Stripe from "stripe"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { POST } from "./webhook"

const SECRET = "whsec_unit_test"
// The webhook only verifies signatures locally; this never reaches Stripe.
const signer = new Stripe("unit-test-key")

function session(overrides: Record<string, unknown> = {}) {
  return {
    id: "cs_1",
    object: "checkout.session",
    livemode: false,
    payment_status: "paid",
    amount_total: 50000,
    currency: "usd",
    payment_intent: "pi_1",
    customer_details: { email: "lab@example.com" },
    metadata: {
      service: "data_collection",
      hours: "10",
      task: "Fold the towel.",
      example_kind: "video",
      example: "https://youtu.be/abc",
      operation: "teleop",
    },
    ...overrides,
  }
}

async function deliver(type: string, object: object, secret = SECRET) {
  const payload = JSON.stringify({ id: "evt_1", object: "event", type, data: { object } })
  const header = await signer.webhooks.generateTestHeaderStringAsync({ payload, secret })
  return POST(
    new Request("https://axol.almond.bot/api/stripe/webhook", {
      method: "POST",
      headers: { "stripe-signature": header },
      body: payload,
    })
  )
}

const notify = vi.fn()

beforeEach(() => {
  process.env.STRIPE_SECRET_KEY = "unit-test-key"
  process.env.STRIPE_WEBHOOK_SECRET = SECRET
  process.env.DATA_COLLECTION_NOTIFY_URL = "https://hooks.example.com/notify"
  notify.mockResolvedValue(new Response("ok"))
  vi.stubGlobal("fetch", notify)
  vi.spyOn(console, "log").mockImplementation(() => {})
  vi.spyOn(console, "error").mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  notify.mockReset()
})

const notified = () => JSON.parse(notify.mock.calls[0][1].body).text as string

describe("POST /api/stripe/webhook", () => {
  it("announces a paid request with everything the team needs", async () => {
    const response = await deliver("checkout.session.completed", session())
    expect(response.status).toBe(200)
    expect(notify).toHaveBeenCalledWith("https://hooks.example.com/notify", expect.anything())
    const text = notified()
    expect(text).toContain("New Data Collection request: 10 h, $500 paid")
    expect(text).toContain("lab@example.com")
    expect(text).toContain("Task: Fold the towel.")
    expect(text).toContain("Example (video): https://youtu.be/abc")
    expect(text).toContain("Teleoperation page")
    expect(text).toContain("https://dashboard.stripe.com/test/payments/pi_1")
  })

  it("waits for a delayed payment to succeed", async () => {
    await deliver("checkout.session.completed", session({ payment_status: "unpaid" }))
    expect(notify).not.toHaveBeenCalled()
    await deliver("checkout.session.async_payment_succeeded", session())
    expect(notified()).toContain("paid")
  })

  it("reports a failed delayed payment", async () => {
    await deliver("checkout.session.async_payment_failed", session({ payment_status: "unpaid" }))
    expect(notified()).toContain("Data Collection payment failed")
  })

  it("ignores other checkouts on the account", async () => {
    const response = await deliver("checkout.session.completed", session({ metadata: {} }))
    expect(response.status).toBe(200)
    expect(notify).not.toHaveBeenCalled()
  })

  it("rejects an event with a bad signature", async () => {
    const response = await deliver("checkout.session.completed", session(), "whsec_other")
    expect(response.status).toBe(400)
    expect(notify).not.toHaveBeenCalled()
  })

  it("asks Stripe to retry when the team can't be notified", async () => {
    notify.mockResolvedValue(new Response("down", { status: 503 }))
    expect((await deliver("checkout.session.completed", session())).status).toBe(500)
  })
})
