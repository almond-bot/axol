// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const create = vi.fn()
const retrieve = vi.fn()
vi.mock("../_lib/stripe.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../_lib/stripe.js")>()
  return {
    ...actual,
    stripeClient: () => ({ checkout: { sessions: { create } }, prices: { retrieve } }),
  }
})

const { POST, OPTIONS } = await import("./checkout.js")
const { GET } = await import("./price.js")

const request = {
  service: "dataset",
  method: "your_axol",
  hours: 12,
  task: "Fold the towel in half twice.",
  example: { kind: "dataset", value: "https://huggingface.co/datasets/org/towels" },
  operation: "collect-data",
  returnUrl: "https://192.168.1.50:8001/control/operations?data_collection=cancelled#top",
}

function post(body: unknown) {
  return POST(
    new Request("https://axol.almond.bot/api/data-collection/checkout", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: typeof body === "string" ? body : JSON.stringify(body),
    })
  )
}

beforeEach(() => {
  process.env.STRIPE_DATA_COLLECTION_PRICE_ID = "price_hourly"
  process.env.STRIPE_TELEOPERATOR_PRICE_ID = "price_teleoperator"
  process.env.STRIPE_REMOTE_DATA_COLLECTION_PRICE_ID = "price_remote"
  vi.spyOn(console, "error").mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
  create.mockReset()
  retrieve.mockReset()
})

describe("POST /api/data-collection/checkout", () => {
  it("charges hours × the configured price and carries the request as metadata", async () => {
    create.mockResolvedValue({ id: "cs_1", url: "https://checkout.stripe.com/c/pay/cs_1" })
    const response = await post(request)

    expect(response.status).toBe(200)
    expect(response.headers.get("access-control-allow-origin")).toBe("*")
    expect(await response.json()).toEqual({ url: "https://checkout.stripe.com/c/pay/cs_1" })

    const params = create.mock.calls[0][0]
    expect(params.mode).toBe("payment")
    expect(params.line_items).toEqual([{ price: "price_remote", quantity: 12 }])
    expect(params).not.toHaveProperty("payment_method_types")
    expect(params.success_url).toBe(
      "https://192.168.1.50:8001/control/operations?data_collection=paid#top"
    )
    expect(params.cancel_url).toBe(
      "https://192.168.1.50:8001/control/operations?data_collection=cancelled#top"
    )
    expect(params.metadata).toEqual({
      service: "data_collection",
      request_type: "dataset",
      collection_method: "your_axol",
      hours: "12",
      task: "Fold the towel in half twice.",
      example_kind: "dataset",
      example: "org/towels",
      operation: "collect-data",
    })
    expect(params.payment_intent_data.metadata).toEqual(params.metadata)
    expect(params.payment_intent_data.description).toBe(
      "Teleop Data Collection, training dataset, collected on your Axol, 12 h"
    )
    expect(params.custom_text.submit.message).toContain("fewer usable hours")
  })

  it("bills a one-off task on the teleoperator price", async () => {
    create.mockResolvedValue({ id: "cs_3", url: "https://checkout.stripe.com/c/pay/cs_3" })
    expect((await post({ ...request, service: "task", hours: 2 })).status).toBe(200)
    const params = create.mock.calls[0][0]
    expect(params.line_items).toEqual([{ price: "price_teleoperator", quantity: 2 }])
    expect(params.metadata.request_type).toBe("task")
    expect(params.metadata.collection_method).toBe("")
    expect(params.payment_intent_data.description).toBe("Teleoperator credit, one-off task, 2 h")
    expect(params.custom_text.submit.message).toContain("by the minute")
    expect(params.custom_text.submit.message).toContain("refunds any credit")
  })

  it("charges data on Almond's Axol at its own price", async () => {
    create.mockResolvedValue({ id: "cs_4", url: "https://checkout.stripe.com/c/pay/cs_4" })
    expect((await post({ ...request, method: "almond_axol" })).status).toBe(200)
    const params = create.mock.calls[0][0]
    expect(params.line_items).toEqual([{ price: "price_hourly", quantity: 12 }])
    expect(params.payment_intent_data.description).toBe(
      "Teleop Data Collection, training dataset, collected by Almond with Axol, 12 h"
    )
  })

  it("sends an empty example when there is none", async () => {
    create.mockResolvedValue({ id: "cs_2", url: "https://checkout.stripe.com/c/pay/cs_2" })
    expect((await post({ ...request, example: { kind: "video", value: "" } })).status).toBe(200)
    const { metadata } = create.mock.calls[0][0]
    expect(metadata.example_kind).toBe("")
    expect(metadata.example).toBe("")
  })

  it("rejects an invalid request without calling Stripe", async () => {
    const response = await post({ ...request, hours: 0 })
    expect(response.status).toBe(400)
    expect((await response.json()).error).toContain("whole number of hours")
    expect(create).not.toHaveBeenCalled()
  })

  it("rejects a body that isn't JSON", async () => {
    expect((await post("not json")).status).toBe(400)
  })

  it("hides Stripe errors from the customer", async () => {
    create.mockRejectedValue(new Error("No such price: 'price_hourly'"))
    const response = await post(request)
    expect(response.status).toBe(502)
    expect((await response.json()).error).toContain("hello@almond.bot")
  })

  it("answers CORS preflight", () => {
    const response = OPTIONS()
    expect(response.status).toBe(204)
    expect(response.headers.get("access-control-allow-methods")).toContain("POST")
  })
})

describe("GET /api/data-collection/price", () => {
  it("reports the Stripe price's amount and currency", async () => {
    retrieve.mockResolvedValue({
      id: "price_hourly",
      active: true,
      unit_amount: 5000,
      currency: "usd",
    })
    const response = await GET()
    expect(retrieve).toHaveBeenCalledWith("price_hourly")
    expect(await response.json()).toEqual({ unitAmount: 5000, currency: "usd" })
  })

  it("reads the price asked for", async () => {
    retrieve.mockResolvedValue({ id: "p", active: true, unit_amount: 1500, currency: "usd" })
    const url = "https://axol.almond.bot/api/data-collection/price"
    for (const key of ["teleoperator", "almond_axol", "your_axol", "bogus"]) {
      await GET(new Request(`${url}?price=${key}`))
    }
    expect(retrieve.mock.calls).toEqual([
      ["price_teleoperator"],
      ["price_hourly"],
      ["price_remote"],
      ["price_hourly"],
    ])
  })

  it("fails when the price is archived", async () => {
    retrieve.mockResolvedValue({
      id: "price_hourly",
      active: false,
      unit_amount: 5000,
      currency: "usd",
    })
    expect((await GET()).status).toBe(502)
  })
})
