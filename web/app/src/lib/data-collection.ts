/**
 * The Teleop Data Collection request a customer sends from the control panel, and
 * the rules it must meet. Shared by the panel's form and the checkout
 * function in web/api, so both reject the same input with the same message.
 * Keep it free of browser APIs and `@/` imports: the serverless bundle
 * imports it directly.
 *
 * The hourly price is not here. It lives on a Stripe Price and the panel
 * reads it at runtime, so it can change without a release.
 */

export const MIN_HOURS = 1
export const MAX_HOURS = 1000
/** Stripe caps each metadata value at 500 characters. */
export const MAX_TASK_LENGTH = 500
export const MAX_LINK_LENGTH = 500

export type ExampleKind = "video" | "dataset"

/**
 * The two Teleop Data Collection services, told apart by what the customer
 * gets:
 * - `task`: teleop as a service. An Almond operator controls the customer's
 *   own Axol to finish a one-off task. The output is the task done. Billed by
 *   the minute: the customer prepays hours and Almond refunds unused minutes.
 * - `dataset`: a training dataset, collected one of the `COLLECTION_METHODS`
 *   ways. The output is the dataset. Billed per hour of usable data, however
 *   long it takes to record.
 * Each is charged at its own hourly Stripe Price (see `priceKey`).
 */
export const DATA_COLLECTION_SERVICES = ["task", "dataset"] as const
export type DataCollectionService = (typeof DATA_COLLECTION_SERVICES)[number]

/** The sidebar page that sells each service. */
export const REQUEST_PAGE = {
  task: "request-teleoperator",
  dataset: "request-data",
} as const satisfies Record<DataCollectionService, string>

/**
 * How Almond collects a training dataset:
 * - `almond_axol`: Almond sets up the environment and records it with teleop
 *   on its own Axol.
 * - `your_axol`: Almond teleoperates the customer's Axol remotely, in the
 *   customer's environment.
 */
export const COLLECTION_METHODS = ["almond_axol", "your_axol"] as const
export type CollectionMethod = (typeof COLLECTION_METHODS)[number]

/**
 * Which hourly Stripe Price a request is charged at: the teleoperator's for a
 * one-off task, else the collection method's. The amounts live in Stripe.
 */
export const PRICE_KEYS = ["teleoperator", ...COLLECTION_METHODS] as const
export type PriceKey = (typeof PRICE_KEYS)[number]

export function priceKey(
  service: DataCollectionService,
  method: CollectionMethod | null
): PriceKey {
  return service === "task" ? "teleoperator" : (method ?? "almond_axol")
}

/** The pages that show the Teleop Data Collection card. */
export const DATA_COLLECTION_OFFER_OPERATIONS = ["overview", "teleop", "collect-data"] as const
export type DataCollectionOfferOperation = (typeof DATA_COLLECTION_OFFER_OPERATIONS)[number]

/** Where a request was made from: a request page itself, or the page whose
 *  card led there. Sent as the checkout's `operation` metadata, which Beaver
 *  shows with the request. */
export const DATA_COLLECTION_OPERATIONS = [
  ...Object.values(REQUEST_PAGE),
  ...DATA_COLLECTION_OFFER_OPERATIONS,
] as const
export type DataCollectionOperation = (typeof DATA_COLLECTION_OPERATIONS)[number]

/** The `?from=` of a link to a service's request page, or that page itself
 *  when it names no page with the card. */
export function requestSource(
  from: string | null,
  service: DataCollectionService
): DataCollectionOperation {
  return (DATA_COLLECTION_OFFER_OPERATIONS as readonly (string | null)[]).includes(from)
    ? (from as DataCollectionOfferOperation)
    : REQUEST_PAGE[service]
}

export interface DataCollectionRequest {
  service: DataCollectionService
  /** How a dataset is collected; null for a one-off task. */
  method: CollectionMethod | null
  hours: number
  task: string
  /** A video or sample dataset of the task; optional. */
  example: { kind: ExampleKind; value: string } | null
  /** The page the request came from, for Almond's records. */
  operation: DataCollectionOperation
  /** Where Stripe sends the customer back to, after paying or cancelling. */
  returnUrl: string
}

/** The hourly price, as the price endpoint reports it. */
export interface DataCollectionPrice {
  /** In the currency's smallest unit (cents for USD). */
  unitAmount: number
  currency: string
}

export type Validation<T> = { ok: true; value: T } | { ok: false; error: string }

/** A Hugging Face dataset id (`owner/name`) from an id or a dataset URL. */
export function parseHuggingFaceDataset(input: string): string | null {
  const text = input.trim()
  const fromUrl = /^https?:\/\/(?:www\.)?huggingface\.co\/datasets\/([^/?#\s]+\/[^/?#\s]+)/i.exec(
    text
  )
  const id = fromUrl ? fromUrl[1] : text
  return /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(id) ? id : null
}

/** An http(s) link to a video, or null. */
export function parseVideoUrl(input: string): string | null {
  const text = input.trim()
  try {
    const url = new URL(text)
    return url.protocol === "https:" || url.protocol === "http:" ? url.toString() : null
  } catch {
    return null
  }
}

function isOperation(value: unknown): value is DataCollectionOperation {
  return (DATA_COLLECTION_OPERATIONS as readonly unknown[]).includes(value)
}

/**
 * Check a request from the form (or an untrusted request body). Errors are
 * written for the customer: they are shown in the form as they are.
 */
export function validateDataCollectionRequest(raw: unknown): Validation<DataCollectionRequest> {
  const body = (raw ?? {}) as Record<string, unknown>
  const example = (body.example ?? {}) as Record<string, unknown>

  const service = body.service
  if (!(DATA_COLLECTION_SERVICES as readonly unknown[]).includes(service)) {
    return { ok: false, error: "Choose a one-off task or a training dataset." }
  }
  let method: CollectionMethod | null = null
  if (service === "dataset") {
    if (!(COLLECTION_METHODS as readonly unknown[]).includes(body.method)) {
      return { ok: false, error: "Choose how Almond collects the data." }
    }
    method = body.method as CollectionMethod
  }

  const hours = Number(body.hours)
  if (!Number.isInteger(hours) || hours < MIN_HOURS || hours > MAX_HOURS) {
    return {
      ok: false,
      error: `Enter a whole number of hours from ${MIN_HOURS} to ${MAX_HOURS}.`,
    }
  }

  const task = typeof body.task === "string" ? body.task.trim() : ""
  if (!task) return { ok: false, error: "Describe the task you want recorded." }
  if (task.length > MAX_TASK_LENGTH) {
    return { ok: false, error: `Keep the task description to ${MAX_TASK_LENGTH} characters.` }
  }

  // The example is optional: an empty field sends none. One that is filled in
  // must be a link or a dataset that opens.
  const kind = example.kind
  const exampleText = typeof example.value === "string" ? example.value.trim() : ""
  let parsedExample: DataCollectionRequest["example"] = null
  if (exampleText) {
    let value: string | null
    if (kind === "video") {
      value = parseVideoUrl(exampleText)
      if (!value) return { ok: false, error: "Enter a link to a video of the task." }
    } else if (kind === "dataset") {
      value = parseHuggingFaceDataset(exampleText)
      if (!value) {
        return {
          ok: false,
          error: "Enter a Hugging Face dataset as owner/name, or paste its link.",
        }
      }
    } else {
      return { ok: false, error: "Choose a video link or a Hugging Face dataset." }
    }
    if (value.length > MAX_LINK_LENGTH) {
      return { ok: false, error: `Use a link of ${MAX_LINK_LENGTH} characters or fewer.` }
    }
    parsedExample = { kind, value }
  }

  const operation = body.operation
  if (!isOperation(operation)) return { ok: false, error: "Unknown page." }

  const returnUrl = parseVideoUrl(typeof body.returnUrl === "string" ? body.returnUrl : "")
  if (!returnUrl) return { ok: false, error: "Unknown return page." }

  return {
    ok: true,
    value: {
      service: service as DataCollectionService,
      method,
      hours,
      task,
      example: parsedExample,
      operation,
      returnUrl,
    },
  }
}

/** "$50" for 5000 USD cents; whole amounts drop the decimals. */
export function formatMoney(amount: number, currency: string): string {
  const major = amount / 100
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency.toUpperCase(),
    minimumFractionDigits: Number.isInteger(major) ? 0 : 2,
  }).format(major)
}
