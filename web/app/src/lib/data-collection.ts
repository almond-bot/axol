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

/** The operations whose pages offer Teleop Data Collection. */
export const DATA_COLLECTION_OPERATIONS = ["teleop", "collect-data"] as const
export type DataCollectionOperation = (typeof DATA_COLLECTION_OPERATIONS)[number]

export interface DataCollectionRequest {
  hours: number
  task: string
  example: { kind: ExampleKind; value: string }
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

  const kind = example.kind
  const exampleText = typeof example.value === "string" ? example.value : ""
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
    return { ok: false, error: "Add a video or a Hugging Face dataset of the task." }
  }
  if (value.length > MAX_LINK_LENGTH) {
    return { ok: false, error: `Use a link of ${MAX_LINK_LENGTH} characters or fewer.` }
  }

  const operation = body.operation
  if (!isOperation(operation)) return { ok: false, error: "Unknown page." }

  const returnUrl = parseVideoUrl(typeof body.returnUrl === "string" ? body.returnUrl : "")
  if (!returnUrl) return { ok: false, error: "Unknown return page." }

  return { ok: true, value: { hours, task, example: { kind, value }, operation, returnUrl } }
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
