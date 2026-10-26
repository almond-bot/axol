import { useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { ArrowRight, Database, Loader2, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  MAX_HOURS,
  MAX_TASK_LENGTH,
  MIN_HOURS,
  formatMoney,
  validateDataCollectionRequest,
  type DataCollectionOperation,
  type DataCollectionPrice,
  type ExampleKind,
} from "@/lib/data-collection"
import { fetchDataCollectionPrice, startDataCollectionCheckout } from "@/lib/data-collection-api"
import { cn } from "@/lib/utils"

const HIDDEN_KEY = "axolDataCollectionOfferHidden"

function readHidden(): boolean {
  try {
    return localStorage.getItem(HIDDEN_KEY) === "1"
  } catch {
    return false
  }
}

const LEAD: Record<DataCollectionOperation, string> = {
  teleop: "Need training data for a task?",
  "collect-data": "Need more episodes than you can record?",
}

/** The current page without a `data_collection` result, for Stripe to
 *  send the customer back to. */
function returnUrl(): string {
  const url = new URL(window.location.href)
  url.searchParams.delete("data_collection")
  return url.toString()
}

/**
 * The Teleop Data Collection offer on the Teleoperation and Collect data pages: a
 * short card that opens the request form, which hands off to Stripe's
 * checkout page. The hourly price comes from Stripe; without a connection to
 * almond.bot the card still shows, without a number.
 */
export function DataCollectionOffer({ operation }: { operation: DataCollectionOperation }) {
  const [hidden, setHidden] = useState(readHidden)
  const [open, setOpen] = useState(false)
  const [price, setPrice] = useState<DataCollectionPrice | null>(null)

  useEffect(() => {
    if (hidden) return
    let active = true
    fetchDataCollectionPrice()
      .then((p) => {
        if (active) setPrice(p)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [hidden])

  if (hidden) return null

  function hide() {
    setHidden(true)
    try {
      localStorage.setItem(HIDDEN_KEY, "1")
    } catch {
      // Blocked storage: it stays hidden for this page view.
    }
  }

  return (
    <>
      <div className="relative flex flex-col gap-3 rounded-xl border border-[#eff483]/20 bg-gradient-to-br from-[#eff483]/[0.07] to-transparent p-4 sm:flex-row sm:items-center">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-[#eff483]/10">
          <Database className="size-4 text-[#eff483]" />
        </div>
        <div className="min-w-0 flex-1 pr-6 sm:pr-0">
          <p className="text-sm font-medium text-white">{LEAD[operation]}</p>
          <p className="mt-0.5 text-xs leading-relaxed text-white/55">
            Almond collects labeled data for your task and delivers it in LeRobot format, ready for
            training.{" "}
            {price ? (
              <span className="text-white/80">
                {formatMoney(price.unitAmount, price.currency)} per hour of usable data.
              </span>
            ) : (
              "Priced per hour of usable data."
            )}
          </p>
        </div>
        <Button
          size="sm"
          className="shrink-0 self-start sm:self-auto"
          onClick={() => setOpen(true)}
        >
          Request Teleop Data Collection
          <ArrowRight />
        </Button>
        <button
          type="button"
          onClick={hide}
          aria-label="Hide this offer"
          title="Hide this offer"
          className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-md text-white/30 transition-colors hover:bg-white/[0.06] hover:text-white/70 sm:static sm:self-start"
        >
          <X className="size-3.5" />
        </button>
      </div>
      {open && <RequestDialog operation={operation} price={price} onClose={() => setOpen(false)} />}
    </>
  )
}

export function RequestDialog({
  operation,
  price,
  onClose,
}: {
  operation: DataCollectionOperation
  price: DataCollectionPrice | null
  onClose: () => void
}) {
  const [hours, setHours] = useState("10")
  const [task, setTask] = useState("")
  const [exampleKind, setExampleKind] = useState<ExampleKind>("video")
  const [example, setExample] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  // An error describes the last attempt; any edit starts a new one.
  const edit =
    <T,>(set: (value: T) => void) =>
    (value: T) => {
      set(value)
      setError(null)
    }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !submitting && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose, submitting])

  const hoursNumber = Number(hours)
  const total =
    price && Number.isInteger(hoursNumber) && hoursNumber >= MIN_HOURS && hoursNumber <= MAX_HOURS
      ? formatMoney(price.unitAmount * hoursNumber, price.currency)
      : null

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    const checked = validateDataCollectionRequest({
      hours: hoursNumber,
      task,
      example: { kind: exampleKind, value: example },
      operation,
      returnUrl: returnUrl(),
    })
    if (!checked.ok) {
      setError(checked.error)
      return
    }
    setError(null)
    setSubmitting(true)
    try {
      window.location.assign(await startDataCollectionCheckout(checked.value))
    } catch (err) {
      setError(String(err).replace(/^Error:\s*/, ""))
      setSubmitting(false)
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={() => !submitting && onClose()} aria-hidden />
      <form
        onSubmit={submit}
        role="dialog"
        aria-modal="true"
        aria-labelledby="data-collection-title"
        className="relative z-10 my-auto flex w-full max-w-lg flex-col rounded-2xl border border-white/10 bg-[#161616] shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Database className="size-4 text-[#eff483]" />
            <span id="data-collection-title" className="font-heading text-base font-semibold">
              Request Teleop Data Collection
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="text-white/40 transition-colors hover:text-white/80"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="flex flex-col gap-5 p-5">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="dc-hours">Hours of usable data</Label>
            <div className="flex items-center gap-3">
              <Input
                id="dc-hours"
                type="number"
                inputMode="numeric"
                min={MIN_HOURS}
                max={MAX_HOURS}
                step={1}
                value={hours}
                onChange={(e) => edit(setHours)(e.target.value)}
                className="w-28"
              />
              {price && (
                <span className="text-xs text-white/50">
                  {formatMoney(price.unitAmount, price.currency)} per hour
                  {total && (
                    <>
                      {" · "}
                      <span className="text-white/85">Total {total}</span>
                    </>
                  )}
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex items-baseline justify-between">
              <Label htmlFor="dc-task">Task</Label>
              <span
                className={cn(
                  "text-[0.7rem] text-white/35",
                  task.length > MAX_TASK_LENGTH && "text-red-300"
                )}
              >
                {task.length}/{MAX_TASK_LENGTH}
              </span>
            </div>
            <textarea
              id="dc-task"
              rows={4}
              value={task}
              onChange={(e) => edit(setTask)(e.target.value)}
              placeholder="For example: pick up each mug from the rack and place it upside down on the tray."
              className="w-full resize-y rounded-md border border-input bg-white/[0.02] px-3 py-2 text-base text-foreground transition-colors outline-none placeholder:text-white/30 focus-visible:border-ring/70 focus-visible:ring-2 focus-visible:ring-ring/30 sm:text-sm"
            />
            <p className="text-xs text-white/40">
              Say what the robot should do, with which objects, and what counts as done.
            </p>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="dc-example">Example of the task</Label>
            <div
              role="tablist"
              aria-label="Example type"
              className="flex w-fit rounded-lg border border-white/10 bg-white/[0.02] p-0.5"
            >
              {(
                [
                  ["video", "Video link"],
                  ["dataset", "Hugging Face dataset"],
                ] as const
              ).map(([kind, label]) => (
                <button
                  key={kind}
                  type="button"
                  role="tab"
                  aria-selected={exampleKind === kind}
                  onClick={() => edit(setExampleKind)(kind)}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs transition-colors",
                    exampleKind === kind
                      ? "bg-[#eff483]/10 text-[#eff483]"
                      : "text-white/55 hover:text-white/85"
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <Input
              id="dc-example"
              value={example}
              onChange={(e) => edit(setExample)(e.target.value)}
              placeholder={exampleKind === "video" ? "https://" : "your-org/your-dataset"}
              autoComplete="off"
              spellCheck={false}
            />
            <p className="text-xs text-white/40">
              {exampleKind === "video"
                ? "A YouTube, Google Drive or Dropbox link that anyone with the link can open."
                : "A dataset of sample episodes, as owner/name or its Hugging Face link."}
            </p>
          </div>

          <p className="rounded-lg border border-white/10 bg-white/[0.02] p-3 text-xs leading-relaxed text-white/55">
            You pay for the hours you request now. If Almond delivers fewer usable hours, we refund
            the difference.
          </p>

          {error && (
            <p role="alert" className="text-xs text-red-300">
              {error}
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-white/10 px-5 py-3">
          <Button type="button" variant="ghost" size="sm" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button type="submit" size="sm" disabled={submitting}>
            {submitting && <Loader2 className="animate-spin" />}
            Continue to payment
          </Button>
        </div>
      </form>
    </div>,
    document.body
  )
}
