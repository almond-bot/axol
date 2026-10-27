import { useState } from "react"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { PageSection } from "@/components/app-shell"
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
import { startDataCollectionCheckout, useDataCollectionPrice } from "@/lib/data-collection-api"
import { cn } from "@/lib/utils"

/** The current page without a `data_collection` result, for Stripe to
 *  send the customer back to. */
function returnUrl(): string {
  const url = new URL(window.location.href)
  url.searchParams.delete("data_collection")
  return url.toString()
}

const STEPS = [
  "Describe the task. An example of it helps, but is optional.",
  "Choose the hours of usable data you need, and pay on Stripe's checkout page.",
  "Almond emails you to plan the collection.",
  "Almond records the episodes and delivers them as a LeRobot dataset.",
]

/**
 * The Request data page: what Teleop Data Collection is, and the form that
 * hands off to Stripe's checkout page. `source` is the page the customer came
 * from (an operation page's card, or this page), kept with the request.
 */
export function RequestDataPage({ source }: { source: DataCollectionOperation }) {
  const price = useDataCollectionPrice()
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <RequestForm source={source} price={price} />
      <PageSection title="How it works">
        <ol className="flex flex-col gap-2.5">
          {STEPS.map((step, i) => (
            <li key={step} className="flex gap-3 text-sm text-white/65">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-[#eff483]/10 font-mono text-[0.7rem] text-[#eff483]">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
        <p className="text-sm text-white/65">
          {price ? (
            <span className="text-white/85">
              {formatMoney(price.unitAmount, price.currency)} per hour of usable data.
            </span>
          ) : (
            "Priced per hour of usable data."
          )}{" "}
          If Almond delivers fewer usable hours than you paid for, we refund the difference.
        </p>
      </PageSection>
    </div>
  )
}

function RequestForm({
  source,
  price,
}: {
  source: DataCollectionOperation
  price: DataCollectionPrice | null
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
      operation: source,
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

  return (
    <form
      onSubmit={submit}
      aria-label="Request Teleop Data Collection"
      className="flex flex-col rounded-xl border border-white/10 bg-white/[0.02]"
    >
      <div className="flex flex-col gap-5 p-5">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="dc-hours">Hours of usable data</Label>
          <div className="flex flex-wrap items-center gap-3">
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
          <Label htmlFor="dc-example">
            Example of the task
            <span className="ml-1.5 font-normal text-white/40">Optional</span>
          </Label>
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

        {error && (
          <p role="alert" className="text-xs text-red-300">
            {error}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-5 py-3">
        <p className="text-xs text-white/40">You pay for the hours you request now.</p>
        <Button type="submit" size="sm" disabled={submitting}>
          {submitting && <Loader2 className="animate-spin" />}
          Continue to payment
        </Button>
      </div>
    </form>
  )
}
