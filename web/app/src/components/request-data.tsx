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
  priceKey,
  validateDataCollectionRequest,
  type DataCollectionOperation,
  type DataCollectionPrice,
  type CollectionMethod,
  type DataCollectionService,
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

interface ServiceCopy {
  hoursLabel: string
  defaultHours: string
  placeholder: string
  steps: string[]
  refund: string
  footer: string
}

const SERVICES: Record<DataCollectionService, ServiceCopy> = {
  task: {
    hoursLabel: "Hours of operator time",
    defaultHours: "1",
    placeholder: "For example: unload the parts from the 3D printer and drop them in the bin.",
    steps: [
      "Describe the task. An example of it helps, but is optional.",
      "Choose how many hours you expect it to take, and pay on Stripe's checkout page.",
      "Almond emails you to schedule the session.",
      "An Almond operator controls your Axol until the task is done.",
    ],
    refund: "Billed by the minute. Almond refunds any time you don't use.",
    footer: "You prepay the hours now. Unused minutes are refunded.",
  },
  dataset: {
    hoursLabel: "Hours of usable data",
    defaultHours: "10",
    placeholder:
      "For example: pick up each mug from the rack and place it upside down on the tray.",
    steps: [
      "Describe the task. An example of it helps, but is optional.",
      "Choose how Almond collects the data.",
      "Choose the hours of usable data you need, and pay on Stripe's checkout page.",
      "Almond emails you to plan the collection.",
      "Almond records the episodes and delivers them as a LeRobot dataset.",
    ],
    refund:
      "You pay per hour of usable data, however long recording takes. If Almond delivers fewer usable hours than you paid for, we refund the difference.",
    footer: "You pay for the hours you request now.",
  },
}

/** The ways Almond collects a training dataset, in the order they are offered. */
const METHODS: { key: CollectionMethod; label: string; hint: string }[] = [
  {
    key: "almond_axol",
    label: "Almond's Axol",
    hint: "Almond builds your setup and records the task with teleop on its own Axol.",
  },
  {
    key: "your_axol",
    label: "Your Axol",
    hint: "An Almond operator controls your Axol remotely and records the task in your space.",
  },
]

/**
 * A Teleop Data Collection request page: Request teleoperator (a one-off
 * task) or Request data (a training dataset), with the form that hands off to
 * Stripe's checkout page. `source` is the page the customer came from (an
 * operation page's card, or this page), kept with the request.
 */
export function RequestPage({
  service,
  source,
}: {
  service: DataCollectionService
  source: DataCollectionOperation
}) {
  const [method, setMethod] = useState<CollectionMethod>(METHODS[0].key)
  const price = useDataCollectionPrice(priceKey(service, method))
  const copy = SERVICES[service]
  const rate = price ? `${formatMoney(price.unitAmount, price.currency)} per hour.` : null
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <RequestForm
        service={service}
        method={method}
        onMethod={setMethod}
        source={source}
        price={price}
      />
      <PageSection title="How it works">
        <ol className="flex flex-col gap-2.5">
          {copy.steps.map((step, i) => (
            <li key={step} className="flex gap-3 text-sm text-white/65">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-[#eff483]/10 font-mono text-[0.7rem] text-[#eff483]">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
        <p className="text-sm text-white/65">
          {rate && <span className="text-white/85">{rate}</span>} {copy.refund}
        </p>
      </PageSection>
    </div>
  )
}

function RequestForm({
  service,
  method,
  onMethod,
  source,
  price,
}: {
  service: DataCollectionService
  method: CollectionMethod
  onMethod: (method: CollectionMethod) => void
  source: DataCollectionOperation
  price: DataCollectionPrice | null
}) {
  const copy = SERVICES[service]
  const [hours, setHours] = useState(copy.defaultHours)
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
      service,
      method,
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
      aria-label={service === "task" ? "Request a teleoperator" : "Request data"}
      className="flex flex-col rounded-xl border border-white/10 bg-white/[0.02]"
    >
      <div className="flex flex-col gap-5 p-5">
        {service === "dataset" && (
          <div className="flex flex-col gap-1.5">
            <Label id="dc-method">How Almond collects it</Label>
            <div
              role="radiogroup"
              aria-labelledby="dc-method"
              className="flex w-fit flex-wrap rounded-lg border border-white/10 bg-white/[0.02] p-0.5"
            >
              {METHODS.map((m) => (
                <button
                  key={m.key}
                  type="button"
                  role="radio"
                  aria-checked={method === m.key}
                  onClick={() => edit(onMethod)(m.key)}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs transition-colors",
                    method === m.key
                      ? "bg-[#eff483]/10 text-[#eff483]"
                      : "text-white/55 hover:text-white/85"
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <p className="text-xs text-white/40">{METHODS.find((m) => m.key === method)?.hint}</p>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="dc-hours">{copy.hoursLabel}</Label>
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
            placeholder={copy.placeholder}
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
        <p className="text-xs text-white/40">{copy.footer}</p>
        <Button type="submit" size="sm" disabled={submitting}>
          {submitting && <Loader2 className="animate-spin" />}
          Continue to payment
        </Button>
      </div>
    </form>
  )
}
