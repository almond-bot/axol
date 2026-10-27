import { useState } from "react"
import { Link } from "react-router-dom"
import { ArrowRight, Database, X } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { useDataCollectionPrice } from "@/lib/data-collection-api"
import {
  REQUEST_PAGE,
  formatMoney,
  priceKey,
  type DataCollectionOfferOperation,
  type DataCollectionService,
} from "@/lib/data-collection"
import { cn } from "@/lib/utils"

const HIDDEN_KEY = "axolDataCollectionOfferHidden"

function readHidden(): boolean {
  try {
    return localStorage.getItem(HIDDEN_KEY) === "1"
  } catch {
    return false
  }
}

/** What each page's card leads with, and the service it opens. A one-off
 *  task comes first everywhere but Collect data. */
const OFFER: Record<
  DataCollectionOfferOperation,
  { lead: string; service: DataCollectionService }
> = {
  overview: { lead: "Want Almond to do a task on your Axol?", service: "task" },
  teleop: { lead: "Want an Almond operator to drive?", service: "task" },
  "collect-data": { lead: "Need more episodes than you can record?", service: "dataset" },
}

const BODY: Record<DataCollectionService, string> = {
  task: "An Almond operator controls your Axol remotely and finishes a task for you.",
  dataset: "Almond records your task with Axol and delivers a LeRobot dataset, ready for training.",
}

const BILLING: Record<DataCollectionService, string> = {
  task: "per hour, billed by the minute.",
  dataset: "per hour of usable data.",
}

/**
 * The Teleop Data Collection offer on Overview and the Teleoperation and
 * Collect data pages: a short card that leads to Request teleoperator or
 * Request data, which stay in the sidebar after the card is hidden. Each
 * service's hourly price comes from Stripe;
 * without a connection to almond.bot the card still shows, without a number.
 */
export function DataCollectionOffer({ operation }: { operation: DataCollectionOfferOperation }) {
  const [hidden, setHidden] = useState(readHidden)
  const { lead, service } = OFFER[operation]
  // A dataset shows its lowest price, collected on Almond's Axol, as "from".
  const price = useDataCollectionPrice(priceKey(service, null), !hidden)

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
    <div className="relative flex flex-col gap-3 rounded-xl border border-[#eff483]/20 bg-gradient-to-br from-[#eff483]/[0.07] to-transparent p-4 sm:flex-row sm:items-center">
      <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-[#eff483]/10">
        <Database className="size-4 text-[#eff483]" />
      </div>
      <div className="min-w-0 flex-1 pr-6 sm:pr-0">
        <p className="text-sm font-medium text-white">{lead}</p>
        <p className="mt-0.5 text-xs leading-relaxed text-white/55">
          {BODY[service]}{" "}
          {price ? (
            <span className="text-white/80">
              {service === "dataset" && "From "}
              {formatMoney(price.unitAmount, price.currency)} {BILLING[service]}
            </span>
          ) : (
            `Priced ${BILLING[service]}`
          )}
        </p>
      </div>
      <Link
        to={`/control/${REQUEST_PAGE[service]}?from=${operation}`}
        className={cn(buttonVariants({ size: "sm" }), "shrink-0 self-start sm:self-auto")}
      >
        {service === "task" ? "Request teleoperator" : "Request data"}
        <ArrowRight />
      </Link>
      <button
        type="button"
        onClick={hide}
        aria-label="Hide this offer"
        title="Hide this offer. You can still request from the sidebar."
        className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-md text-white/30 transition-colors hover:bg-white/[0.06] hover:text-white/70 sm:static sm:self-start"
      >
        <X className="size-3.5" />
      </button>
    </div>
  )
}
