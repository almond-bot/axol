import { useState } from "react"
import { Link } from "react-router-dom"
import { ArrowRight, Database, X } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { useDataCollectionPrice } from "@/lib/data-collection-api"
import { formatMoney, type DataCollectionOfferOperation } from "@/lib/data-collection"
import { cn } from "@/lib/utils"

const HIDDEN_KEY = "axolDataCollectionOfferHidden"

function readHidden(): boolean {
  try {
    return localStorage.getItem(HIDDEN_KEY) === "1"
  } catch {
    return false
  }
}

const LEAD: Record<DataCollectionOfferOperation, string> = {
  overview: "Want Almond to collect your training data?",
  teleop: "Need training data for a task?",
  "collect-data": "Need more episodes than you can record?",
}

/**
 * The Teleop Data Collection offer on Overview and the Teleoperation and
 * Collect data pages: a short card that leads to the Request data page, which stays in the
 * sidebar after the card is hidden. The hourly price comes from Stripe;
 * without a connection to almond.bot the card still shows, without a number.
 */
export function DataCollectionOffer({ operation }: { operation: DataCollectionOfferOperation }) {
  const [hidden, setHidden] = useState(readHidden)
  const price = useDataCollectionPrice(!hidden)

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
      <Link
        to={`/control/request-data?from=${operation}`}
        className={cn(buttonVariants({ size: "sm" }), "shrink-0 self-start sm:self-auto")}
      >
        Request data
        <ArrowRight />
      </Link>
      <button
        type="button"
        onClick={hide}
        aria-label="Hide this offer"
        title="Hide this offer. Request data stays in the sidebar."
        className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-md text-white/30 transition-colors hover:bg-white/[0.06] hover:text-white/70 sm:static sm:self-start"
      >
        <X className="size-3.5" />
      </button>
    </div>
  )
}
