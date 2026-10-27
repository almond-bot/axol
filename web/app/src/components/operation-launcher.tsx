import { useEffect, useState } from "react"
import { ChevronRight, Loader2 } from "lucide-react"
import { OPERATION_GROUPS, type CatalogEntry, type OperationStatus } from "@/lib/operation-catalog"
import type { StartupView } from "@/lib/operation-progress"
import type { OperationId } from "@/lib/supervisor"
import { cn } from "@/lib/utils"

export interface LauncherItem {
  entry: CatalogEntry
  status: OperationStatus
}

const STATUS_CLASS: Record<OperationStatus["tone"], string> = {
  running: "text-emerald-300",
  ready: "text-white/45",
  attention: "text-amber-300/90",
  off: "text-white/35",
}

/** The operation's icon on a tinted square, sized for a tile or a header. */
export function OperationIcon({
  entry,
  size = "md",
  running = false,
}: {
  entry: CatalogEntry
  size?: "sm" | "md" | "lg"
  running?: boolean
}) {
  const Icon = entry.icon
  return (
    <span
      className={cn(
        "relative flex shrink-0 items-center justify-center rounded-xl border",
        running
          ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
          : "border-[#eff483]/15 bg-[#eff483]/[0.07] text-[#eff483]",
        size === "sm" && "size-9 rounded-lg",
        size === "md" && "size-11",
        size === "lg" && "size-14 rounded-2xl"
      )}
    >
      <Icon className={size === "lg" ? "size-6" : size === "md" ? "size-5" : "size-4"} />
      {running && (
        <span className="absolute -top-0.5 -right-0.5 size-2.5 animate-pulse rounded-full border-2 border-[#121212] bg-emerald-400" />
      )}
    </span>
  )
}

function StatusLine({ status }: { status: OperationStatus }) {
  return (
    <span className={cn("flex items-center gap-1.5 text-xs", STATUS_CLASS[status.tone])}>
      {status.tone === "running" && (
        <span className="size-1.5 animate-pulse rounded-full bg-emerald-400" />
      )}
      {status.tone === "attention" && <span className="size-1.5 rounded-full bg-amber-400" />}
      <span className="truncate">{status.label}</span>
    </span>
  )
}

/** One operation as a tile that opens its screen. */
export function OperationTile({
  item,
  onOpen,
}: {
  item: LauncherItem
  onOpen: (id: OperationId) => void
}) {
  const { entry, status } = item
  return (
    <button
      type="button"
      onClick={() => onOpen(entry.id)}
      className={cn(
        // A row on phones, a card from `sm` up.
        "group flex items-start gap-3 rounded-2xl border p-4 text-left transition-all sm:min-h-36 sm:flex-col",
        status.tone === "running"
          ? "border-emerald-400/30 bg-emerald-400/[0.04] hover:border-emerald-400/50"
          : "border-white/10 bg-white/[0.02] hover:-translate-y-0.5 hover:border-white/25 hover:bg-white/[0.04]"
      )}
    >
      <span className="flex items-start justify-between gap-2 sm:w-full">
        <OperationIcon entry={entry} running={status.tone === "running"} />
        <ChevronRight className="hidden size-4 text-white/20 transition-colors group-hover:text-white/60 sm:block" />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-1 sm:w-full">
        <span className="text-[0.95rem] font-medium text-white">{entry.label}</span>
        <span className="line-clamp-2 text-xs leading-relaxed text-white/50">{entry.summary}</span>
        <span className="mt-1.5 sm:mt-auto sm:pt-2">
          <StatusLine status={status} />
        </span>
      </span>
    </button>
  )
}

/** Every operation, as tiles grouped by what they are for. */
export function OperationLauncher({
  items,
  onOpen,
}: {
  items: LauncherItem[]
  onOpen: (id: OperationId) => void
}) {
  return (
    <div className="flex flex-col gap-8">
      {OPERATION_GROUPS.map((group) => {
        const inGroup = items.filter((i) => i.entry.group === group.key)
        if (inGroup.length === 0) return null
        return (
          <section key={group.key} className="flex flex-col gap-3">
            <h2 className="font-heading text-sm font-semibold text-white/60">{group.label}</h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {inGroup.map((item) => (
                <OperationTile key={item.entry.id} item={item} onOpen={onOpen} />
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}

/** Time since `startedAt` (seconds since the epoch), ticking each second:
 *  "4:07", or "1:02:09" past an hour. */
export function ElapsedTime({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [])
  const total = Math.max(0, Math.floor(now / 1000 - startedAt))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, "0")
  return (
    <span className="font-mono tabular-nums">
      {h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`}
    </span>
  )
}

/** What a starting operation is doing, how far along it is, and a bar. */
export function StartupBar({ view }: { view: StartupView }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <div className="flex min-w-0 items-center justify-between gap-3 text-sm">
        <span className="flex min-w-0 items-center gap-2 text-white/80">
          <Loader2 className="size-3.5 shrink-0 animate-spin text-[#eff483]" />
          <span className="truncate">{view.label}</span>
        </span>
        <span className="shrink-0 font-mono text-white/60 tabular-nums">{view.percent}%</span>
      </div>
      <div
        role="progressbar"
        aria-label={view.label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={view.percent}
        className="h-1.5 w-full overflow-hidden rounded-full bg-white/10"
      >
        <div
          className="h-full rounded-full bg-[#eff483] transition-[width] duration-500 ease-out"
          style={{ width: `${view.percent}%` }}
        />
      </div>
    </div>
  )
}
