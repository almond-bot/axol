import { Link } from "react-router-dom"
import { ArrowRight, CircleAlert, Database, Loader2, Plug, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ElapsedTime, OperationIcon } from "@/components/operation-launcher"
import type { HardwareNotice } from "@/lib/hardware-status"
import { STATUS_DOT_CLASS, type StatusDot } from "@/lib/jelly-view"
import type { CatalogEntry } from "@/lib/operation-catalog"
import type { DatasetInfo, OperationId } from "@/lib/supervisor"
import { cn } from "@/lib/utils"

/** The first thing on Home until the Host Computer is connected. */
export function ConnectHero({ failed, onConnect }: { failed: boolean; onConnect: () => void }) {
  return (
    <section className="flex flex-col gap-4 rounded-2xl border border-[#eff483]/20 bg-gradient-to-br from-[#eff483]/[0.07] via-transparent to-transparent p-6 sm:flex-row sm:items-center">
      <span className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-[#eff483]/10 text-[#eff483]">
        <Plug className="size-6" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="font-heading text-lg font-semibold">
          {failed ? "Can't reach the Host Computer" : "Connect to your robot"}
        </h2>
        <p className="mt-1 max-w-prose text-sm text-white/55">
          {failed
            ? "Check that it is on and on this network, then connect again."
            : "Operations run on the Host Computer. Connect to it to start one."}
        </p>
      </div>
      <Button size="lg" onClick={onConnect}>
        Connect
      </Button>
    </section>
  )
}

/** The operation in progress, with how long it has run, at the top of Home. */
export function RunningCard({
  entry,
  startedAt,
  stopping,
  busy,
  onOpen,
  onStop,
}: {
  entry: CatalogEntry
  startedAt: number | null
  stopping: boolean
  busy: boolean
  onOpen: () => void
  onStop: () => void
}) {
  return (
    <section className="flex flex-wrap items-center gap-4 rounded-2xl border border-emerald-400/25 bg-gradient-to-br from-emerald-400/[0.07] to-transparent p-5">
      <OperationIcon entry={entry} running />
      <div className="min-w-0 flex-1">
        <p className="truncate font-heading text-base font-semibold text-white">{entry.label}</p>
        <p className="text-sm text-emerald-200/80">
          {stopping ? (
            "Stopping…"
          ) : (
            <>
              Running
              {startedAt != null && (
                <>
                  {" · "}
                  <ElapsedTime startedAt={startedAt} />
                </>
              )}
            </>
          )}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="outline" onClick={onOpen}>
          Open
        </Button>
        <Button variant="destructive" onClick={onStop} disabled={busy || stopping}>
          {stopping ? <Loader2 className="animate-spin" /> : <Square />}
          {stopping ? "Stopping…" : "Stop"}
        </Button>
      </div>
    </section>
  )
}

export interface AttentionItem {
  key: string
  text: string
  action: string
  onAction: () => void
}

/** What to sort out before running: each item with the button that fixes it. */
export function AttentionList({ items }: { items: AttentionItem[] }) {
  if (items.length === 0) return null
  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-heading text-sm font-semibold text-white/60">Set up</h2>
      <ul className="flex flex-col divide-y divide-white/[0.06] overflow-hidden rounded-xl border border-amber-400/20 bg-amber-400/[0.03]">
        {items.map((item) => (
          <li key={item.key} className="flex flex-wrap items-center gap-3 px-4 py-3">
            <CircleAlert className="size-4 shrink-0 text-amber-300/80" />
            <span className="min-w-0 flex-1 text-sm text-white/75">{item.text}</span>
            <Button variant="outline" size="sm" onClick={item.onAction}>
              {item.action}
            </Button>
          </li>
        ))}
      </ul>
    </section>
  )
}

/** A few operations to start from Home: recent ones first. */
export function QuickStart({
  entries,
  onOpen,
}: {
  entries: CatalogEntry[]
  onOpen: (id: OperationId) => void
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-heading text-sm font-semibold text-white/60">Quick start</h2>
        <Link
          to="/control/operations"
          className="flex items-center gap-1 text-xs text-white/45 transition-colors hover:text-white/80"
        >
          All operations
          <ArrowRight className="size-3.5" />
        </Link>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {entries.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => onOpen(entry.id)}
            className="group flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-3.5 text-left transition-all hover:-translate-y-0.5 hover:border-white/25 hover:bg-white/[0.04]"
          >
            <OperationIcon entry={entry} size="sm" />
            <span className="min-w-0 flex-1 truncate text-sm font-medium text-white/90">
              {entry.label}
            </span>
            <ArrowRight className="size-4 shrink-0 text-white/20 transition-colors group-hover:text-white/60" />
          </button>
        ))}
      </div>
    </section>
  )
}

/** The newest datasets on the Host Computer, each opening the Datasets page. */
export function RecentDatasets({ datasets }: { datasets: DatasetInfo[] }) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-heading text-sm font-semibold text-white/60">Recent datasets</h2>
        <Link
          to="/control/datasets"
          className="flex items-center gap-1 text-xs text-white/45 transition-colors hover:text-white/80"
        >
          All datasets
          <ArrowRight className="size-3.5" />
        </Link>
      </div>
      {datasets.length === 0 ? (
        <p className="rounded-xl border border-dashed border-white/10 p-4 text-sm text-white/45">
          No datasets yet. Collect data records one.
        </p>
      ) : (
        <ul className="flex flex-col divide-y divide-white/[0.06] overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
          {datasets.map((d) => (
            <li key={d.repoId}>
              <Link
                to="/control/datasets"
                className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-white/[0.03]"
              >
                <Database className="size-4 shrink-0 text-white/35" />
                <span className="min-w-0 flex-1 truncate font-mono text-sm text-white/80">
                  {d.repoId}
                </span>
                {d.episodes != null && (
                  <span className="shrink-0 text-xs text-white/45">
                    {d.episodes} episode{d.episodes === 1 ? "" : "s"}
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export interface DeviceSummary {
  key: string
  name: string
  dot: StatusDot
  status: string
  /** The host's own words, for the tooltip. */
  detail?: string | null
  /** Offer Connect: the device is plugged in but not connected. */
  connectable?: boolean
  connectBusy?: boolean
}

/** The devices at a glance on Home: one line each, a Connect button where
 *  one is needed, and the whole row opening the Devices page. */
export function DevicesRow({
  devices,
  onConnect,
}: {
  devices: DeviceSummary[]
  onConnect: (key: string) => void
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-heading text-sm font-semibold text-white/60">Devices</h2>
        <Link
          to="/control/devices"
          className="flex items-center gap-1 text-xs text-white/45 transition-colors hover:text-white/80"
        >
          Manage devices
          <ArrowRight className="size-3.5" />
        </Link>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {devices.map((d) => (
          <div
            key={d.key}
            title={d.detail ?? undefined}
            className="flex min-w-0 items-center gap-3 rounded-xl border border-white/10 bg-white/[0.02] px-3.5 py-3"
          >
            <span className={cn("size-2 shrink-0 rounded-full", STATUS_DOT_CLASS[d.dot])} />
            <Link to="/control/devices" className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium text-white/90">{d.name}</span>
              <span
                className={cn(
                  "block truncate text-xs",
                  d.dot === "err" ? "text-red-300/90" : "text-white/45"
                )}
              >
                {d.status}
              </span>
            </Link>
            {d.connectable && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onConnect(d.key)}
                disabled={d.connectBusy}
                className="shrink-0"
              >
                {d.connectBusy ? <Loader2 className="animate-spin" /> : null}
                Connect
              </Button>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

/** A hardware problem in plain words, with "Check again" only where the
 *  panel can't check on its own without disturbing a connected robot. */
export function HardwareNoticeCard({
  notice,
  onCheckAgain,
  checking,
}: {
  notice: HardwareNotice
  onCheckAgain?: () => void
  checking: boolean
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-xl border px-4 py-3",
        notice.tone === "error"
          ? "border-red-400/25 bg-red-400/[0.04]"
          : "border-amber-400/20 bg-amber-400/[0.03]"
      )}
    >
      {checking ? (
        <Loader2 className="size-4 shrink-0 animate-spin text-white/45" />
      ) : (
        <CircleAlert
          className={cn(
            "size-4 shrink-0",
            notice.tone === "error" ? "text-red-300" : "text-amber-300/80"
          )}
        />
      )}
      <p className="min-w-0 flex-1 text-sm text-white/75">{notice.text}</p>
      {onCheckAgain && (
        <Button variant="outline" size="sm" onClick={onCheckAgain} disabled={checking}>
          Check again
        </Button>
      )}
    </div>
  )
}
