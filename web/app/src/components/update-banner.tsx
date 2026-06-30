import { Download, Loader2 } from "lucide-react"
import type { UpdateStatus } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"

/** First 7 chars of a git commit, the conventional short form. */
function short(commit: string | null): string {
  return commit ? commit.slice(0, 7) : "unknown"
}

/**
 * Banner shown above the connections bar when the tracked ref has moved past
 * the installed commit. The Update button is disabled unless the server is idle
 * (no op running, robot disconnected), since applying it restarts the server.
 */
export function UpdateBanner({
  update,
  updating,
  busyReason,
  onUpdate,
}: {
  update: UpdateStatus
  updating: boolean
  /** Why a restart is currently unsafe, e.g. "Disconnect Axol" (no period). */
  busyReason?: string
  onUpdate: () => void
}) {
  const blocked = !update.idle
  const hint = busyReason ?? "The server is busy"
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium text-amber-200">A new version is available</span>
        <span className="font-mono text-amber-200/60">
          {short(update.commit)} &rarr; {short(update.remoteCommit)}
        </span>
        {blocked && !updating && <span className="text-amber-200/60">{hint} to update.</span>}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onUpdate}
        disabled={blocked || updating}
        title={blocked ? `${hint} first` : undefined}
      >
        {updating ? <Loader2 className="animate-spin" /> : <Download />}
        {updating ? "Updating…" : "Update"}
      </Button>
    </div>
  )
}
