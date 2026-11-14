import { Download, ExternalLink, Loader2 } from "lucide-react"
import type { UpdateChannel, UpdatePhase, UpdateStatus } from "@/lib/supervisor"
import { Button, buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { PHASE_LABEL, describeChange } from "@/lib/update-view"

/** The banner's heading, for an update within the installed channel or a switch to `to`. */
function title(update: UpdateStatus, to: UpdateChannel): string {
  const from = update.channel ?? "release"
  if (to !== from) return to === "main" ? "Switching to main" : "Switching to the latest release"
  return from === "main" ? "New changes are available on main" : "A new release is available"
}

/**
 * Banner shown above the page while newer code is available or being
 * installed: a release with a higher version, new commits on main, or a switch
 * between the two channels. It shows what the host moves from and to, a link
 * to the changes and the update's progress. The Update button is disabled
 * while an operation is running (a connected robot is fine), since applying it
 * restarts the server.
 */
export function UpdateBanner({
  update,
  switchTo = null,
  updating,
  phase,
  blocked,
  busyReason,
  onUpdate,
}: {
  update: UpdateStatus
  /** The channel being switched to, while a channel switch applies. */
  switchTo?: UpdateChannel | null
  updating: boolean
  /** Current update step, for the button label while updating. */
  phase: UpdatePhase | null
  /** Whether a restart is currently unsafe (e.g. an operation is running). */
  blocked: boolean
  /** Why a restart is currently unsafe, e.g. "Stop the running operation" (no period). */
  busyReason?: string
  onUpdate: () => void
}) {
  const hint = busyReason ?? "The Host Computer is busy"
  const updatingLabel = (phase && PHASE_LABEL[phase]) || "Updating…"
  const to = switchTo ?? update.channel ?? "release"
  const { jump, link, linkLabel } = describeChange(update, to)
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium text-amber-200">{title(update, to)}</span>
        <span className="font-mono text-amber-200/60">{jump}</span>
        {blocked && !updating && <span className="text-amber-200/60">{hint} to update.</span>}
      </div>
      <div className="flex items-center gap-2">
        <a
          href={link}
          target="_blank"
          rel="noreferrer"
          className={cn(
            buttonVariants({ variant: "ghost", size: "sm" }),
            "text-amber-200/80 hover:bg-amber-400/10 hover:text-amber-200"
          )}
        >
          {linkLabel}
          <ExternalLink />
        </a>
        <Button
          variant="outline"
          size="sm"
          onClick={onUpdate}
          disabled={blocked || updating}
          title={blocked ? `${hint} first` : undefined}
        >
          {updating ? <Loader2 className="animate-spin" /> : <Download />}
          {updating ? updatingLabel : "Update"}
        </Button>
      </div>
    </div>
  )
}
