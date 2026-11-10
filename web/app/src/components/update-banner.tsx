import { Download, ExternalLink, Loader2 } from "lucide-react"
import type { UpdatePhase, UpdateStatus } from "@/lib/supervisor"
import { Button, buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { PHASE_LABEL, shortCommit } from "@/lib/update-view"

const REPO_URL = "https://github.com/almond-bot/axol"
const RELEASES_URL = `${REPO_URL}/releases`

/** Release version with the conventional "v" prefix, e.g. "v0.1.2". */
function tag(version: string | null): string {
  return version ? `v${version}` : "unknown"
}

/** What the banner says and links to, for the installed channel. */
function describe(update: UpdateStatus): {
  title: string
  jump: string
  link: string
  linkLabel: string
} {
  if (update.channel === "main") {
    const { commit, remoteCommit } = update
    return {
      title: "New changes are available on main",
      jump: `main ${shortCommit(commit)} \u2192 ${shortCommit(remoteCommit)}`,
      link:
        commit && remoteCommit
          ? `${REPO_URL}/compare/${commit}...${remoteCommit}`
          : `${REPO_URL}/commits/main`,
      linkLabel: "View changes",
    }
  }
  return {
    title: "A new release is available",
    jump: `${tag(update.version)} \u2192 ${tag(update.remoteVersion)}`,
    link: RELEASES_URL,
    linkLabel: "Release notes",
  }
}

/**
 * Banner shown above the connections bar when newer code exists on the
 * installed channel: a release with a higher version, or new commits on main. The Update button is disabled while an
 * operation is running (a connected robot is fine), since applying it restarts
 * the server.
 */
export function UpdateBanner({
  update,
  updating,
  phase,
  blocked,
  busyReason,
  onUpdate,
}: {
  update: UpdateStatus
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
  const { title, jump, link, linkLabel } = describe(update)
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium text-amber-200">{title}</span>
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
