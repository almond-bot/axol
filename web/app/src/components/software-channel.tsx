import { GitBranch, Loader2, Tag } from "lucide-react"
import { useState } from "react"
import type { UpdateChannel, UpdatePhase, UpdateStatus } from "@/lib/supervisor"
import { PHASE_LABEL } from "@/lib/update-view"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const CHANNEL_LABEL: Record<UpdateChannel, string> = {
  release: "Release",
  main: "Main",
}

/** The confirmation for switching to `target`, given what the host reports. */
function confirmation(
  target: UpdateChannel,
  update: UpdateStatus
): { title: string; body: string; button: string } {
  if (target === "main") {
    return {
      title: "Switch to main",
      body:
        "Install the newest code from the main branch. It has changes that are not in a " +
        "release yet and has had less testing. The Host Computer installs it and restarts, " +
        "which can take several minutes.",
      button: "Switch to main",
    }
  }
  const release = update.remoteVersion ? ` (v${update.remoteVersion})` : ""
  return {
    title: "Switch to the latest release",
    body:
      `Install the latest release${release}. The Host Computer installs it and restarts, ` +
      "which can take several minutes.",
    button: "Switch to release",
  }
}

/**
 * Picks where the robot software comes from: the latest release or the main
 * branch. Switching reinstalls the software on the Host Computer and restarts
 * it, so it asks first. Only shown for hosts that can update themselves.
 */
export function SoftwareChannel({
  update,
  updating,
  phase,
  opRunning,
  hostInstalling,
  onSwitch,
}: {
  update: UpdateStatus
  /** An update or channel switch is applying. */
  updating: boolean
  /** Current update step, for the label while switching. */
  phase: UpdatePhase | null
  /** An operation or session is running, so the host can't restart. */
  opRunning: boolean
  /** The host is installing packages, so the host can't restart. */
  hostInstalling: boolean
  onSwitch: (channel: UpdateChannel) => void
}) {
  const [target, setTarget] = useState<UpdateChannel | null>(null)
  const current: UpdateChannel = update.channel ?? "release"
  const releaseUnknown = target === "release" && !update.remoteVersion
  const blocker = opRunning
    ? "An operation is running. Stop it first."
    : hostInstalling
      ? "The robot is installing updates. Wait for it to finish."
      : releaseUnknown
        ? "The latest release is not known yet. Check the robot's internet connection."
        : null
  const copy = target ? confirmation(target, update) : null

  return (
    <>
      <div className="flex items-center gap-2">
        {updating && (
          <span className="flex items-center gap-1 text-xs text-white/45">
            <Loader2 className="size-3 animate-spin" />
            {(phase && PHASE_LABEL[phase]) || "Updating…"}
          </span>
        )}
        <div
          role="radiogroup"
          aria-label="Software channel"
          className="flex rounded-lg border border-white/10 bg-white/[0.03] p-0.5"
        >
          {(["release", "main"] as const).map((channel) => {
            const selected = channel === current
            const Icon = channel === "release" ? Tag : GitBranch
            return (
              <button
                key={channel}
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={updating}
                onClick={() => {
                  if (!selected) setTarget(channel)
                }}
                className={cn(
                  "flex items-center gap-1 rounded-md px-2 py-0.5 text-xs transition-colors disabled:opacity-50",
                  selected
                    ? "bg-[#eff483]/15 text-[#eff483]"
                    : "text-white/55 hover:bg-white/[0.06] hover:text-white/80"
                )}
              >
                <Icon className="size-3" />
                {CHANNEL_LABEL[channel]}
              </button>
            )
          })}
        </div>
      </div>

      {target && copy && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setTarget(null)}
        >
          <Card
            className="w-full max-w-sm gap-4 bg-[#1a1a1a] p-5 text-left"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col gap-1">
              <h3 className="font-heading text-base font-semibold">{copy.title}</h3>
              <p className="text-sm leading-relaxed whitespace-normal text-white/45">{copy.body}</p>
            </div>
            {blocker && <p className="text-xs whitespace-normal text-amber-200/70">{blocker}</p>}
            <div className="flex items-center justify-end gap-2 pt-1">
              <Button variant="ghost" size="sm" onClick={() => setTarget(null)}>
                Cancel
              </Button>
              <Button
                size="sm"
                disabled={blocker !== null || updating}
                onClick={() => {
                  onSwitch(target)
                  setTarget(null)
                }}
              >
                {target === "main" ? <GitBranch /> : <Tag />}
                {copy.button}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
