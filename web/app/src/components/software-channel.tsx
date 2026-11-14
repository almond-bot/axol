import { ExternalLink, GitBranch, Tag } from "lucide-react"
import { useState } from "react"
import type { UpdateChannel, UpdateStatus } from "@/lib/supervisor"
import { describeChange } from "@/lib/update-view"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/** The confirmation for switching to `target`. */
function confirmation(target: UpdateChannel): { title: string; body: string; button: string } {
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
  return {
    title: "Switch to the latest release",
    body:
      "Install the latest release. The Host Computer installs it and restarts, " +
      "which can take several minutes.",
    button: "Switch to release",
  }
}

/**
 * Switches where the robot software comes from: the latest release or the
 * main branch. Opens a confirmation that shows what the host moves from and
 * to, like the update banner. Once confirmed, the update banner shows the
 * switch's progress. Only shown for hosts that can update themselves.
 */
export function SoftwareChannel({
  update,
  updating,
  opRunning,
  hostInstalling,
  onSwitch,
}: {
  update: UpdateStatus
  /** An update or channel switch is applying. */
  updating: boolean
  /** An operation or session is running, so the host can't restart. */
  opRunning: boolean
  /** The host is installing packages, so the host can't restart. */
  hostInstalling: boolean
  onSwitch: (channel: UpdateChannel) => void
}) {
  const [open, setOpen] = useState(false)
  const current: UpdateChannel = update.channel ?? "release"
  const target: UpdateChannel = current === "main" ? "release" : "main"
  const releaseUnknown = target === "release" && !update.remoteVersion
  const blocker = opRunning
    ? "An operation is running. Stop it first."
    : hostInstalling
      ? "The robot is installing updates. Wait for it to finish."
      : releaseUnknown
        ? "The latest release is not known yet. Check the robot's internet connection."
        : null
  const copy = confirmation(target)
  const { jump, link, linkLabel } = describeChange(update, target)
  const Icon = target === "main" ? GitBranch : Tag

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        disabled={updating}
        title={updating ? "An update is installing." : undefined}
        onClick={() => setOpen(true)}
        className="h-7 px-2 text-xs text-white/60"
      >
        <Icon />
        {copy.button}
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setOpen(false)}
        >
          <Card
            className="w-full max-w-sm gap-4 bg-[#1a1a1a] p-5 text-left"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col gap-1">
              <h3 className="font-heading text-base font-semibold">{copy.title}</h3>
              <p className="font-mono text-xs text-white/70">{jump}</p>
              <p className="text-sm leading-relaxed whitespace-normal text-white/45">{copy.body}</p>
            </div>
            {blocker && <p className="text-xs whitespace-normal text-amber-200/70">{blocker}</p>}
            <div className="flex items-center gap-2 pt-1">
              <a
                href={link}
                target="_blank"
                rel="noreferrer"
                className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "text-white/60")}
              >
                {linkLabel}
                <ExternalLink />
              </a>
              <span className="ml-auto" />
              <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                disabled={blocker !== null || updating}
                onClick={() => {
                  onSwitch(target)
                  setOpen(false)
                }}
              >
                <Icon />
                {copy.button}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
