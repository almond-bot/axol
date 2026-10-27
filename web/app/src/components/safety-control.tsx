import { useEffect, useRef, useState } from "react"
import { OctagonX, Power } from "lucide-react"
import { Button } from "@/components/ui/button"
import { safetyView, type SafetyTone } from "@/lib/safety-view"
import { fetchSafetyStatus, safetyStart, safetyStop, type SafetyLinkStatus } from "@/lib/supervisor"
import { cn } from "@/lib/utils"

const POLL_MS = 1000

const PILL_TONE: Record<SafetyTone, string> = {
  on: "border-emerald-400/30 text-emerald-200",
  off: "border-amber-400/35 bg-amber-400/[0.08] text-amber-200",
  estop: "border-red-400/40 bg-red-400/[0.1] text-red-200",
  fault: "border-red-400/40 bg-red-400/[0.1] text-red-200",
  unknown: "border-white/15 bg-white/[0.03] text-white/60",
}

const DOT_TONE: Record<SafetyTone, string> = {
  on: "bg-emerald-400",
  off: "bg-amber-400",
  estop: "bg-red-400",
  fault: "bg-red-400",
  unknown: "bg-white/30",
}

function errorText(error: unknown): string {
  return String(error).replace(/^Error:\s*/, "")
}

/**
 * The software E-stop for Jelly builds with a safety board, at the top right of
 * every page. While Axol has power it is one red **Cut power** button that
 * stops at the first click. Otherwise it shows why power is off and opens a
 * panel to restore it. Renders nothing on a robot without a safety board.
 */
export function SafetyControl({ enabled }: { enabled: boolean }) {
  const [link, setLink] = useState<SafetyLinkStatus | null>(null)
  const [pending, setPending] = useState<"stop" | "start" | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!enabled) return
    let active = true
    const poll = () =>
      fetchSafetyStatus()
        .then((status) => {
          if (active) setLink(status)
        })
        .catch((e) => {
          // A host that predates the safety link has no such endpoint. Any
          // other failure keeps the last state until the next poll.
          if (active && String(e).includes("HTTP 404")) setLink(null)
        })
    void poll()
    const timer = setInterval(poll, POLL_MS)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [enabled])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("mousedown", onDown)
    window.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onDown)
      window.removeEventListener("keydown", onKey)
    }
  }, [open])

  const view = safetyView(enabled ? link : null)
  if (!view.visible) return null

  async function run(action: "stop" | "start") {
    setPending(action)
    setError(null)
    try {
      setLink(await (action === "stop" ? safetyStop() : safetyStart()))
      setOpen(false)
    } catch (e) {
      setError(errorText(e))
      setOpen(true)
    } finally {
      setPending(null)
    }
  }

  const stopLabel = pending === "stop" ? "Cutting power…" : "Cut power"

  return (
    <div ref={ref} className="relative shrink-0">
      {view.tone === "on" ? (
        <button
          type="button"
          onClick={() => void run("stop")}
          disabled={pending !== null}
          title="Cuts power to Axol, like the E-Stop."
          className="flex h-8 items-center gap-1.5 rounded-full bg-red-600 px-3 text-xs font-semibold text-white shadow-sm transition-colors outline-none hover:bg-red-500 focus-visible:ring-2 focus-visible:ring-red-300 disabled:opacity-60"
        >
          <OctagonX className="size-4" aria-hidden />
          {stopLabel}
        </button>
      ) : (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-haspopup="dialog"
          className={cn(
            "flex h-8 items-center gap-2 rounded-full border px-3 text-xs font-medium transition-colors outline-none hover:brightness-125 focus-visible:ring-2 focus-visible:ring-white/60",
            PILL_TONE[view.tone]
          )}
        >
          <span className={cn("size-2 shrink-0 rounded-full", DOT_TONE[view.tone])} />
          {pending === "start" ? "Restoring power…" : view.label}
        </button>
      )}

      {open && (
        <div
          role="dialog"
          aria-label="Power"
          className={cn(
            "z-40 flex flex-col gap-3 rounded-xl border border-white/15 bg-[#181818] p-4 text-sm shadow-2xl",
            // Below sm the control isn't at the screen edge, so the panel
            // spans the screen under the header instead of hanging off it.
            "fixed inset-x-4 top-[calc(3.75rem+env(safe-area-inset-top))]",
            "sm:absolute sm:inset-x-auto sm:top-full sm:right-0 sm:mt-2 sm:w-72"
          )}
        >
          <div className="flex flex-col gap-1">
            <h2 className="font-heading text-sm font-semibold text-white">{view.label}</h2>
            <p className="text-xs leading-relaxed text-white/65">{view.detail}</p>
            {view.notes.map((note) => (
              <p key={note} className="text-xs leading-relaxed text-white/45">
                {note}
              </p>
            ))}
          </div>
          {error && (
            <p role="alert" className="text-xs leading-relaxed text-red-300">
              {error}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {view.canStart && (
              <Button size="sm" onClick={() => void run("start")} disabled={pending !== null}>
                <Power aria-hidden />
                {pending === "start" ? "Restoring power…" : "Restore power"}
              </Button>
            )}
            {view.canStop && view.tone !== "on" && (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => void run("stop")}
                disabled={pending !== null}
              >
                <OctagonX aria-hidden />
                {stopLabel}
              </Button>
            )}
            {!view.canStart && view.startHint && (
              <span className="text-xs text-white/45">{view.startHint}</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
