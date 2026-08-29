import { useEffect, useMemo, useRef, useState } from "react"
import { Bluetooth, Check, Loader2, RadioTower, Square } from "lucide-react"
import {
  fetchSessions,
  fetchTrackerBindings,
  sendSessionInput,
  stopSession,
  useSessionLogs,
  type SessionInfo,
  type TrackerBackend,
  type TrackerBinding,
} from "@/lib/supervisor"
import { startDiagnosticsRun } from "@/lib/telemetry"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/toast"

function backendFor(source: string): TrackerBackend | null {
  if (source === "lighthouse") return "survive"
  if (source === "ultimate") return "ultimate"
  return null
}

/** Backend installation and guided left/right binding in Settings → Mantis. */
export function TrackerBindingPanel({ source }: { source: string }) {
  const toast = useToast()
  const backend = backendFor(source)
  const [bindings, setBindings] = useState<Partial<Record<TrackerBackend, TrackerBinding>>>({})
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [dismissed, setDismissed] = useState<{ id: string; lines: number } | null>(null)
  const { lines, status } = useSessionLogs(session?.id ?? null)
  const completionRef = useRef<string | null>(null)

  useEffect(() => {
    fetchTrackerBindings()
      .then(({ bindings: found }) => setBindings(found))
      .catch(() => {})
    fetchSessions()
      .then((sessions) => {
        const active = sessions.find(
          (candidate) =>
            (candidate.command === "tracker.identify" || candidate.command === "tracker.pair") &&
            (candidate.status === "starting" || candidate.status === "running")
        )
        if (active) setSession(active)
      })
      .catch(() => {})
  }, [])

  const prompt = useMemo(() => {
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i]
      if (!line.trim() || line.startsWith("[serve]")) continue
      return line.startsWith("[prompt] ") ? line.slice("[prompt] ".length).trim() : null
    }
    return null
  }, [lines])
  const pendingPrompt =
    prompt && !(dismissed && dismissed.id === session?.id && dismissed.lines === lines.length)
      ? prompt
      : null
  const activeLine =
    [...lines]
      .reverse()
      .find(
        (line) => line.trim() && !line.startsWith("[serve]") && !line.startsWith("[prompt] ")
      ) ?? null
  const current = status ?? session
  const running = current?.status === "starting" || current?.status === "running"
  const terminal = current?.status === "exited" || current?.status === "error"

  useEffect(() => {
    if (!terminal || !session || completionRef.current === session.id) return
    completionRef.current = session.id
    const paired = session.command === "tracker.pair"
    if (!paired) {
      fetchTrackerBindings()
        .then(({ bindings: found }) => setBindings(found))
        .catch(() => {})
    }
    if (current?.status === "exited" && (current.exitCode ?? 0) === 0) {
      toast.success(paired ? "Lighthouse tracker paired." : "Mantis tracker binding saved.")
    } else {
      toast.error(
        paired
          ? "Tracker pairing failed. See the status below."
          : "Tracker identification failed. See the status below."
      )
    }
  }, [terminal, session, current, toast])

  if (backend === null) {
    return (
      <div className="rounded-lg border border-white/10 bg-white/[0.02] p-4">
        <div className="flex items-center gap-2 text-sm text-white/75">
          <Check className="size-4 text-emerald-400" />
          Quest identifies its left and right controllers automatically.
        </div>
      </div>
    )
  }

  const binding = bindings[backend]
  const label = backend === "survive" ? "Lighthouse" : "Ultimate"
  const pairing = session?.command === "tracker.pair"

  async function pair() {
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const { session: started } = await startDiagnosticsRun("tracker.pair", {})
      setSession(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function identify() {
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const { session: started } = await startDiagnosticsRun("tracker.identify", {
        backend,
        web_prompts: true,
      })
      setSession(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function capture() {
    if (!session) return
    setDismissed({ id: session.id, lines: lines.length })
    try {
      await sendSessionInput(session.id)
    } catch (error) {
      setDismissed(null)
      toast.error(String(error))
    }
  }

  async function stop() {
    if (!session) return
    setBusy(true)
    try {
      await stopSession(session.id)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <RadioTower className="size-4 text-white/50" />
        <span className="text-sm font-medium">{label} tracker binding</span>
        <span
          className={
            binding?.complete
              ? "rounded-full bg-emerald-400/10 px-2 py-0.5 text-[11px] text-emerald-300"
              : "rounded-full bg-amber-400/10 px-2 py-0.5 text-[11px] text-amber-300"
          }
        >
          {binding?.complete ? "Left + right bound" : "Not configured"}
        </span>
        {!running && (
          <div className="ml-auto flex flex-wrap gap-2">
            {backend === "survive" && (
              <Button variant="outline" size="sm" onClick={pair} disabled={busy}>
                {busy ? <Loader2 className="animate-spin" /> : <Bluetooth />}
                Pair tracker
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={identify} disabled={busy}>
              {busy ? <Loader2 className="animate-spin" /> : <RadioTower />}
              {binding?.complete ? "Identify again" : "Identify trackers"}
            </Button>
          </div>
        )}
      </div>
      <p className="max-w-prose text-xs leading-relaxed text-white/40">
        {backend === "survive"
          ? "Pair each tracker with its Watchman dongle first, then identify which Mantis it is mounted to. Both steps run on this host without SteamVR."
          : "The tracker serials do not indicate which Mantis they are mounted to. This guided check watches which tracker moves for each side and saves the mapping on this host."}
      </p>
      {binding?.complete && !running && !terminal && (
        <p className="font-mono text-[11px] text-white/35">
          Left {binding.left} · Right {binding.right}
        </p>
      )}
      {running && (
        <div className="flex flex-col gap-2 rounded-md border border-white/10 bg-black/20 p-3">
          {pendingPrompt && !pairing ? (
            <>
              <p className="text-sm text-amber-100/85">{pendingPrompt}</p>
              <Button size="sm" className="self-start" onClick={capture}>
                Start 3-second capture
              </Button>
            </>
          ) : dismissed?.id === session?.id ? (
            <p className="flex items-center gap-2 text-sm text-white/60">
              <Loader2 className="size-4 animate-spin" /> Capturing motion…
            </p>
          ) : (
            <p className="flex items-center gap-2 text-sm text-white/60">
              <Loader2 className="size-4 animate-spin" />
              {activeLine ?? (pairing ? "Starting dongle pairing…" : "Waiting for trackers…")}
            </p>
          )}
          <Button variant="ghost" size="sm" className="self-start" onClick={stop} disabled={busy}>
            <Square /> Stop
          </Button>
        </div>
      )}
      {terminal && !running && (
        <p
          className={
            current?.status === "exited" && (current.exitCode ?? 0) === 0
              ? "text-xs text-emerald-300/80"
              : "text-xs text-red-300/80"
          }
        >
          {current?.status === "exited" && (current.exitCode ?? 0) === 0
            ? pairing
              ? "Tracker paired. Pair the other tracker if needed, then identify trackers."
              : "Binding saved. Mantis teleop can start now."
            : (activeLine ??
              current?.error ??
              (pairing ? "Tracker pairing failed." : "Tracker identification failed."))}
        </p>
      )}
    </div>
  )
}
