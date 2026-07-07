import { useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, Square } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { LogConsole } from "@/components/log-console"
import { ConfigForm } from "@/components/config-form"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"
import {
  computeArgs,
  flattenFields,
  missingRequired,
  stopSession,
  useSessionLogs,
  type CommandSpec,
  type FormValue,
  type SessionInfo,
} from "@/lib/supervisor"
import { startDiagnosticsRun, type DiagnosticsRunMeta } from "@/lib/telemetry"

/**
 * Launch pad for the diagnostics scripts (motor health/info, homing, gripper
 * check, ROM soak…). Every launch goes through `/api/diagnostics/run`, so it
 * is recorded as a run — telemetry captured while it executes lands in the
 * run history below.
 */
export function ScriptRunner({
  commands,
  disabled,
  onRunStarted,
  onRunFinished,
}: {
  commands: CommandSpec[]
  /** Robot not connected / another operation owns the bus. */
  disabled: boolean
  onRunStarted: (run: DiagnosticsRunMeta) => void
  onRunFinished: () => void
}) {
  const toast = useToast()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [overrides, setOverrides] = useState<Record<string, FormValue>>({})
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)

  const { lines, status } = useSessionLogs(session?.id ?? null)

  const spec = useMemo(
    () => commands.find((c) => c.id === selectedId) ?? null,
    [commands, selectedId]
  )
  const fields = useMemo(() => (spec ? flattenFields(spec.schema) : []), [spec])

  const effective = status ?? session
  const isLive =
    effective != null && (effective.status === "starting" || effective.status === "running")
  // The finished notification fires once per session transition to a terminal
  // state; keyed on the id so a new launch re-arms it.
  const notifiedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!effective || isLive || !session || notifiedRef.current === session.id) return
    notifiedRef.current = session.id
    // Give the server a beat to persist the run record before refreshing.
    const t = setTimeout(onRunFinished, 800)
    return () => clearTimeout(t)
  }, [effective, isLive, session, onRunFinished])

  function selectCommand(id: string) {
    setSelectedId(id)
    setOverrides({})
  }

  async function handleRun() {
    if (!spec) return
    const missing = missingRequired(fields, overrides)
    if (missing.length > 0) {
      toast.error(`Missing required: ${missing.join(", ")}`)
      return
    }
    setBusy(true)
    try {
      const { run, session: s } = await startDiagnosticsRun(
        spec.id,
        computeArgs(fields, overrides)
      )
      setSession(s)
      onRunStarted(run)
    } catch (e) {
      toast.error(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    if (!session) return
    setBusy(true)
    try {
      setSession(await stopSession(session.id))
    } catch (e) {
      toast.error(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {commands.map((cmd) => {
          const active = cmd.id === selectedId
          const running = isLive && session?.command === cmd.id
          return (
            <button
              key={cmd.id}
              type="button"
              onClick={() => selectCommand(cmd.id)}
              className={cn(
                "flex flex-col gap-1 rounded-xl border p-3 text-left transition-all",
                active
                  ? "border-[#eff483]/40 bg-[#eff483]/10"
                  : "border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.05]"
              )}
            >
              <div className="flex items-center gap-2">
                <span className={cn("text-sm font-medium", !active && "text-white/85")}>
                  {cmd.label}
                </span>
                {running && <span className="size-2 animate-pulse rounded-full bg-emerald-400" />}
              </div>
              <span className="line-clamp-2 text-xs text-white/40">{cmd.description}</span>
            </button>
          )
        })}
      </div>

      {spec && (
        <Card className="gap-4 p-4">
          <div className="flex items-center gap-3">
            <h3 className="font-heading text-sm font-semibold">{spec.label}</h3>
            {effective && (
              <Badge
                variant={
                  isLive
                    ? "success"
                    : effective.status === "error" || effective.exitCode
                      ? "destructive"
                      : "neutral"
                }
              >
                {effective.status}
                {effective.exitCode != null ? ` (${effective.exitCode})` : ""}
              </Badge>
            )}
            <div className="ml-auto flex items-center gap-2">
              {isLive ? (
                <Button variant="outline" size="sm" onClick={handleStop} disabled={busy}>
                  <Square /> Stop
                </Button>
              ) : (
                <Button size="sm" onClick={handleRun} disabled={busy || disabled}>
                  {busy ? <Loader2 className="animate-spin" /> : <Play />} Run
                </Button>
              )}
            </div>
          </div>
          {disabled && !isLive && (
            <p className="text-xs text-amber-200/70">
              Connect the robot (and stop any running operation) to run diagnostics.
            </p>
          )}
          {!spec.available && (
            <p className="text-xs text-red-300">Unavailable: {spec.error}</p>
          )}
          {spec.schema.length > 0 && (
            <ConfigForm
              schema={spec.schema}
              overrides={overrides}
              disabled={isLive || busy}
              onChange={(key, value) => setOverrides((prev) => ({ ...prev, [key]: value }))}
              onReset={(key) =>
                setOverrides((prev) => {
                  const next = { ...prev }
                  delete next[key]
                  return next
                })
              }
            />
          )}
          {session && <LogConsole lines={lines} />}
        </Card>
      )}
    </div>
  )
}
