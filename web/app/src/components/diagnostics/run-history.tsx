import { useEffect, useMemo, useState } from "react"
import { ChevronRight, History, Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { LogConsole } from "@/components/log-console"
import { JointFilter } from "@/components/diagnostics/joint-filter"
import {
  TelemetryChart,
  type ChartSeries,
  type ChartView,
} from "@/components/diagnostics/telemetry-chart"
import { cn } from "@/lib/utils"
import {
  JOINTS,
  JOINT_COLORS,
  fetchDiagnosticsRun,
  jointLabel,
  motorKey,
  type ArmSide,
  type DiagnosticsRunData,
  type DiagnosticsRunMeta,
  type JointName,
} from "@/lib/telemetry"

function fmtWhen(t: number): string {
  return new Date(t * 1000).toLocaleString(undefined, { hour12: false })
}

function fmtDuration(meta: DiagnosticsRunMeta): string {
  if (meta.endedAt == null) return "running"
  const s = Math.max(0, Math.round(meta.endedAt - meta.startedAt))
  if (s < 90) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 90 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`
}

function runBadge(meta: DiagnosticsRunMeta): {
  variant: "success" | "destructive" | "neutral"
  text: string
} {
  if (meta.status === "running") return { variant: "neutral", text: "running" }
  if (meta.status === "error" || (meta.exitCode ?? 0) !== 0)
    return { variant: "destructive", text: `failed (${meta.exitCode ?? "?"})` }
  return { variant: "success", text: "ok" }
}

/**
 * Past diagnostics runs (ROM test, homing, …) with their captured telemetry
 * re-charted — same joint filter and zoom/pan as the live charts. Captures
 * record position + torque (velocity is not cached by the motor layer during
 * a script's own control loop).
 */
export function RunHistory({
  runs,
  loading,
  onRefresh,
}: {
  runs: DiagnosticsRunMeta[]
  loading: boolean
  onRefresh: () => void
}) {
  const [openId, setOpenId] = useState<string | null>(null)

  return (
    <Card className="gap-3 p-4">
      <div className="flex items-center gap-2">
        <History className="size-4 text-white/40" />
        <h3 className="font-heading text-sm font-semibold">Run history</h3>
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto text-white/50"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? <Loader2 className="animate-spin" /> : "Refresh"}
        </Button>
      </div>
      {runs.length === 0 ? (
        <p className="text-sm text-white/35">
          No runs yet — launch a diagnostic above and it will be recorded here.
        </p>
      ) : (
        <div className="flex flex-col divide-y divide-white/[0.06]">
          {runs.map((meta) => {
            const open = openId === meta.id
            const badge = runBadge(meta)
            return (
              <div key={meta.id} className="py-1">
                <button
                  type="button"
                  onClick={() => setOpenId(open ? null : meta.id)}
                  className="flex w-full items-center gap-3 rounded-md px-1 py-2 text-left hover:bg-white/[0.03]"
                >
                  <ChevronRight
                    className={cn(
                      "size-4 shrink-0 text-white/40 transition-transform",
                      open && "rotate-90"
                    )}
                  />
                  <span className="font-mono text-xs text-white/85">{meta.command}</span>
                  <Badge variant={badge.variant}>{badge.text}</Badge>
                  <span className="ml-auto hidden text-xs text-white/40 sm:inline">
                    {fmtDuration(meta)}
                  </span>
                  <span className="text-xs text-white/40">{fmtWhen(meta.startedAt)}</span>
                </button>
                {open && <RunDetail id={meta.id} />}
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}

function RunDetail({ id }: { id: string }) {
  // Keyed by run id so switching runs shows the loading state without a
  // synchronous reset inside the effect.
  const [result, setResult] = useState<{
    id: string
    data?: DiagnosticsRunData
    error?: string
  } | null>(null)
  const [arm, setArm] = useState<ArmSide>("left")
  const [hiddenJoints, setHiddenJoints] = useState<Set<JointName>>(new Set())
  const [view, setView] = useState<ChartView | null>(null)
  const [showLog, setShowLog] = useState(false)

  useEffect(() => {
    let active = true
    fetchDiagnosticsRun(id)
      .then((d) => {
        if (active) setResult({ id, data: d })
      })
      .catch((e) => {
        if (active) setResult({ id, error: String(e).replace(/^Error:\s*/, "") })
      })
    return () => {
      active = false
    }
  }, [id])

  const data = result?.id === id ? (result.data ?? null) : null
  const error = result?.id === id ? (result.error ?? null) : null

  // Only offer arms that actually appear in the capture.
  const arms = useMemo(() => {
    const present = new Set<ArmSide>()
    for (const frame of data?.frames ?? []) {
      for (const key of Object.keys(frame.m)) {
        present.add(key.split(":")[0] as ArmSide)
        if (present.size === 2) return ["left", "right"] as ArmSide[]
      }
    }
    return [...present]
  }, [data])

  const effectiveArm = arms.includes(arm) ? arm : (arms[0] ?? "left")
  const series: ChartSeries[] = useMemo(
    () =>
      JOINTS.filter((j) => !hiddenJoints.has(j)).map((joint) => ({
        key: motorKey(effectiveArm, joint),
        label: jointLabel(joint),
        color: JOINT_COLORS[joint],
      })),
    [effectiveArm, hiddenJoints]
  )

  // Default view fits the whole capture; zoom/pan pins a sub-range.
  const fitView = useMemo<ChartView | null>(() => {
    const frames = data?.frames ?? []
    if (frames.length === 0) return null
    return { t0: frames[0].t, t1: frames[frames.length - 1].t }
  }, [data])
  const effectiveView = view ?? fitView

  if (error) return <p className="px-8 pb-3 text-sm text-red-300">{error}</p>
  if (!data)
    return (
      <div className="flex items-center gap-2 px-8 pb-3 text-sm text-white/40">
        <Loader2 className="size-4 animate-spin" /> Loading run…
      </div>
    )

  const hasFrames = data.frames.length > 0 && effectiveView != null

  return (
    <div className="flex flex-col gap-3 px-1 pt-1 pb-3 sm:px-8">
      {hasFrames && (
        <div className="flex flex-wrap items-center gap-2">
          {arms.length > 1 && (
            <div className="flex overflow-hidden rounded-md border border-white/10">
              {arms.map((a) => (
                <button
                  key={a}
                  type="button"
                  onClick={() => setArm(a)}
                  className={cn(
                    "px-2.5 py-1 text-xs capitalize transition-colors",
                    effectiveArm === a
                      ? "bg-[#eff483]/15 text-[#eff483]"
                      : "text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {a} arm
                </button>
              ))}
            </div>
          )}
          <JointFilter hidden={hiddenJoints} onChange={setHiddenJoints} />
          {view != null && (
            <Button
              variant="ghost"
              size="sm"
              className="text-white/50"
              onClick={() => setView(null)}
            >
              Reset zoom
            </Button>
          )}
        </div>
      )}
      {hasFrames ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <TelemetryChart
            title="Position"
            unit="rad"
            series={series}
            frames={data.frames}
            version={0}
            metric={0}
            view={effectiveView}
            onViewChange={setView}
          />
          <TelemetryChart
            title="Torque"
            unit="Nm"
            series={series}
            frames={data.frames}
            version={0}
            metric={2}
            view={effectiveView}
            onViewChange={setView}
          />
        </div>
      ) : (
        <p className="text-sm text-white/35">
          No telemetry was captured for this run (the script owned the CAN bus and did not write
          its own capture).
        </p>
      )}
      <div>
        <Button variant="ghost" size="sm" onClick={() => setShowLog((v) => !v)}>
          {showLog ? "Hide log" : `Show log (${data.log.length} lines)`}
        </Button>
      </div>
      {showLog && <LogConsole lines={data.log} />}
    </div>
  )
}
