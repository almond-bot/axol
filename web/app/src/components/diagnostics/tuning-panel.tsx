import { useCallback, useEffect, useMemo, useState } from "react"
import { GitCompareArrows, Loader2, RefreshCw, Trash2, X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"
import { RunChart, type RunChartSeries } from "@/components/diagnostics/run-chart"
import {
  clearTuningRuns,
  deleteTuningRun,
  fetchTuningRun,
  fetchTuningRuns,
  type TuningRunData,
  type TuningRunMeta,
} from "@/lib/tuning"

// A/B palette: A wears the accent, B the cool counterpart (dashed).
const A_COLOR = "#eff483"
const B_COLOR = "#6bc5f0"
const A_TARGET = "rgba(255,255,255,0.45)"
const B_TARGET = "rgba(255,255,255,0.28)"

const KIND_LABEL: Record<string, string> = {
  sine: "sine",
  step: "step",
  motion: "motion",
  wifi: "wifi",
  filtering: "filtering",
  kinematics: "kinematics",
}

/** The one number worth showing in the run list, per suite. */
function headline(meta: TuningRunMeta): { label: string; value: string } | null {
  const m = meta.metrics as Record<string, unknown>
  const num = (v: unknown): number | null => (typeof v === "number" ? v : null)
  switch (meta.kind) {
    case "sine":
    case "step": {
      const v = num(m.score)
      return v == null ? null : { label: "score", value: fmtMetric(v) }
    }
    case "motion": {
      const v = num(m.mean_rms_err)
      return v == null ? null : { label: "mean rms err", value: fmtMetric(v) }
    }
    case "wifi": {
      const v = num(m.interarrival_p99_ms)
      return v == null ? null : { label: "p99 gap", value: `${fmtMetric(v)} ms` }
    }
    case "filtering": {
      const v = num(m.mean_passthrough)
      return v == null ? null : { label: "pass-through", value: fmtMetric(v) }
    }
    case "kinematics": {
      const rms = m.ee_rms_mm as Record<string, number> | undefined
      if (!rms) return null
      const worst = Math.max(...Object.values(rms))
      return Number.isFinite(worst)
        ? { label: "ee rms", value: `${fmtMetric(worst)} mm` }
        : null
    }
    default:
      return null
  }
}

function fmtMetric(v: number): string {
  const a = Math.abs(v)
  if (a >= 100) return v.toFixed(0)
  if (a >= 1) return v.toFixed(2)
  if (a >= 0.001) return v.toFixed(4)
  if (a === 0) return "0"
  return v.toExponential(2)
}

function fmtWhen(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

function gainsSummary(gains: Record<string, number>): string {
  const entries = Object.entries(gains)
  if (entries.length === 0) return ""
  return entries.map(([k, v]) => `${k}=${v}`).join("  ")
}

/** Scalar metric rows, in insertion order; nested dicts are skipped. */
function scalarRows(metrics: Record<string, unknown>): [string, unknown][] {
  return Object.entries(metrics).filter(
    ([, v]) => v == null || typeof v === "number" || typeof v === "string" || typeof v === "boolean"
  )
}

/** The nested per-joint / per-axis table if the suite has one. */
function nestedTable(
  metrics: Record<string, unknown>
): { title: string; rows: Record<string, Record<string, unknown>> } | null {
  for (const key of ["per_joint", "per_axis"]) {
    const v = metrics[key]
    if (v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length > 0) {
      return { title: key.replace("_", " "), rows: v as Record<string, Record<string, unknown>> }
    }
  }
  return null
}

function fmtCell(v: unknown): string {
  if (v == null) return "–"
  if (typeof v === "number") return fmtMetric(v)
  if (typeof v === "boolean") return v ? "yes" : "no"
  return String(v)
}

interface ChartSpec {
  title: string
  unit?: string
  series: RunChartSeries[]
}

function seriesOf(
  run: TuningRunData,
  key: string
): { x: (number | null)[]; data: (number | null)[] } | null {
  const data = run.series[key]
  if (!data) return null
  const x = run.series.t ?? run.series.arrival_t ?? data.map((_, i) => i)
  return { x, data }
}

/** Overlay one named signal from A (solid) and B (dashed) on a chart. */
function abSeries(
  key: string,
  label: string,
  a: TuningRunData,
  b: TuningRunData | null,
  colorA = A_COLOR,
  colorB = B_COLOR
): RunChartSeries[] {
  const out: RunChartSeries[] = []
  const sa = seriesOf(a, key)
  if (sa) out.push({ label: `${label} (A)`, color: colorA, ...sa })
  if (b) {
    const sb = seriesOf(b, key)
    if (sb) out.push({ label: `${label} (B)`, color: colorB, dashed: true, ...sb })
  }
  return out
}

function subtract(
  a: { x: (number | null)[]; data: (number | null)[] },
  b: { data: (number | null)[] }
): (number | null)[] {
  return a.data.map((v, i) => {
    const w = b.data[i]
    return v == null || w == null ? null : v - w
  })
}

/** Suite-specific chart layout for one run (A) with an optional B overlay. */
function buildCharts(a: TuningRunData, b: TuningRunData | null, column: number | null): ChartSpec[] {
  const kind = a.meta.kind
  if (kind === "sine" || kind === "step") {
    return [
      {
        title: "Target vs actual position",
        unit: "rad",
        series: [
          ...abSeries("target", "target", a, b, A_TARGET, B_TARGET),
          ...abSeries("actual", "actual", a, b),
        ],
      },
      { title: "Tracking error", unit: "rad", series: abSeries("error", "error", a, b) },
      { title: "Torque", unit: "Nm", series: abSeries("torque", "torque", a, b) },
    ]
  }
  if (kind === "motion") {
    if (column == null) return []
    const col = `/${column}`
    const charts: ChartSpec[] = [
      {
        title: "Target vs actual position",
        unit: "rad",
        series: [
          ...abSeries(`target${col}`, "target", a, b, A_TARGET, B_TARGET),
          ...abSeries(`actual${col}`, "actual", a, b),
        ],
      },
    ]
    // Error isn't stored for motion runs — derive it per run.
    const errSeries: RunChartSeries[] = []
    for (const [run, color, dashed, tag] of [
      [a, A_COLOR, false, "A"],
      [b, B_COLOR, true, "B"],
    ] as const) {
      if (!run) continue
      const tgt = seriesOf(run, `target${col}`)
      const act = seriesOf(run, `actual${col}`)
      if (tgt && act) {
        errSeries.push({
          label: `error (${tag})`,
          color,
          dashed,
          x: act.x,
          data: subtract(act, tgt),
        })
      }
    }
    charts.push({ title: "Tracking error", unit: "rad", series: errSeries })
    charts.push({
      title: "Torque",
      unit: "Nm",
      series: abSeries(`torque${col}`, "torque", a, b),
    })
    return charts
  }
  if (kind === "wifi") {
    return [
      {
        title: "VR frame inter-arrival",
        unit: "ms",
        series: abSeries("interarrival_ms", "gap", a, b),
      },
    ]
  }
  if (kind === "filtering") {
    if (column == null) return []
    const col = `/${column % 3}`
    const side = column < 3 ? "l" : "r"
    return [
      {
        title: "Raw vs filtered vs EE target",
        unit: "m",
        series: [
          ...abSeries(`raw_${side}${col}`, "raw", a, b, "rgba(255,255,255,0.35)", B_TARGET),
          ...abSeries(`filt_${side}${col}`, "filtered", a, b, "#9085e9", "#6a5fd0"),
          ...abSeries(`tgt_${side}${col}`, "EE target", a, b),
        ],
      },
    ]
  }
  if (kind === "kinematics") {
    return [
      {
        title: "End-effector error (FK vs commanded target)",
        unit: "m",
        series: [
          ...abSeries("ee_err_l", "left", a, b),
          ...abSeries("ee_err_r", "right", a, b, "#e66767", "#a34848"),
        ],
      },
    ]
  }
  // Unknown suite: chart every 1-D series against t so nothing is invisible.
  return Object.keys(a.series)
    .filter((k) => k !== "t")
    .slice(0, 6)
    .map((k) => ({ title: k, series: abSeries(k, k, a, b) }))
}

/** The selectable columns of a multi-column run (motion / filtering). */
function columnChoices(run: TuningRunData): { index: number; label: string }[] {
  const kind = run.meta.kind
  if (kind === "motion") {
    const columns = (run.meta.params.columns as string[] | undefined) ?? []
    const out: { index: number; label: string }[] = []
    for (let i = 0; i < columns.length; i++) {
      const data = run.series[`actual/${i}`]
      if (!data || !data.some((v) => v != null)) continue
      // Only joints that actually moved are worth charting (> ~1° of travel).
      const tgt = run.series[`target/${i}`]
      if (tgt) {
        let min = Infinity
        let max = -Infinity
        for (const v of tgt) {
          if (v == null) continue
          if (v < min) min = v
          if (v > max) max = v
        }
        if (max - min < 0.017) continue
      }
      out.push({ index: i, label: columns[i] ?? `col ${i}` })
    }
    return out
  }
  if (kind === "filtering") {
    const axes = ["x", "y", "z"]
    const out: { index: number; label: string }[] = []
    for (let i = 0; i < 6; i++) {
      const side = i < 3 ? "l" : "r"
      const key = `raw_${side}/${i % 3}`
      const data = run.series[key]
      if (data && data.some((v) => v != null)) {
        out.push({ index: i, label: `${side === "l" ? "left" : "right"} ${axes[i % 3]}` })
      }
    }
    return out
  }
  return []
}

/**
 * Tuning runs: the persisted, scored experiments behind deterministic tuning
 * — sine/step PID probes, reference-motion replays, and the offline
 * wifi/filtering/kinematics analyses. Select a run for its charts and
 * scorecard; pin a second one to A/B-compare gains on the same axes.
 */
export function TuningPanel({ enabled }: { enabled: boolean }) {
  const toast = useToast()
  const [runs, setRuns] = useState<TuningRunMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [aId, setAId] = useState<string | null>(null)
  const [bId, setBId] = useState<string | null>(null)
  const [runA, setRunA] = useState<TuningRunData | null>(null)
  const [runB, setRunB] = useState<TuningRunData | null>(null)
  const [column, setColumn] = useState<number | null>(null)

  const refresh = useCallback(() => {
    setLoading(true)
    fetchTuningRuns()
      .then(({ runs }) => setRuns(runs))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on connect
    if (enabled) refresh()
  }, [enabled, refresh])

  // Selection setters clear the loaded data immediately (so stale charts
  // never show under a new selection); the effects only fetch.
  const selectA = useCallback((id: string | null) => {
    setAId(id)
    setRunA(null)
  }, [])
  const selectB = useCallback((id: string | null) => {
    setBId(id)
    setRunB(null)
  }, [])

  // Load the selected runs' series. Column resets with A so a stale joint
  // index never points into a different run's columns.
  useEffect(() => {
    if (!aId) return
    let active = true
    fetchTuningRun(aId)
      .then((r) => {
        if (!active) return
        setRunA(r)
        const cols = columnChoices(r)
        setColumn(cols.length > 0 ? cols[0].index : null)
      })
      .catch((e) => toast.error(String(e)))
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- toast is stable
  }, [aId])

  useEffect(() => {
    if (!bId) return
    let active = true
    fetchTuningRun(bId)
      .then((r) => {
        if (active) setRunB(r)
      })
      .catch((e) => toast.error(String(e)))
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- toast is stable
  }, [bId])

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteTuningRun(id)
        setRuns((prev) => prev.filter((r) => r.id !== id))
        if (aId === id) selectA(null)
        if (bId === id) selectB(null)
      } catch (e) {
        toast.error(String(e))
      }
    },
    [aId, bId, selectA, selectB, toast]
  )

  const clearAll = useCallback(async () => {
    try {
      await clearTuningRuns()
      setRuns([])
      selectA(null)
      selectB(null)
    } catch (e) {
      toast.error(String(e))
    }
  }, [selectA, selectB, toast])

  const metaA = runA?.meta ?? null
  const metaB = runB?.meta ?? null
  // B only overlays on the same suite — comparing a sine to a wifi run has
  // no shared axes. It still shows in the metrics table.
  const overlayB = runB && metaA && runB.meta.kind === metaA.kind ? runB : null
  const charts = useMemo(
    () => (runA ? buildCharts(runA, overlayB, column) : []),
    [runA, overlayB, column]
  )
  const columns = useMemo(() => (runA ? columnChoices(runA) : []), [runA])

  const scalarA = metaA ? scalarRows(metaA.metrics as Record<string, unknown>) : []
  const nestedA = metaA ? nestedTable(metaA.metrics as Record<string, unknown>) : null
  const nestedB = metaB ? nestedTable(metaB.metrics as Record<string, unknown>) : null

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-heading text-base font-semibold">Tuning runs</h2>
        <span className="text-xs text-white/40">
          {runs.length > 0 ? `${runs.length} saved` : ""}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={refresh} disabled={!enabled || loading}>
            {loading ? <Loader2 className="animate-spin" /> : <RefreshCw />} Refresh
          </Button>
          {runs.length > 0 && (
            <Button variant="ghost" size="sm" className="text-white/50" onClick={clearAll}>
              <Trash2 /> Clear
            </Button>
          )}
        </div>
      </div>

      {runs.length === 0 ? (
        <Card className="p-4">
          <p className="text-sm text-white/50">
            No tuning runs saved yet. Runs come from the PID probes{" "}
            <code className="text-white/70">axol tune.pid … --save-run</code>, reference-motion
            replays <code className="text-white/70">axol tune.motion --motion NAME</code>, and the
            offline analyses <code className="text-white/70">axol diag.offline SUITE PREFIX
            --save-run</code> — or launch them from the Diagnostics cards above. Each saved run
            appears here with its tracking-accuracy and smoothness scorecard and charts.
          </p>
        </Card>
      ) : (
        <div className="flex max-h-72 flex-col gap-1 overflow-y-auto rounded-lg border border-white/10 bg-white/[0.02] p-2">
          {runs.map((r) => {
            const isA = r.id === aId
            const isB = r.id === bId
            const head = headline(r)
            return (
              <div
                key={r.id}
                role="button"
                tabIndex={0}
                onClick={() => selectA(isA ? null : r.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    selectA(isA ? null : r.id)
                  }
                }}
                className={cn(
                  "flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
                  isA
                    ? "bg-[#eff483]/10 ring-1 ring-[#eff483]/30"
                    : isB
                      ? "bg-[#6bc5f0]/10 ring-1 ring-[#6bc5f0]/30"
                      : "hover:bg-white/[0.04]"
                )}
              >
                {isA && <Badge className="bg-[#eff483]/15 text-[#eff483]">A</Badge>}
                {isB && <Badge className="bg-[#6bc5f0]/15 text-[#6bc5f0]">B</Badge>}
                <Badge variant="neutral">{KIND_LABEL[r.kind] ?? r.kind}</Badge>
                <span className="text-white/70">
                  {[r.side, r.joint, (r.params.motion as string) ?? null]
                    .filter(Boolean)
                    .join(" · ") || "—"}
                </span>
                {Object.keys(r.gains).length > 0 && (
                  <span className="font-mono text-white/45">{gainsSummary(r.gains)}</span>
                )}
                {r.label && <span className="italic text-white/45">{r.label}</span>}
                {head && (
                  <span className="font-mono text-white/60 tabular-nums">
                    {head.label} {head.value}
                  </span>
                )}
                <span className="ml-auto text-white/35">{fmtWhen(r.startedAt)}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn("size-6", isB ? "text-[#6bc5f0]" : "text-white/35")}
                  title={isB ? "Unpin from compare" : "Pin as B for A/B compare"}
                  onClick={(e) => {
                    e.stopPropagation()
                    selectB(isB ? null : r.id)
                  }}
                >
                  <GitCompareArrows />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-6 text-white/35"
                  title="Delete this run"
                  onClick={(e) => {
                    e.stopPropagation()
                    remove(r.id)
                  }}
                >
                  <Trash2 />
                </Button>
              </div>
            )
          })}
        </div>
      )}

      {metaA && (
        <div className="flex flex-col gap-4">
          {/* Selection summary + column picker */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge className="bg-[#eff483]/15 text-[#eff483]">A</Badge>
            <span className="text-white/60">
              {metaA.kind}
              {metaA.joint ? ` · ${metaA.side} ${metaA.joint}` : ""}
              {metaA.params.motion ? ` · ${metaA.params.motion as string}` : ""}
              {Object.keys(metaA.gains).length > 0 ? ` · ${gainsSummary(metaA.gains)}` : ""}
            </span>
            {metaB && (
              <>
                <Badge className="bg-[#6bc5f0]/15 text-[#6bc5f0]">B</Badge>
                <span className="text-white/60">
                  {metaB.kind}
                  {metaB.joint ? ` · ${metaB.side} ${metaB.joint}` : ""}
                  {metaB.params.motion ? ` · ${metaB.params.motion as string}` : ""}
                  {Object.keys(metaB.gains).length > 0 ? ` · ${gainsSummary(metaB.gains)}` : ""}
                </span>
                {!overlayB && (
                  <span className="text-amber-200/70">
                    different suites — metrics only, no chart overlay
                  </span>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-white/40"
                  onClick={() => selectB(null)}
                >
                  <X /> Unpin B
                </Button>
              </>
            )}
          </div>

          {columns.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              <span className="mr-1 text-xs text-white/40">
                {metaA.kind === "motion" ? "Joint:" : "Axis:"}
              </span>
              {columns.map((c) => (
                <button
                  key={c.index}
                  type="button"
                  onClick={() => setColumn(c.index)}
                  className={cn(
                    "rounded-md border px-2 py-0.5 text-xs transition-colors",
                    column === c.index
                      ? "border-[#eff483]/40 bg-[#eff483]/10 text-[#eff483]"
                      : "border-white/10 text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {c.label}
                </button>
              ))}
            </div>
          )}

          {/* Scorecard */}
          {(scalarA.length > 0 || nestedA) && (
            <Card className="gap-3 p-4">
              <h3 className="font-heading text-sm font-semibold">Scorecard</h3>
              {scalarA.length > 0 && (
                <table className="w-full max-w-2xl text-xs">
                  <thead>
                    <tr className="text-left text-white/40">
                      <th className="py-1 pr-4 font-normal">metric</th>
                      <th className="py-1 pr-4 font-normal text-[#eff483]/80">A</th>
                      {metaB && <th className="py-1 font-normal text-[#6bc5f0]/80">B</th>}
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {scalarA.map(([k, v]) => (
                      <tr key={k} className="border-t border-white/[0.06]">
                        <td className="py-1 pr-4 font-sans text-white/55">{k}</td>
                        <td className="py-1 pr-4 text-white/85">{fmtCell(v)}</td>
                        {metaB && (
                          <td className="py-1 text-white/85">
                            {fmtCell((metaB.metrics as Record<string, unknown>)[k])}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {nestedA && (
                <NestedMetricsTable
                  title={nestedA.title}
                  a={nestedA.rows}
                  b={metaB && nestedB ? nestedB.rows : null}
                />
              )}
            </Card>
          )}

          {/* Charts */}
          <div className="grid grid-cols-1 gap-4">
            {charts
              .filter((c) => c.series.length > 0)
              .map((c) => (
                <RunChart key={c.title} title={c.title} unit={c.unit} series={c.series} />
              ))}
          </div>
        </div>
      )}
    </section>
  )
}

/**
 * Per-joint (or per-axis) metric grid. When a B run is pinned, each cell
 * shows A over B so regressions stand out row by row.
 */
function NestedMetricsTable({
  title,
  a,
  b,
}: {
  title: string
  a: Record<string, Record<string, unknown>>
  b: Record<string, Record<string, unknown>> | null
}) {
  const rows = Object.keys(a)
  const cols = rows.length > 0 ? Object.keys(a[rows[0]]) : []
  if (rows.length === 0) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-white/40">
            <th className="py-1 pr-4 font-normal">{title}</th>
            {cols.map((c) => (
              <th key={c} className="py-1 pr-4 font-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((r) => (
            <tr key={r} className="border-t border-white/[0.06]">
              <td className="py-1 pr-4 font-sans text-white/55">{r}</td>
              {cols.map((c) => (
                <td key={c} className="py-1 pr-4">
                  <span className="text-white/85">{fmtCell(a[r]?.[c])}</span>
                  {b && (
                    <span className="ml-1 text-[#6bc5f0]/80">{fmtCell(b[r]?.[c])}</span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {b && (
        <p className="mt-1 text-[0.65rem] text-white/35">
          white = A, <span className="text-[#6bc5f0]/80">blue = B</span>
        </p>
      )}
    </div>
  )
}
