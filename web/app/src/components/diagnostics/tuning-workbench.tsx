import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ChevronDown,
  ChevronRight,
  GitCompareArrows,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Trash2,
  X,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"
import { RunChart, type RunChartSeries } from "@/components/diagnostics/run-chart"
import type { CommandSpec, FormValue } from "@/lib/supervisor"
import {
  clearTuningRuns,
  deleteTuningRun,
  fetchTuningMotions,
  fetchTuningRun,
  fetchTuningRuns,
  type TuningMotion,
  type TuningRunData,
  type TuningRunMeta,
} from "@/lib/tuning"

// A/B palette: A wears the accent, B the cool counterpart (dashed).
const A_COLOR = "#eff483"
const B_COLOR = "#6bc5f0"
const A_TARGET = "rgba(255,255,255,0.45)"
const B_TARGET = "rgba(255,255,255,0.28)"

const ARM_JOINT_OPTIONS = [
  "shoulder_1",
  "shoulder_2",
  "shoulder_3",
  "elbow",
  "wrist_1",
  "wrist_2",
  "wrist_3",
]

/* ------------------------------------------------------------------ */
/* Inline launcher: what to run, its parameters, and the Run button   */
/* ------------------------------------------------------------------ */

interface WbField {
  key: string
  label: string
  type: "number" | "text" | "select" | "boolean"
  options?: string[]
  /** Placeholder shown when empty; empty means "command default". */
  placeholder?: string
  hint?: string
  advanced?: boolean
  /** Tailwind width class for the input (defaults to a narrow number box). */
  width?: string
}

interface WbTab {
  key: string
  label: string
  command: string
  description: string
  presets: Record<string, FormValue>
  fields: WbField[]
  required: string[]
  drivesMotors: boolean
}

const TABS: WbTab[] = [
  {
    key: "sine",
    label: "Sine",
    command: "tune.pid",
    description:
      "Drive one joint through a sine wave and score tracking. Leave kp/kd " +
      "empty to test the configured gains; several space-separated values " +
      "sweep the grid (each candidate becomes its own run).",
    presets: { mode: "sine", save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      { key: "kp", label: "kp", type: "text", placeholder: "config", hint: "space-separated sweeps" },
      { key: "kd", label: "kd", type: "text", placeholder: "config", hint: "space-separated sweeps" },
      { key: "amp", label: "amp (rad)", type: "number", placeholder: "0.175" },
      { key: "freq", label: "freq (Hz)", type: "number", placeholder: "1.0" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "5" },
      { key: "host_kd", label: "kd_host", type: "number", placeholder: "config", advanced: true },
      {
        key: "ff",
        label: "feedforward",
        type: "select",
        options: ["full", "gravity", "friction", "none"],
        advanced: true,
      },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "—", advanced: true },
      {
        key: "target_noise",
        label: "target noise (rad)",
        type: "number",
        placeholder: "off",
        advanced: true,
      },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: ["arm", "joint"],
    drivesMotors: true,
  },
  {
    key: "step",
    label: "Step",
    command: "tune.pid",
    description:
      "Step one joint and score settling, overshoot, and ring frequency. " +
      "Leave kp/kd empty to test the configured gains; several values sweep.",
    presets: { mode: "step", save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      { key: "kp", label: "kp", type: "text", placeholder: "config", hint: "space-separated sweeps" },
      { key: "kd", label: "kd", type: "text", placeholder: "config", hint: "space-separated sweeps" },
      { key: "amp", label: "amp (rad)", type: "number", placeholder: "0.175" },
      { key: "hold", label: "hold (s)", type: "number", placeholder: "2" },
      { key: "host_kd", label: "kd_host", type: "number", placeholder: "config", advanced: true },
      {
        key: "ff",
        label: "feedforward",
        type: "select",
        options: ["full", "gravity", "friction", "none"],
        advanced: true,
      },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "—", advanced: true },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: ["arm", "joint"],
    drivesMotors: true,
  },
  {
    key: "motion",
    label: "Recorded motion",
    command: "tune.motion",
    description:
      "Replay a committed reference motion through the production control " +
      "path and score tracking per joint. Gain overrides (e.g. " +
      "left.elbow.kd=4.5 shoulder_3.kd_host=0, space-separated) apply for " +
      "this run only — run once plain, once with overrides, and compare.",
    presets: {},
    fields: [
      { key: "motion", label: "motion", type: "select", options: [] },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "0.5" },
      {
        key: "gain",
        label: "gain overrides",
        type: "text",
        placeholder: "left.elbow.kd=4.5 …",
        width: "w-64",
      },
      {
        key: "torque_threshold",
        label: "contact Nm",
        type: "number",
        placeholder: "8",
        advanced: true,
      },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: ["motion"],
    drivesMotors: true,
  },
  {
    key: "friction",
    label: "Friction",
    command: "tune.friction",
    description:
      "Identify one joint's friction model (Coulomb, viscous, offset) with a " +
      "bidirectional velocity sweep, for the feedforward. Check save to " +
      "write the fit into this robot's calibration file.",
    presets: {},
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      { key: "save", label: "save to calibration", type: "boolean" },
      {
        key: "velocities",
        label: "velocities (rad/s)",
        type: "text",
        placeholder: "0.1 0.3 0.6 0.9 1.3",
        advanced: true,
        width: "w-44",
      },
      { key: "kp", label: "kp", type: "number", placeholder: "config", advanced: true },
      { key: "kd", label: "kd", type: "number", placeholder: "config", advanced: true },
    ],
    required: ["arm", "joint"],
    drivesMotors: true,
  },
  {
    key: "analysis",
    label: "Analyze recording",
    command: "diag.offline",
    description:
      "Analyze a teleop flight recording without moving the robot: wifi " +
      "transport jitter, the pose filter stack, or IK-injected motion. " +
      "Record with axol teleop --teleop.jitter_record PREFIX first.",
    presets: { save_run: true },
    fields: [
      {
        key: "suite",
        label: "suite",
        type: "select",
        options: ["wifi", "filtering", "kinematics"],
      },
      { key: "prefix", label: "recording prefix", type: "text", placeholder: "/tmp/jit17", width: "w-52" },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: ["suite", "prefix"],
    drivesMotors: false,
  },
  {
    key: "build",
    label: "Build motion",
    command: "motion.build",
    description:
      "Turn a teleop flight recording into a reference motion: clip to the " +
      "engaged span, resample, smooth, and project through the collision " +
      "solver. The motion is then selectable under Recorded motion (commit " +
      "it to git to run it on other robots).",
    presets: {},
    fields: [
      { key: "prefix", label: "recording prefix", type: "text", placeholder: "/tmp/jit17", width: "w-52" },
      { key: "name", label: "motion name", type: "text", placeholder: "reach_and_place" },
      { key: "cutoff", label: "cutoff (Hz)", type: "number", placeholder: "6", advanced: true },
      { key: "rate", label: "rate (Hz)", type: "number", placeholder: "100", advanced: true },
      {
        key: "time_scale",
        label: "time scale",
        type: "number",
        placeholder: "1.0",
        advanced: true,
      },
      { key: "notes", label: "notes", type: "text", placeholder: "provenance", width: "w-40", advanced: true },
    ],
    required: ["prefix", "name"],
    drivesMotors: false,
  },
]

/* ------------------------------------------------------------------ */
/* Run presentation helpers (list rows, scorecard, chart builders)    */
/* ------------------------------------------------------------------ */

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
  return Object.entries(gains)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ")
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
function buildCharts(
  a: TuningRunData,
  b: TuningRunData | null,
  column: number | null
): ChartSpec[] {
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
    charts.push({ title: "Torque", unit: "Nm", series: abSeries(`torque${col}`, "torque", a, b) })
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

/* ------------------------------------------------------------------ */
/* The workbench                                                       */
/* ------------------------------------------------------------------ */

/**
 * The tuning workbench: pick what to run (sine / step / recorded motion /
 * offline analysis), type the numbers inline, hit Run — and the result lands
 * straight on the graphs below. Each new run auto-selects as A with the
 * previous selection sliding to B, so consecutive gain candidates compare on
 * the same axes without any clicking around.
 */
export function TuningWorkbench({
  enabled,
  commands,
  activeCommand,
  busy,
  disabled,
  onLaunch,
  onStop,
}: {
  enabled: boolean
  commands: CommandSpec[]
  /** Command id of the diagnostics run in flight, if any. */
  activeCommand: string | null
  busy: boolean
  disabled: boolean
  onLaunch: (command: string, args: Record<string, FormValue>) => void
  onStop: () => void
}) {
  const toast = useToast()
  const [tabKey, setTabKey] = useState(TABS[0].key)
  const tab = TABS.find((t) => t.key === tabKey) ?? TABS[0]
  const spec = commands.find((c) => c.id === tab.command) ?? null
  // Per-tab form values, kept across tab switches.
  const [values, setValues] = useState<Record<string, Record<string, string>>>({})
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [missing, setMissing] = useState<string[]>([])
  const [motions, setMotions] = useState<TuningMotion[]>([])

  const [runs, setRuns] = useState<TuningRunMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [aId, setAId] = useState<string | null>(null)
  const [bId, setBId] = useState<string | null>(null)
  const [runA, setRunA] = useState<TuningRunData | null>(null)
  const [runB, setRunB] = useState<TuningRunData | null>(null)
  const [column, setColumn] = useState<number | null>(null)

  const tuningCommandIds = useMemo(
    () => new Set(TABS.map((t) => t.command)),
    []
  )
  const runningOurs = activeCommand != null && tuningCommandIds.has(activeCommand)
  const runningThisTab = activeCommand === tab.command

  const refreshMotions = useCallback(() => {
    fetchTuningMotions()
      .then(({ motions }) => setMotions(motions))
      .catch(() => {})
  }, [])

  const refreshRuns = useCallback((): Promise<TuningRunMeta[]> => {
    setLoading(true)
    return fetchTuningRuns()
      .then(({ runs }) => {
        setRuns(runs)
        return runs
      })
      .catch(() => [] as TuningRunMeta[])
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!enabled) return
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on connect
    refreshRuns()
    refreshMotions()
  }, [enabled, refreshRuns, refreshMotions])

  // Selection setters clear loaded data immediately so stale charts never
  // show under a new selection; the effects below only fetch.
  const selectA = useCallback((id: string | null) => {
    setAId(id)
    setRunA(null)
  }, [])
  const selectB = useCallback((id: string | null) => {
    setBId(id)
    setRunB(null)
  }, [])

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

  // When a tuning run we launched finishes, pull the new artifacts in and
  // rotate the comparison: newest run becomes A, the previous A becomes B.
  const prevActive = useRef<string | null>(null)
  const knownNewest = useRef<string | null>(null)
  useEffect(() => {
    const was = prevActive.current
    prevActive.current = activeCommand
    if (activeCommand != null || was == null || !tuningCommandIds.has(was)) return
    const prevA = aId
    refreshRuns().then((fresh) => {
      const newest = fresh[0]
      if (!newest || newest.id === knownNewest.current) return
      knownNewest.current = newest.id
      selectA(newest.id)
      if (prevA && prevA !== newest.id) selectB(prevA)
    })
    if (was === "motion.build") refreshMotions()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fires on run completion only
  }, [activeCommand])

  // Remember the newest run we've seen so completion only rotates for
  // genuinely new artifacts (a canceled run saves nothing).
  useEffect(() => {
    if (runs.length > 0 && knownNewest.current == null) knownNewest.current = runs[0].id
  }, [runs])

  const setValue = useCallback(
    (key: string, v: string) => {
      setValues((prev) => ({ ...prev, [tab.key]: { ...(prev[tab.key] ?? {}), [key]: v } }))
    },
    [tab.key]
  )
  const tabValues = values[tab.key] ?? {}

  function handleRun() {
    const miss = tab.required.filter((k) => !(tabValues[k] ?? "").trim())
    setMissing(miss)
    if (miss.length > 0) return
    const args: Record<string, FormValue> = { ...tab.presets }
    for (const f of tab.fields) {
      const raw = (tabValues[f.key] ?? "").trim()
      if (!raw) continue
      args[f.key] = f.type === "boolean" ? raw === "true" : raw
    }
    onLaunch(tab.command, args)
  }

  const fieldsShown = tab.fields.filter((f) => !f.advanced || showAdvanced)
  const hasAdvanced = tab.fields.some((f) => f.advanced)

  const metaA = runA?.meta ?? null
  const metaB = runB?.meta ?? null
  // B only overlays on the same suite — different suites share no axes. It
  // still shows in the scorecard.
  const overlayB = runB && metaA && runB.meta.kind === metaA.kind ? runB : null
  const charts = useMemo(
    () => (runA ? buildCharts(runA, overlayB, column) : []),
    [runA, overlayB, column]
  )
  const columns = useMemo(() => (runA ? columnChoices(runA) : []), [runA])
  const scalarA = metaA ? scalarRows(metaA.metrics as Record<string, unknown>) : []
  const nestedA = metaA ? nestedTable(metaA.metrics as Record<string, unknown>) : null
  const nestedB = metaB ? nestedTable(metaB.metrics as Record<string, unknown>) : null

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

  return (
    <section className="flex flex-col gap-4">
      <h2 className="font-heading text-base font-semibold">Tuning</h2>

      {/* Launcher: source tabs + inline parameters + Run. */}
      <Card className="gap-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex overflow-hidden rounded-md border border-white/10">
            {TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => {
                  setTabKey(t.key)
                  setMissing([])
                }}
                className={cn(
                  "px-3 py-1.5 text-xs transition-colors",
                  tabKey === t.key
                    ? "bg-[#eff483]/15 text-[#eff483]"
                    : "text-white/50 hover:bg-white/[0.05]"
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          {tab.key === "motion" && motions.length === 0 && (
            <span className="text-xs text-amber-200/70">
              no reference motions yet — build one from a recording first
            </span>
          )}
        </div>
        <p className="max-w-3xl text-xs leading-relaxed text-white/45">{tab.description}</p>

        <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
          {fieldsShown.map((f) => (
            <label key={f.key} className="flex flex-col gap-1">
              <span className="text-[0.65rem] text-white/40">
                {f.label}
                {tab.required.includes(f.key) && <span className="text-[#eff483]/70"> *</span>}
              </span>
              {f.type === "boolean" ? (
                <span className="flex h-8 cursor-pointer items-center gap-2 rounded-md border border-white/10 bg-[#1c1c1c] px-2 text-xs text-white/70">
                  <input
                    type="checkbox"
                    checked={tabValues[f.key] === "true"}
                    disabled={runningOurs || busy}
                    onChange={(e) => setValue(f.key, e.target.checked ? "true" : "")}
                    className="accent-[#eff483]"
                  />
                  {tabValues[f.key] === "true" ? "on" : "off"}
                </span>
              ) : f.type === "select" ? (
                <select
                  value={tabValues[f.key] ?? ""}
                  onChange={(e) => setValue(f.key, e.target.value)}
                  disabled={runningOurs || busy}
                  className={cn(
                    "h-8 rounded-md border border-white/10 bg-[#1c1c1c] px-2 text-xs text-white/85 outline-none focus:border-[#eff483]/40",
                    f.width ?? "w-32"
                  )}
                >
                  <option value="">
                    {tab.required.includes(f.key) ? "select…" : "default"}
                  </option>
                  {(f.key === "motion" ? motions.map((m) => m.name) : (f.options ?? [])).map(
                    (o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    )
                  )}
                </select>
              ) : (
                <input
                  type="text"
                  inputMode={f.type === "number" ? "decimal" : undefined}
                  value={tabValues[f.key] ?? ""}
                  placeholder={f.placeholder}
                  title={f.hint}
                  onChange={(e) => setValue(f.key, e.target.value)}
                  disabled={runningOurs || busy}
                  className={cn(
                    "h-8 rounded-md border border-white/10 bg-[#1c1c1c] px-2 font-mono text-xs text-white/85 outline-none placeholder:text-white/25 focus:border-[#eff483]/40",
                    f.width ?? (f.type === "number" ? "w-24" : "w-28")
                  )}
                />
              )}
            </label>
          ))}
          {hasAdvanced && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-white/40"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? <ChevronDown /> : <ChevronRight />} advanced
            </Button>
          )}
          <div className="ml-auto">
            {runningThisTab ? (
              <Button variant="destructive" size="sm" onClick={onStop} disabled={busy}>
                {busy ? <Loader2 className="animate-spin" /> : <Square />} Stop
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleRun}
                disabled={
                  !enabled ||
                  disabled ||
                  busy ||
                  activeCommand != null ||
                  (spec != null && !spec.available)
                }
              >
                <Play /> Run
              </Button>
            )}
          </div>
        </div>

        {missing.length > 0 && (
          <p className="text-xs text-red-300">Missing required: {missing.join(", ")}</p>
        )}
        {spec && !spec.available && (
          <p className="text-xs text-red-300">Unavailable: {spec.error}</p>
        )}
        {runningOurs && (
          <p className="text-xs text-emerald-300/80">
            Running — the charts below update when it finishes.
          </p>
        )}
      </Card>

      {/* Selection summary + column picker */}
      {metaA && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge className="bg-[#eff483]/15 text-[#eff483]">A</Badge>
          <span className="text-white/60">
            {metaA.kind}
            {metaA.joint ? ` · ${metaA.side} ${metaA.joint}` : ""}
            {metaA.params.motion ? ` · ${metaA.params.motion as string}` : ""}
            {Object.keys(metaA.gains).length > 0 ? ` · ${gainsSummary(metaA.gains)}` : ""}
            {metaA.label ? ` · ${metaA.label}` : ""}
          </span>
          {metaB && (
            <>
              <Badge className="bg-[#6bc5f0]/15 text-[#6bc5f0]">B</Badge>
              <span className="text-white/60">
                {metaB.kind}
                {metaB.joint ? ` · ${metaB.side} ${metaB.joint}` : ""}
                {metaB.params.motion ? ` · ${metaB.params.motion as string}` : ""}
                {Object.keys(metaB.gains).length > 0 ? ` · ${gainsSummary(metaB.gains)}` : ""}
                {metaB.label ? ` · ${metaB.label}` : ""}
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
          {columns.length > 0 && (
            <span className="ml-2 flex flex-wrap items-center gap-1">
              <span className="text-white/40">{metaA.kind === "motion" ? "Joint:" : "Axis:"}</span>
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
            </span>
          )}
        </div>
      )}

      {/* Charts front and center. */}
      {metaA && (
        <div className="grid grid-cols-1 gap-4">
          {charts
            .filter((c) => c.series.length > 0)
            .map((c) => (
              <RunChart key={c.title} title={c.title} unit={c.unit} series={c.series} />
            ))}
        </div>
      )}

      {/* Scorecard under the charts. */}
      {metaA && (scalarA.length > 0 || nestedA) && (
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

      {/* Past runs, compact. */}
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="font-heading text-sm font-semibold text-white/70">Past runs</h3>
          <span className="text-xs text-white/40">
            {runs.length > 0 ? `${runs.length} saved` : ""}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refreshRuns()}
              disabled={!enabled || loading}
            >
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
          <p className="text-xs text-white/40">
            No runs yet — pick a source above and hit Run. Every run is saved with its full time
            series and scorecard, here and on the CLI.
          </p>
        ) : (
          <div className="flex max-h-64 flex-col gap-1 overflow-y-auto rounded-lg border border-white/10 bg-white/[0.02] p-2">
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
                  <Badge variant="neutral">{r.kind}</Badge>
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
      </div>
    </section>
  )
}

/**
 * Per-joint (or per-axis) metric grid. When a B run is pinned, each cell
 * shows A next to B so regressions stand out row by row.
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
                  {b && <span className="ml-1 text-[#6bc5f0]/80">{fmtCell(b[r]?.[c])}</span>}
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
