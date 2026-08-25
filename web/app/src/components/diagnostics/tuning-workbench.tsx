import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Trash2,
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

const COMMANDED_COLOR = "rgba(255,255,255,0.45)"
const ACTUAL_COLOR = "#eff483"
const NOISY_COLOR = "rgba(230,103,103,0.5)"

const ARM_JOINT_OPTIONS = [
  "shoulder_1",
  "shoulder_2",
  "shoulder_3",
  "elbow",
  "wrist_1",
  "wrist_2",
  "wrist_3",
]

// Run kinds this workbench presents. Anything else in the store (e.g. old
// offline-analysis artifacts) is hidden rather than half-rendered.
const KNOWN_KINDS = new Set(["sine", "step", "motion", "filter"])

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
    key: "filter",
    label: "Filter",
    command: "tune.filter",
    description:
      "Test the teleop filter stack without moving the robot: inject stalls, " +
      "outliers, and jitter into a clean motion (a synthetic sine, or a " +
      "committed reference motion) and replay it through the production " +
      "smoothing chain. The output is scored against the clean signal — same " +
      "seed, same corrupted stream, so runs compare exactly.",
    presets: { save_run: true },
    fields: [
      { key: "motion", label: "motion", type: "select", options: [] },
      { key: "amp", label: "sine amp (rad)", type: "number", placeholder: "0.3" },
      { key: "freq", label: "sine freq (Hz)", type: "number", placeholder: "0.5" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "10" },
      { key: "jitter", label: "jitter (rad RMS)", type: "number", placeholder: "0.005" },
      { key: "outlier_amp", label: "outlier (rad)", type: "number", placeholder: "0.2" },
      { key: "outlier_rate", label: "outliers /s", type: "number", placeholder: "0.5" },
      { key: "stall_ms", label: "stall (ms)", type: "number", placeholder: "150" },
      { key: "stall_rate", label: "stalls /s", type: "number", placeholder: "0.5" },
      { key: "cutoff", label: "cutoff (Hz)", type: "number", placeholder: "config", advanced: true },
      { key: "seed", label: "seed", type: "number", placeholder: "0", advanced: true },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: [],
    drivesMotors: false,
  },
  {
    key: "motion",
    label: "Recorded motion",
    command: "tune.motion",
    description:
      "Replay a committed reference motion through the production control " +
      "path and score joint-space tracking per joint. Gain overrides (e.g. " +
      "left.elbow.kd=4.5 shoulder_3.kd_host=0, space-separated) apply for " +
      "this run only — run once plain, once with overrides, and compare the " +
      "scores in the run list.",
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
    key: "build",
    label: "Build motion",
    command: "motion.build",
    description:
      "Turn a teleop flight recording into a reference motion: clip to the " +
      "engaged span, resample, smooth, and project through the collision " +
      "solver. The motion is then selectable under Recorded motion (commit " +
      "it to git to run it on other robots). Record with " +
      "axol teleop --teleop.jitter_record PREFIX first.",
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
/* Run presentation helpers (list rows, per-joint charts and scores)  */
/* ------------------------------------------------------------------ */

function toDeg(v: number): number {
  return (v * 180) / Math.PI
}

function fmtNum(v: unknown, digits = 2): string {
  if (v == null || typeof v !== "number" || !Number.isFinite(v)) return "–"
  const a = Math.abs(v)
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  return v.toFixed(digits)
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

/**
 * The one comparison number per run, shown in the run list: how close the
 * joints tracked the commanded positions. Motion runs report the mean
 * per-joint tracking RMS in degrees; sine/step report their ranking score
 * (tracking RMS dominated — lower is better on both).
 */
function headline(meta: TuningRunMeta): { label: string; value: string } | null {
  const m = meta.metrics as Record<string, unknown>
  const num = (v: unknown): number | null => (typeof v === "number" ? v : null)
  if (meta.kind === "motion") {
    const v = num(m.mean_rms_err)
    return v == null ? null : { label: "tracking", value: `${fmtNum(toDeg(v))}°` }
  }
  if (meta.kind === "filter") {
    const v = num(m.mean_rms_lagfree)
    return v == null ? null : { label: "residual", value: `${fmtNum(toDeg(v))}°` }
  }
  if (meta.kind === "sine" || meta.kind === "step") {
    const v = num(m.score)
    return v == null ? null : { label: "score", value: fmtNum(v, 3) }
  }
  return null
}

/** One per-joint chart: commanded vs actual position for a single joint. */
interface JointChart {
  joint: string
  series: RunChartSeries[]
}

/** Commanded-vs-actual charts for every joint of `arm` that actually moved. */
function motionJointCharts(run: TuningRunData, arm: string): JointChart[] {
  const columns = (run.meta.params.columns as string[] | undefined) ?? []
  const t = run.series.t ?? []
  const out: JointChart[] = []
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i]
    if (!name?.startsWith(`${arm}.`)) continue
    const commanded = run.series[`target/${i}`]
    const actual = run.series[`actual/${i}`]
    if (!commanded || !actual || !actual.some((v) => v != null)) continue
    // Only joints that were actually commanded to move (> ~1° of travel).
    let min = Infinity
    let max = -Infinity
    for (const v of commanded) {
      if (v == null) continue
      if (v < min) min = v
      if (v > max) max = v
    }
    if (max - min < 0.017) continue
    out.push({
      joint: name.slice(arm.length + 1),
      series: [
        { label: "commanded", color: COMMANDED_COLOR, x: t, data: commanded },
        { label: "actual", color: ACTUAL_COLOR, x: t, data: actual },
      ],
    })
  }
  return out
}

/**
 * Charts for a filter run: clean reference, corrupted input, and the filter
 * stack's output, per channel. `arm` is null for runs whose channels carry
 * no arm prefix (the synthetic sine).
 */
function filterJointCharts(run: TuningRunData, arm: string | null): JointChart[] {
  const columns = (run.meta.params.columns as string[] | undefined) ?? []
  const t = run.series.t ?? []
  const out: JointChart[] = []
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i]
    if (arm != null && !name?.startsWith(`${arm}.`)) continue
    const clean = run.series[`clean/${i}`]
    const noisy = run.series[`noisy/${i}`]
    const filtered = run.series[`filtered/${i}`]
    if (!clean || !filtered) continue
    if (columns.length > 1) {
      let min = Infinity
      let max = -Infinity
      for (const v of clean) {
        if (v == null) continue
        if (v < min) min = v
        if (v > max) max = v
      }
      if (max - min < 0.017) continue
    }
    const series: RunChartSeries[] = [
      { label: "clean", color: COMMANDED_COLOR, x: t, data: clean },
    ]
    if (noisy) series.push({ label: "noisy input", color: NOISY_COLOR, x: t, data: noisy })
    series.push({ label: "filtered", color: ACTUAL_COLOR, x: t, data: filtered })
    out.push({ joint: arm != null ? name.slice(arm.length + 1) : name, series })
  }
  return out
}

/** The arms a run has chartable per-joint data for (single-arm rigs chart one). */
function runArms(run: TuningRunData): string[] {
  if (run.meta.kind === "motion") {
    return ["left", "right"].filter((arm) => motionJointCharts(run, arm).length > 0)
  }
  if (run.meta.kind === "filter") {
    return ["left", "right"].filter((arm) => filterJointCharts(run, arm).length > 0)
  }
  return []
}

/** The one commanded-vs-actual chart of a single-joint sine/step run. */
function pidJointChart(run: TuningRunData): JointChart | null {
  const t = run.series.t
  const commanded = run.series.target
  const actual = run.series.actual
  if (!t || !commanded || !actual) return null
  return {
    joint: run.meta.joint ?? "joint",
    series: [
      { label: "commanded", color: COMMANDED_COLOR, x: t, data: commanded },
      { label: "actual", color: ACTUAL_COLOR, x: t, data: actual },
    ],
  }
}

/** A scorecard column: which metric key, how to show it. */
interface ScoreCol {
  key: string
  label: string
  /** Convert radians to degrees for display. */
  deg?: boolean
  digits?: number
}

const MOTION_COLS: ScoreCol[] = [
  { key: "rms_err", label: "tracking RMS °", deg: true, digits: 3 },
  { key: "lag_ms", label: "lag ms", digits: 0 },
  { key: "err_band_mid", label: "jitter °", deg: true, digits: 3 },
  { key: "amplification", label: "ringing ×" },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
]

const SINE_COLS: ScoreCol[] = [
  { key: "rms", label: "tracking RMS °", deg: true, digits: 3 },
  { key: "max", label: "max err °", deg: true, digits: 3 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
  { key: "pos_ripple", label: "ripple", digits: 4 },
  { key: "score", label: "score", digits: 3 },
]

const FILTER_COLS: ScoreCol[] = [
  { key: "input_rms", label: "noise in °", deg: true, digits: 3 },
  { key: "rms_err", label: "error out °", deg: true, digits: 3 },
  { key: "rms_err_lagfree", label: "lag-free °", deg: true, digits: 3 },
  { key: "lag_ms", label: "lag ms", digits: 0 },
  { key: "jitter_passed", label: "jitter passed ×" },
  { key: "peak_err", label: "peak err °", deg: true, digits: 3 },
  { key: "accel_peak", label: "peak accel rad/s²", digits: 1 },
]

const STEP_COLS: ScoreCol[] = [
  { key: "settling_s", label: "settling s" },
  { key: "overshoot", label: "overshoot °", deg: true, digits: 3 },
  { key: "ss_rms", label: "steady-state RMS °", deg: true, digits: 3 },
  { key: "ring_hz", label: "ring Hz", digits: 1 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
  { key: "score", label: "score", digits: 3 },
]

const SCORE_LEGEND: Record<string, string> = {
  motion:
    "tracking RMS = average distance from the commanded joint position " +
    "(lower is better; the run-list number is the all-joint mean). lag = " +
    "command→measurement delay. jitter = 3–15 Hz vibration in the error — " +
    "what the operator feels. ringing ×>1 = the joint oscillates more than " +
    "commanded. torque chatter = cycle-to-cycle torque noise.",
  sine:
    "tracking RMS = average distance from the commanded sine (lower is " +
    "better). score = RMS + 0.2 × worst excursion, the number to compare " +
    "runs by. torque chatter / ripple = high-frequency roughness.",
  step:
    "settling = time to stay within 5% of the step. overshoot = travel past " +
    "the target. ring Hz = post-step oscillation frequency, if any. score " +
    "folds settling, overshoot, and steady-state error — lower is better.",
  filter:
    "noise in = error the injected noise put on the input; error out = " +
    "what's left after the stack (raw, includes the stack's delay); " +
    "lag-free = residual with the delay removed — the cleanliness number. " +
    "jitter passed <1 = the 3–15 Hz band was attenuated. peak accel must " +
    "stay under the teleop limit, so outliers and stall catch-ups can " +
    "never slam the arm. Error during a stall is missing data, not filter " +
    "failure — the filter owns the smooth catch-up.",
}

/** Per-joint score rows for one run, in display units. */
function scoreRows(
  meta: TuningRunMeta,
  arm: string | null
): { cols: ScoreCol[]; rows: { joint: string; values: Record<string, unknown> }[] } | null {
  const m = meta.metrics as Record<string, unknown>
  if (meta.kind === "motion" || meta.kind === "filter") {
    const perJoint = m.per_joint as Record<string, Record<string, unknown>> | undefined
    if (!perJoint) return null
    const rows = Object.entries(perJoint)
      .filter(([name]) => arm == null || name.startsWith(`${arm}.`))
      .map(([name, values]) => ({
        joint: arm == null ? name : name.slice(arm.length + 1),
        values,
      }))
    const cols = meta.kind === "motion" ? MOTION_COLS : FILTER_COLS
    return rows.length > 0 ? { cols, rows } : null
  }
  if (meta.kind === "sine" || meta.kind === "step") {
    return {
      cols: meta.kind === "sine" ? SINE_COLS : STEP_COLS,
      rows: [{ joint: meta.joint ?? "joint", values: m }],
    }
  }
  return null
}

function fmtScore(values: Record<string, unknown>, col: ScoreCol): string {
  const v = values[col.key]
  if (typeof v !== "number" || !Number.isFinite(v)) return "–"
  return fmtNum(col.deg ? toDeg(v) : v, col.digits ?? 2)
}

/* ------------------------------------------------------------------ */
/* The workbench                                                       */
/* ------------------------------------------------------------------ */

/**
 * The tuning workbench: pick what to run (sine / step / filter noise test /
 * recorded motion), type the numbers inline, hit Run — and the result lands
 * straight on the graphs below. Everything is joint space: per joint, one
 * chart of commanded vs actual position (clean vs noisy vs filtered for
 * filter runs), split into left/right arm tabs when a run covers both arms,
 * and one score row per joint. Runs are compared by their headline tracking
 * score in the run list.
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
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [run, setRun] = useState<TuningRunData | null>(null)
  // Which arm's joints to chart for a motion run — mirrors the live
  // telemetry arm toggle (and starts from the same remembered choice).
  const [arm, setArm] = useState<string>(
    () => localStorage.getItem("axolDiagArm") ?? "left"
  )

  const tuningCommandIds = useMemo(() => new Set(TABS.map((t) => t.command)), [])
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
        const known = runs.filter((r) => KNOWN_KINDS.has(r.kind))
        setRuns(known)
        return known
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

  // The selection setter clears loaded data immediately so a stale chart
  // never shows under a new selection; the effect below only fetches.
  const select = useCallback((id: string | null) => {
    setSelectedId(id)
    setRun(null)
  }, [])

  useEffect(() => {
    if (!selectedId) return
    let active = true
    fetchTuningRun(selectedId)
      .then((r) => {
        if (!active) return
        setRun(r)
        // Keep the arm toggle on an arm the run actually has data for.
        const arms = runArms(r)
        if (arms.length > 0) {
          setArm((prev) => (arms.includes(prev) ? prev : arms[0]))
        }
      })
      .catch((e) => toast.error(String(e)))
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- toast is stable
  }, [selectedId])

  // When a tuning run we launched finishes, pull the new artifacts in and
  // show the newest run's graphs right away.
  const prevActive = useRef<string | null>(null)
  const knownNewest = useRef<string | null>(null)
  useEffect(() => {
    const was = prevActive.current
    prevActive.current = activeCommand
    if (activeCommand != null || was == null || !tuningCommandIds.has(was)) return
    refreshRuns().then((fresh) => {
      const newest = fresh[0]
      if (!newest || newest.id === knownNewest.current) return
      knownNewest.current = newest.id
      select(newest.id)
    })
    if (was === "motion.build") refreshMotions()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fires on run completion only
  }, [activeCommand])

  // Remember the newest run we've seen so completion only auto-selects
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

  const meta = run?.meta ?? null
  const arms = useMemo(() => (run ? runArms(run) : []), [run])
  const armed = arms.length > 0
  const jointCharts = useMemo(() => {
    if (!run) return []
    if (run.meta.kind === "motion") return motionJointCharts(run, arm)
    if (run.meta.kind === "filter") {
      return filterJointCharts(run, armed ? arm : null)
    }
    const single = pidJointChart(run)
    return single ? [single] : []
  }, [run, arm, armed])
  const scores = meta ? scoreRows(meta, armed ? arm : null) : null
  const legend = meta ? SCORE_LEGEND[meta.kind] : null

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteTuningRun(id)
        setRuns((prev) => prev.filter((r) => r.id !== id))
        if (selectedId === id) select(null)
      } catch (e) {
        toast.error(String(e))
      }
    },
    [selectedId, select, toast]
  )

  const clearAll = useCallback(async () => {
    try {
      await clearTuningRuns()
      setRuns([])
      select(null)
    } catch (e) {
      toast.error(String(e))
    }
  }, [select, toast])

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

      {/* Selected run: what it is, arm tabs, per-joint graphs, scores. */}
      {meta && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant="neutral">{meta.kind}</Badge>
          <span className="text-white/60">
            {meta.joint ? `${meta.side} ${meta.joint}` : ""}
            {meta.params.motion ? `${meta.params.motion as string}` : ""}
            {meta.params.source ? `${meta.params.source as string}` : ""}
            {Object.keys(meta.gains).length > 0 ? ` · ${gainsSummary(meta.gains)}` : ""}
            {meta.label ? ` · ${meta.label}` : ""}
          </span>
          {armed && (
            <span className="ml-2 flex overflow-hidden rounded-md border border-white/10">
              {arms.map((a) => (
                <button
                  key={a}
                  type="button"
                  onClick={() => setArm(a)}
                  className={cn(
                    "px-3 py-1 text-xs capitalize transition-colors",
                    arm === a
                      ? "bg-[#eff483]/15 text-[#eff483]"
                      : "text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {a} arm
                </button>
              ))}
            </span>
          )}
        </div>
      )}

      {/* Commanded vs actual position, one chart per joint. */}
      {run && jointCharts.length > 0 && (
        <div
          className={cn("grid grid-cols-1 gap-4", jointCharts.length > 1 && "xl:grid-cols-2")}
        >
          {jointCharts.map((c) => (
            <RunChart
              key={c.joint}
              title={c.joint}
              unit="rad"
              series={c.series}
              height={jointCharts.length > 1 ? 190 : 240}
            />
          ))}
        </div>
      )}
      {run && jointCharts.length === 0 && (
        <p className="text-xs text-white/40">
          No joint moved more than 1° in this run — nothing to chart.
        </p>
      )}

      {/* Per-joint scores under the graphs. */}
      {scores && (
        <Card className="gap-3 p-4">
          <h3 className="font-heading text-sm font-semibold">
            Tracking scores{armed ? ` — ${arm} arm` : ""}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full max-w-3xl text-xs">
              <thead>
                <tr className="text-left text-white/40">
                  <th className="py-1 pr-4 font-normal">joint</th>
                  {scores.cols.map((c) => (
                    <th key={c.key} className="py-1 pr-4 font-normal">
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {scores.rows.map((row) => (
                  <tr key={row.joint} className="border-t border-white/[0.06]">
                    <td className="py-1 pr-4 font-sans text-white/55">{row.joint}</td>
                    {scores.cols.map((c) => (
                      <td key={c.key} className="py-1 pr-4 text-white/85">
                        {fmtScore(row.values, c)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {legend && (
            <p className="max-w-3xl text-[0.65rem] leading-relaxed text-white/35">{legend}</p>
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
            series and scores, here and on the CLI.
          </p>
        ) : (
          <div className="flex max-h-64 flex-col gap-1 overflow-y-auto rounded-lg border border-white/10 bg-white/[0.02] p-2">
            {runs.map((r) => {
              const selected = r.id === selectedId
              const head = headline(r)
              return (
                <div
                  key={r.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => select(selected ? null : r.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      select(selected ? null : r.id)
                    }
                  }}
                  className={cn(
                    "flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
                    selected
                      ? "bg-[#eff483]/10 ring-1 ring-[#eff483]/30"
                      : "hover:bg-white/[0.04]"
                  )}
                >
                  <Badge variant="neutral">{r.kind}</Badge>
                  <span className="text-white/70">
                    {[
                      r.side,
                      r.joint,
                      (r.params.motion as string) ?? null,
                      (r.params.source as string) ?? null,
                    ]
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
