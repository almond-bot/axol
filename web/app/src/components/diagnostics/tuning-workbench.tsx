import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, RefreshCw, Square, Trash2 } from "lucide-react"
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
  fetchTuningGains,
  fetchTuningMotions,
  fetchTuningRun,
  fetchTuningRuns,
  type TuningGains,
  type TuningMotion,
  type TuningRunData,
  type TuningRunMeta,
} from "@/lib/tuning"

const COMMANDED_COLOR = "rgba(255,255,255,0.45)"
const ACTUAL_COLOR = "#eff483"
const NOISY_COLOR = "rgba(230,103,103,0.5)"
const ERROR_COLOR = "#e6906b"
const MAP_GOOD = "#79c98c"
const MAP_WARN = "#e6c067"
const MAP_BAD = "#e66767"

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
  /** Tailwind width class for the input (defaults to a narrow number box). */
  width?: string
  /**
   * Key into the fetched per-joint gains (`kp` | `kd` | `kd_host` |
   * `kd_host_hz`): the field shows the selected joint's current config value
   * and an empty box means "run with config".
   */
  gainKey?: string
  /** Render a slider next to the value box, over this range. */
  slider?: { min: number; max: number; step: number }
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

/**
 * The gain knobs shared by the sine and step tabs. Each shows the selected
 * joint's current config value (defaults + this robot's calibration) and a
 * slider seeded there; an empty box runs with config. kp/kd also take
 * space-separated sweeps typed into the box. Slider ceilings are the
 * hardware encodings' (kp 500, kd 5) and the hardware-verified host-damping
 * range (kd_host up to the 45 ceiling, band centre up to ~19 Hz).
 */
const GAIN_FIELDS: WbField[] = [
  {
    key: "kp",
    label: "kp",
    type: "text",
    gainKey: "kp",
    slider: { min: 0, max: 500, step: 5 },
    hint: "space-separated sweeps",
  },
  {
    key: "kd",
    label: "kd",
    type: "text",
    gainKey: "kd",
    slider: { min: 0, max: 5, step: 0.05 },
    hint: "space-separated sweeps",
  },
  {
    key: "host_kd",
    label: "kd_host",
    type: "number",
    gainKey: "kd_host",
    slider: { min: 0, max: 60, step: 0.5 },
    hint: "host-side damping via t_ff",
  },
  {
    key: "host_kd_hz",
    label: "kd_host_hz (Hz)",
    type: "number",
    gainKey: "kd_host_hz",
    slider: { min: 1, max: 19, step: 0.1 },
    hint: "band-pass centre of the host damping — aim it at the ring Hz",
  },
]

const TABS: WbTab[] = [
  {
    key: "sine",
    label: "Sine",
    command: "tune.pid",
    description:
      "Drive one joint through a sine wave and score tracking. Gain sliders " +
      "start at the joint's current config value (shown next to each label); " +
      "leave a box empty to run with config, or type several space-separated " +
      "kp/kd values to sweep the grid (each candidate becomes its own run).",
    presets: { mode: "sine", save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      ...GAIN_FIELDS,
      { key: "amp", label: "amp (°)", type: "number", placeholder: "10" },
      { key: "freq", label: "freq (Hz)", type: "number", placeholder: "1.0" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "5" },
      {
        key: "ff",
        label: "feedforward",
        type: "select",
        options: ["full", "gravity", "friction", "none"],
      },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "—" },
      {
        key: "target_noise",
        label: "target noise (°)",
        type: "number",
        placeholder: "off",
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
      "Gain sliders start at the joint's current config value; leave a box " +
      "empty to run with config, or type several kp/kd values to sweep.",
    presets: { mode: "step", save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      ...GAIN_FIELDS,
      { key: "amp", label: "amp (°)", type: "number", placeholder: "10" },
      { key: "hold", label: "hold (s)", type: "number", placeholder: "2" },
      {
        key: "ff",
        label: "feedforward",
        type: "select",
        options: ["full", "gravity", "friction", "none"],
      },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "—" },
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
      "Test the teleop filter stack without moving the robot, one noise " +
      "source at a time: network noise (jitter, outliers, stalls) is " +
      "injected before the pose low-pass, IK noise (solver churn, solution " +
      "jumps) after it — each at its real entry point in the pipeline — or " +
      "both combined. The corrupted stream replays through the production " +
      "smoothing chain and is scored against the clean signal; same seed, " +
      "same corrupted stream, so runs compare exactly.",
    presets: { save_run: true },
    fields: [
      { key: "noise", label: "noise", type: "select", options: ["combined", "network", "ik"] },
      { key: "motion", label: "motion", type: "select", options: [] },
      { key: "amp", label: "sine amp (°)", type: "number", placeholder: "15" },
      { key: "freq", label: "sine freq (Hz)", type: "number", placeholder: "0.5" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "10" },
      { key: "jitter", label: "net jitter (° RMS)", type: "number", placeholder: "0.3" },
      { key: "outlier_amp", label: "net outlier (°)", type: "number", placeholder: "10" },
      { key: "outlier_rate", label: "net outliers /s", type: "number", placeholder: "0.5" },
      { key: "stall_ms", label: "net stall (ms)", type: "number", placeholder: "150" },
      { key: "stall_rate", label: "net stalls /s", type: "number", placeholder: "0.5" },
      { key: "ik_churn", label: "ik churn (° RMS)", type: "number", placeholder: "0.2" },
      { key: "ik_jump_amp", label: "ik jump (°)", type: "number", placeholder: "3" },
      { key: "ik_jump_rate", label: "ik jumps /s", type: "number", placeholder: "0.2" },
      { key: "cutoff", label: "cutoff (Hz)", type: "number", placeholder: "config" },
      { key: "seed", label: "seed", type: "number", placeholder: "0" },
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
        label: "velocities (°/s)",
        type: "text",
        placeholder: "7.2 18 36 54 72",
        width: "w-44",
      },
      {
        key: "kp",
        label: "kp",
        type: "number",
        gainKey: "kp",
        slider: { min: 0, max: 500, step: 5 },
      },
      {
        key: "kd",
        label: "kd",
        type: "number",
        gainKey: "kd",
        slider: { min: 0, max: 5, step: 0.05 },
      },
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
      { key: "cutoff", label: "cutoff (Hz)", type: "number", placeholder: "6" },
      { key: "rate", label: "rate (Hz)", type: "number", placeholder: "100" },
      {
        key: "time_scale",
        label: "time scale",
        type: "number",
        placeholder: "1.0",
      },
      { key: "notes", label: "notes", type: "text", placeholder: "provenance", width: "w-40" },
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

/** Radian series → degrees for display (nulls pass through). */
function degSeries(data: (number | null)[]): (number | null)[] {
  return data.map((v) => (v == null ? null : toDeg(v)))
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
  /** Error lane (reference − output, in degrees) under the position plot. */
  sub: RunChartSeries[]
}

/**
 * The error trace for a chart's lane, in degrees. Position traces overlap
 * whenever tracking is halfway decent — the error at its own scale is where
 * a failure actually shows.
 */
function errorLane(
  t: (number | null)[],
  reference: (number | null)[],
  output: (number | null)[]
): RunChartSeries[] {
  const n = Math.min(reference.length, output.length)
  const err: (number | null)[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const r = reference[i]
    const o = output[i]
    err[i] = r == null || o == null ? null : toDeg(o - r)
  }
  return [{ label: "error °", color: ERROR_COLOR, x: t, data: err }]
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
        { label: "commanded", color: COMMANDED_COLOR, x: t, data: degSeries(commanded) },
        { label: "actual", color: ACTUAL_COLOR, x: t, data: degSeries(actual) },
      ],
      sub: errorLane(t, commanded, actual),
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
      { label: "clean", color: COMMANDED_COLOR, x: t, data: degSeries(clean) },
    ]
    if (noisy) {
      series.push({ label: "noisy input", color: NOISY_COLOR, x: t, data: degSeries(noisy) })
    }
    series.push({ label: "filtered", color: ACTUAL_COLOR, x: t, data: degSeries(filtered) })
    out.push({
      joint: arm != null ? name.slice(arm.length + 1) : name,
      series,
      sub: errorLane(t, clean, filtered),
    })
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
      { label: "commanded", color: COMMANDED_COLOR, x: t, data: degSeries(commanded) },
      { label: "actual", color: ACTUAL_COLOR, x: t, data: degSeries(actual) },
    ],
    sub: errorLane(t, commanded, actual),
  }
}

/** A scorecard column: which metric key, how to show it. */
interface ScoreCol {
  key: string
  label: string
  /** Convert radians to degrees for display. */
  deg?: boolean
  digits?: number
  /**
   * Coloring thresholds in display units, lower-is-better: at or above
   * `warn` the cell turns amber, at or above `bad` red — so a failing joint
   * is visible without reading every number.
   */
  warn?: number
  bad?: number
}

const MOTION_COLS: ScoreCol[] = [
  { key: "rms_err", label: "tracking RMS °", deg: true, digits: 3, warn: 1.0, bad: 2.5 },
  { key: "lag_ms", label: "lag ms", digits: 0, warn: 40, bad: 80 },
  { key: "err_band_mid", label: "jitter °", deg: true, digits: 3, warn: 0.3, bad: 0.8 },
  { key: "amplification", label: "ringing ×", warn: 1.15, bad: 1.5 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
]

const SINE_COLS: ScoreCol[] = [
  { key: "rms", label: "tracking RMS °", deg: true, digits: 3, warn: 1.0, bad: 2.5 },
  { key: "max", label: "max err °", deg: true, digits: 3, warn: 3, bad: 6 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
  { key: "pos_ripple", label: "ripple", digits: 4 },
  { key: "score", label: "score", digits: 3 },
]

const FILTER_COLS: ScoreCol[] = [
  { key: "input_rms", label: "noise in °", deg: true, digits: 3 },
  { key: "rms_err", label: "error out °", deg: true, digits: 3 },
  { key: "rms_err_lagfree", label: "lag-free °", deg: true, digits: 3, warn: 1.0, bad: 2.5 },
  { key: "lag_ms", label: "lag ms", digits: 0, warn: 60, bad: 120 },
  { key: "jitter_passed", label: "jitter passed ×", warn: 0.7, bad: 1.0 },
  { key: "peak_err", label: "peak err °", deg: true, digits: 3 },
  { key: "accel_peak", label: "peak accel °/s²", deg: true, digits: 0 },
]

const STEP_COLS: ScoreCol[] = [
  { key: "settling_s", label: "settling s", warn: 0.4, bad: 0.8 },
  { key: "overshoot", label: "overshoot °", deg: true, digits: 3, warn: 1, bad: 3 },
  { key: "ss_rms", label: "steady-state RMS °", deg: true, digits: 3, warn: 0.3, bad: 0.8 },
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

/** Display value in the column's units, or null when absent. */
function scoreValue(values: Record<string, unknown>, col: ScoreCol): number | null {
  const v = values[col.key]
  if (typeof v !== "number" || !Number.isFinite(v)) return null
  return col.deg ? toDeg(v) : v
}

/** Text color class for a score cell — amber at warn, red at bad. */
function scoreClass(v: number | null, col: ScoreCol): string {
  if (v == null || col.warn == null) return "text-white/85"
  if (v >= (col.bad ?? Infinity)) return "text-red-300"
  if (v >= col.warn) return "text-amber-200"
  return "text-white/85"
}

/* ------------------------------------------------------------------ */
/* Failure map: both arms at once, one bar per joint                  */
/* ------------------------------------------------------------------ */

const MAP_JOINT_ORDER = [...ARM_JOINT_OPTIONS, "gripper"]

/** Which per-joint metric localizes failure for each run kind. */
const MAP_SPECS: Record<
  string,
  { metric: string; deg: boolean; label: string; unit: string; warn: number; bad: number }
> = {
  motion: {
    metric: "rms_err",
    deg: true,
    label: "tracking RMS",
    unit: "°",
    warn: 1.0,
    bad: 2.5,
  },
  filter: {
    metric: "rms_err_lagfree",
    deg: true,
    label: "lag-free residual",
    unit: "°",
    warn: 1.0,
    bad: 2.5,
  },
}

function mapColor(v: number, warn: number, bad: number): string {
  return v >= bad ? MAP_BAD : v >= warn ? MAP_WARN : MAP_GOOD
}

/**
 * The at-a-glance failure map: every joint of both arms in one view, one
 * horizontal bar per joint sized by its headline error and colored by the
 * same thresholds as the score table — the failing joint and side jump out
 * without flipping arm tabs or reading numbers. Clicking a joint switches
 * the charts to that arm and scrolls to that joint's graph.
 */
function FailureMap({
  perJoint,
  kind,
  arm,
  onPick,
}: {
  perJoint: Record<string, Record<string, unknown>>
  kind: string
  arm: string
  onPick: (side: string, joint: string) => void
}) {
  const spec = MAP_SPECS[kind]
  if (!spec) return null
  const sides = ["left", "right"].filter((s) =>
    Object.keys(perJoint).some((k) => k.startsWith(`${s}.`))
  )
  if (sides.length === 0) return null

  const value = (side: string, joint: string): number | null => {
    const v = perJoint[`${side}.${joint}`]?.[spec.metric]
    if (typeof v !== "number" || !Number.isFinite(v)) return null
    return spec.deg ? toDeg(v) : v
  }
  const joints = MAP_JOINT_ORDER.filter((j) =>
    sides.some((s) => `${s}.${j}` in perJoint)
  )
  for (const key of Object.keys(perJoint)) {
    const j = key.split(".").slice(1).join(".")
    if (j && !joints.includes(j)) joints.push(j)
  }
  let maxV = 0
  for (const s of sides) {
    for (const j of joints) {
      const v = value(s, j)
      if (v != null && v > maxV) maxV = v
    }
  }
  if (maxV <= 0) return null

  return (
    <Card className="gap-3 p-4">
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">Failure map</h3>
        <span className="text-xs text-white/35">
          {spec.label} per joint, both arms — click a joint to jump to its graph
        </span>
      </div>
      <div className={cn("grid gap-x-8 gap-y-1", sides.length > 1 && "sm:grid-cols-2")}>
        {sides.map((side) => (
          <div key={side} className="flex flex-col gap-1">
            <span
              className={cn(
                "text-xs font-semibold capitalize",
                side === arm ? "text-[#eff483]" : "text-white/45"
              )}
            >
              {side} arm{side === arm ? " · charted" : ""}
            </span>
            {joints.map((joint) => {
              const v = value(side, joint)
              return (
                <button
                  key={joint}
                  type="button"
                  onClick={() => onPick(side, joint)}
                  className="group flex items-center gap-2 rounded px-1 py-0.5 text-left transition-colors hover:bg-white/[0.05]"
                >
                  <span className="w-24 shrink-0 truncate text-xs text-white/55 group-hover:text-white/85">
                    {joint}
                  </span>
                  <span className="relative h-2.5 min-w-0 flex-1 overflow-hidden rounded-sm bg-white/[0.05]">
                    {v != null && (
                      <span
                        className="absolute inset-y-0 left-0 rounded-sm"
                        style={{
                          width: `${Math.max(2, (v / maxV) * 100)}%`,
                          background: mapColor(v, spec.warn, spec.bad),
                        }}
                      />
                    )}
                  </span>
                  <span className="w-14 shrink-0 text-right font-mono text-xs text-white/70 tabular-nums">
                    {v == null ? "–" : `${fmtNum(v)}${spec.unit}`}
                  </span>
                </button>
              )
            })}
          </div>
        ))}
      </div>
      <p className="text-[0.65rem] text-white/35">
        <span style={{ color: MAP_GOOD }}>green</span> under {spec.warn}
        {spec.unit} · <span style={{ color: MAP_WARN }}>amber</span> worth a look ·{" "}
        <span style={{ color: MAP_BAD }}>red</span> at or over {spec.bad}
        {spec.unit} — bar length is relative to the worst joint in this run.
      </p>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* The workbench                                                       */
/* ------------------------------------------------------------------ */

/**
 * The tuning workbench: pick what to run (sine / step / filter noise test /
 * recorded motion), set the numbers inline, hit Run — and the result lands
 * straight on the graphs below. Every parameter is always visible (no
 * advanced fold), and gain fields (kp / kd / kd_host / kd_host_hz) show the
 * selected joint's current config value with a slider seeded there — leave
 * the box empty to run with config. Everything is joint space and built to
 * localize failures visually: a failure map shows both arms' per-joint error
 * at once (click a joint to jump to its graph), each joint gets a commanded
 * vs actual chart (clean vs noisy vs filtered for filter runs) with an
 * error lane underneath at the error's own scale, and the score table
 * colors cells amber/red past the same thresholds. Runs are compared by
 * their headline tracking score in the run list.
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
  const [missing, setMissing] = useState<string[]>([])
  const [motions, setMotions] = useState<TuningMotion[]>([])
  // Effective per-joint config gains (defaults + calibration): the slider
  // baselines and "config N" labels on the gain fields.
  const [gains, setGains] = useState<TuningGains | null>(null)

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

  const refreshGains = useCallback(() => {
    fetchTuningGains()
      .then(({ gains }) => setGains(gains))
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
    refreshGains()
  }, [enabled, refreshRuns, refreshMotions, refreshGains])

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
    // A finished run may have --save'd new calibration values.
    refreshGains()
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
  const tabValues = useMemo(() => values[tab.key] ?? {}, [values, tab.key])

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

  /**
   * The selected joint's current config value for a gain field, or null
   * until an arm and joint are picked (or while gains haven't loaded).
   */
  const configValue = useCallback(
    (f: WbField): number | null => {
      if (!f.gainKey || !gains) return null
      const side = tabValues["arm"]
      const joint = tabValues["joint"]
      if (!side || !joint) return null
      const v = gains[side]?.[joint]?.[f.gainKey]
      return typeof v === "number" && Number.isFinite(v) ? v : null
    },
    [gains, tabValues]
  )

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
  const perJoint = (meta?.metrics as Record<string, unknown> | undefined)?.per_joint as
    | Record<string, Record<string, unknown>>
    | undefined

  // Failure-map click: chart that arm, then scroll to that joint's graph
  // (after the arm switch has re-rendered the chart grid).
  const pickJoint = useCallback((side: string, joint: string) => {
    setArm(side)
    setTimeout(() => {
      document
        .getElementById(`joint-chart-${joint}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" })
    }, 60)
  }, [])

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
          {tab.fields.map((f) => {
            const cfg = configValue(f)
            return (
            <label key={f.key} className="flex flex-col gap-1">
              <span className="text-[0.65rem] text-white/40">
                {f.label}
                {tab.required.includes(f.key) && <span className="text-[#eff483]/70"> *</span>}
                {cfg != null && (
                  <span className="text-white/25"> · config {fmtNum(cfg)}</span>
                )}
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
              ) : f.slider ? (
                (() => {
                  // The slider tracks the typed value (first number of a
                  // sweep) and starts at the joint's config value; dragging
                  // it fills the box, an empty box runs with config.
                  const raw = (tabValues[f.key] ?? "").trim()
                  const first = Number.parseFloat(raw.split(/\s+/)[0] ?? "")
                  const sliderVal = Number.isFinite(first)
                    ? first
                    : (cfg ?? f.slider.min)
                  return (
                    <span className="flex h-8 items-center gap-2">
                      <input
                        type="range"
                        min={f.slider.min}
                        max={f.slider.max}
                        step={f.slider.step}
                        value={sliderVal}
                        onChange={(e) => setValue(f.key, e.target.value)}
                        disabled={runningOurs || busy || (cfg == null && !raw)}
                        title={f.hint}
                        className="w-24 accent-[#eff483] disabled:opacity-40"
                      />
                      <input
                        type="text"
                        inputMode="decimal"
                        value={tabValues[f.key] ?? ""}
                        placeholder={cfg != null ? fmtNum(cfg) : "config"}
                        title={f.hint}
                        onChange={(e) => setValue(f.key, e.target.value)}
                        disabled={runningOurs || busy}
                        className="h-8 w-16 rounded-md border border-white/10 bg-[#1c1c1c] px-2 font-mono text-xs text-white/85 outline-none placeholder:text-white/25 focus:border-[#eff483]/40"
                      />
                    </span>
                  )
                })()
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
            )
          })}
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
            {meta.params.noise ? ` · ${meta.params.noise as string} noise` : ""}
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

      {/* Both arms at a glance: where the tracking error concentrates. */}
      {meta && perJoint && (
        <FailureMap perJoint={perJoint} kind={meta.kind} arm={arm} onPick={pickJoint} />
      )}

      {/* Commanded vs actual position + error lane, one chart per joint. */}
      {run && jointCharts.length > 0 && (
        <div
          className={cn("grid grid-cols-1 gap-4", jointCharts.length > 1 && "xl:grid-cols-2")}
        >
          {jointCharts.map((c) => (
            <RunChart
              key={c.joint}
              id={`joint-chart-${c.joint}`}
              title={c.joint}
              unit="°"
              series={c.series}
              sub={c.sub}
              height={jointCharts.length > 1 ? 260 : 310}
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
                    {scores.cols.map((c) => {
                      const v = scoreValue(row.values, c)
                      return (
                        <td key={c.key} className={cn("py-1 pr-4", scoreClass(v, c))}>
                          {v == null ? "–" : fmtNum(v, c.digits ?? 2)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {legend && (
            <p className="max-w-3xl text-[0.65rem] leading-relaxed text-white/35">
              {legend} Amber cells are worth a look, red cells are failing.
            </p>
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
                      r.params.noise ? `${r.params.noise as string} noise` : null,
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
