import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, RefreshCw, Square, Trash2, X } from "lucide-react"
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
// Compare mode: run A keeps the yellow "actual" color, run B gets blue.
const B_COLOR = "#7fb4e6"
const B_ERROR_COLOR = "rgba(127,180,230,0.8)"
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
  {
    key: "host_kd_q",
    label: "kd_host_q",
    type: "number",
    gainKey: "kd_host_q",
    slider: { min: 0.4, max: 4, step: 0.1 },
    hint:
      "band width = centre/q. 0.8 default is an octave wide and drags the slow " +
      "final approach when the centre sits low (accuracy slips); q 2-3 with the " +
      "centre on the measured ring damps the ring only",
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
      {
        key: "center",
        label: "center (°)",
        type: "number",
        placeholder: "auto",
        hint:
          "joint-frame start angle (0 = rest, empty = joint midpoint) — probe " +
          "under gravity load too, e.g. 45 / -45",
      },
      { key: "freq", label: "freq (Hz)", type: "number", placeholder: "1.0" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "5" },
      {
        key: "rate",
        label: "rate (Hz)",
        type: "number",
        placeholder: "100",
        hint: "command-loop rate — production teleop runs 240; the loop Hz score shows what was actually sustained",
      },
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
      {
        key: "center",
        label: "center (°)",
        type: "number",
        placeholder: "auto",
        hint:
          "joint-frame start angle the step is framed around (0 = rest, " +
          "empty = current position) — probe under gravity load too, e.g. 45 / -45",
      },
      { key: "hold", label: "hold (s)", type: "number", placeholder: "2" },
      {
        key: "rate",
        label: "rate (Hz)",
        type: "number",
        placeholder: "100",
        hint: "command-loop rate — production teleop runs 240; the loop Hz score shows what was actually sustained",
      },
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
    key: "gravity",
    label: "Gravity",
    command: "tune.gravity",
    description:
      "Fit one link's real centre of mass from a friction-cancelled torque " +
      "sweep, correcting the gravity feedforward. This removes the static " +
      "droop a joint shows under load (parked error = unmodeled torque / " +
      "kp) — something no kp/kd tuning fixes cleanly. Run distal→proximal " +
      "(wrist_3 first, shoulder_1 last); check save to write the CoM into " +
      "this robot's calibration.",
    presets: { save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      { key: "joint", label: "joint", type: "select", options: ARM_JOINT_OPTIONS },
      { key: "save", label: "save to calibration", type: "boolean" },
      {
        key: "velocity",
        label: "velocity (°/s)",
        type: "number",
        placeholder: "18",
        hint: "sweep speed — keep ≤25 so shoulder torque telemetry stays clean",
      },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
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
      {
        key: "prefix",
        label: "recording prefix",
        type: "text",
        placeholder: "/tmp/jit17",
        width: "w-52",
      },
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

/**
 * Map a saved run's metadata back onto its launcher tab's form fields, so
 * clicking a run in the list re-arms the launcher with exactly the settings
 * that produced it (rerun as-is, or nudge one knob and rerun). Returns the
 * complete form state for the tab — fields the run didn't set are cleared
 * back to "command default" — or null for kinds without a launcher tab.
 *
 * Angles stored in radians (the filter suite's params) convert to the
 * degrees the form speaks; tune.pid params are already saved in degrees.
 */
function runFormValues(meta: TuningRunMeta): Record<string, string> | null {
  const p = meta.params ?? {}
  const g = meta.gains ?? {}
  const out: Record<string, string> = {}
  const put = (key: string, v: unknown, opts?: { deg?: boolean }) => {
    if (typeof v === "number" && Number.isFinite(v)) {
      const x = opts?.deg ? toDeg(v) : v
      out[key] = String(Math.round(x * 1000) / 1000)
    } else if (typeof v === "string" && v) {
      out[key] = v
    }
  }
  switch (meta.kind) {
    case "sine":
    case "step":
      put("arm", meta.side)
      put("joint", meta.joint)
      put("kp", g.kp)
      put("kd", g.kd)
      put("host_kd", g.kd_host)
      put("host_kd_hz", g.kd_host_hz)
      put("host_kd_q", g.kd_host_q)
      put("amp", p.amp_deg)
      put("center", p.center_deg)
      put("freq", p.freq)
      put("duration", p.duration)
      put("hold", p.hold)
      put("rate", p.rate)
      put("ff", p.ff)
      put("stiffness", p.stiffness)
      put("target_noise", p.target_noise_deg)
      break
    case "filter":
      put("noise", p.noise)
      if (typeof p.source === "string" && p.source !== "sine") out["motion"] = p.source
      put("amp", p.amp, { deg: true })
      put("freq", p.freq)
      put("duration", p.duration)
      put("jitter", p.jitter_rms, { deg: true })
      put("outlier_amp", p.outlier_amp, { deg: true })
      put("outlier_rate", p.outlier_rate)
      put("stall_ms", p.stall_ms)
      put("stall_rate", p.stall_rate)
      put("ik_churn", p.ik_churn, { deg: true })
      put("ik_jump_amp", p.ik_jump_amp, { deg: true })
      put("ik_jump_rate", p.ik_jump_rate)
      put("cutoff", p.cutoff)
      put("seed", p.seed)
      break
    case "motion": {
      put("motion", p.motion)
      put("stiffness", p.stiffness)
      const overrides = Object.entries(g)
        .map(([k, v]) => `${k}=${v}`)
        .join(" ")
      if (overrides) out["gain"] = overrides
      break
    }
    case "gravity":
      put("arm", meta.side)
      put("joint", meta.joint)
      put("velocity", p.velocity_deg_s)
      break
    default:
      return null
  }
  put("label", meta.label)
  return out
}

/** Radian series → degrees for display (nulls pass through). */
function degSeries(data: (number | null)[]): (number | null)[] {
  return data.map((v) => (v == null ? null : toDeg(v)))
}

/* ------------------------------------------------------------------ */
/* Live probe stream (@@live lines from tuning/runner.py LiveStream)   */
/* ------------------------------------------------------------------ */

interface LiveProbe {
  mode: string
  joint: string
  t: (number | null)[]
  target: (number | null)[]
  actual: (number | null)[]
}

/**
 * The in-flight probe's samples, parsed from the active session's log lines.
 * The runner prints a `new` marker at each probe start (a gain sweep runs
 * several) and sample batches after it; only the latest probe is charted.
 */
function parseLiveProbe(lines: string[]): LiveProbe | null {
  let probe: LiveProbe | null = null
  for (const l of lines) {
    if (!l.startsWith("@@live ")) continue
    try {
      const msg = JSON.parse(l.slice("@@live ".length)) as {
        new?: { mode: string; joint: string }
        samples?: [number, number, number][]
      }
      if (msg.new) {
        probe = { mode: msg.new.mode, joint: msg.new.joint, t: [], target: [], actual: [] }
      } else if (msg.samples && probe) {
        for (const [t, tgt, act] of msg.samples) {
          probe.t.push(t)
          probe.target.push(tgt)
          probe.actual.push(act)
        }
      }
    } catch {
      // a malformed line (e.g. output interleaving) just skips that batch
    }
  }
  return probe
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
  if (meta.kind === "gravity") {
    const v = num(m.droop_after_deg)
    return v == null ? null : { label: "droop", value: `${fmtNum(v, 3)}°` }
  }
  return null
}

/** One per-joint chart: commanded vs actual position for a single joint. */
interface JointChart {
  joint: string
  series: RunChartSeries[]
  /** Error lane (reference − output, in degrees) under the position plot. */
  sub: RunChartSeries[]
  /** Y unit override (default "°") — gravity sweeps chart torque in Nm. */
  unit?: string
  /** X unit override (default "s") — gravity sweeps chart against angle. */
  xUnit?: string
}

/**
 * The error trace for a chart's lane, in degrees. Position traces overlap
 * whenever tracking is halfway decent — the error at its own scale is where
 * a failure actually shows.
 */
function errorLane(
  t: (number | null)[],
  reference: (number | null)[],
  output: (number | null)[],
  label = "error °",
  color = ERROR_COLOR
): RunChartSeries[] {
  const n = Math.min(reference.length, output.length)
  const err: (number | null)[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const r = reference[i]
    const o = output[i]
    err[i] = r == null || o == null ? null : toDeg(o - r)
  }
  return [{ label, color, x: t, data: err }]
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

/**
 * The torque-vs-angle chart of a gravity sweep: measured (friction-cancelled)
 * torque against the model before and after the CoM fit. X is the joint angle
 * in degrees, Y is Nm. The lane shows the residuals — the "before" trace's
 * shape is exactly the gravity error the fit removes.
 */
function gravityJointChart(run: TuningRunData): JointChart | null {
  const q = run.series.q
  const measured = run.series.measured
  const before = run.series.model_before
  const after = run.series.model_after
  if (!q || !measured || !before || !after) return null
  const x = degSeries(q)
  const lane = (model: (number | null)[], label: string, color: string): RunChartSeries => ({
    label,
    color,
    x,
    data: measured.map((v, i) => {
      const m = model[i]
      return v == null || m == null ? null : v - m
    }),
  })
  return {
    joint: run.meta.joint ?? "joint",
    unit: "Nm",
    xUnit: "°",
    series: [
      { label: "measured", color: ACTUAL_COLOR, x, data: measured },
      { label: "model (CAD)", color: NOISY_COLOR, x, data: before },
      { label: "model (fitted)", color: COMMANDED_COLOR, x, data: after },
    ],
    sub: [
      lane(before, "residual before Nm", NOISY_COLOR),
      lane(after, "residual after Nm", ERROR_COLOR),
    ],
  }
}

/* ------------------------------------------------------------------ */
/* Compare mode: overlay two runs of the same kind                     */
/* ------------------------------------------------------------------ */

/** The run-list headline as a raw comparable number (lower is better). */
function headlineNum(meta: TuningRunMeta): number | null {
  const m = meta.metrics as Record<string, unknown>
  const key =
    meta.kind === "motion"
      ? "mean_rms_err"
      : meta.kind === "filter"
        ? "mean_rms_lagfree"
        : meta.kind === "gravity"
          ? "droop_after_deg"
          : "score"
  const v = m[key]
  return typeof v === "number" && Number.isFinite(v) ? v : null
}

function firstFinite(data: (number | null)[]): number {
  for (const v of data) if (v != null) return v
  return 0
}

function rebased(data: (number | null)[], base: number): (number | null)[] {
  return data.map((v) => (v == null ? null : v - base))
}

/**
 * Overlay charts for two runs of the same kind: the reference plus both
 * runs' outputs per joint, with both error traces in the lane. Each series
 * carries its own time base (the runs may have different durations).
 * Sine/step positions are rebased to each run's starting position so runs
 * probed at different centers still overlay; errors are unaffected.
 */
function compareJointCharts(a: TuningRunData, b: TuningRunData, arm: string | null): JointChart[] {
  const kind = a.meta.kind
  const tA = a.series.t ?? []
  const tB = b.series.t ?? []

  if (kind === "sine" || kind === "step") {
    const cmdA = a.series.target
    const actA = a.series.actual
    const cmdB = b.series.target
    const actB = b.series.actual
    if (!cmdA || !actA || !cmdB || !actB) return []
    const baseA = firstFinite(actA)
    const baseB = firstFinite(actB)
    const joint =
      a.meta.joint === b.meta.joint
        ? (a.meta.joint ?? "joint")
        : `${a.meta.joint ?? "?"} (A) vs ${b.meta.joint ?? "?"} (B)`
    return [
      {
        joint,
        series: [
          {
            label: "commanded (A)",
            color: COMMANDED_COLOR,
            x: tA,
            data: degSeries(rebased(cmdA, baseA)),
          },
          { label: "A actual", color: ACTUAL_COLOR, x: tA, data: degSeries(rebased(actA, baseA)) },
          { label: "B actual", color: B_COLOR, x: tB, data: degSeries(rebased(actB, baseB)) },
        ],
        sub: [
          ...errorLane(tA, cmdA, actA, "A error °"),
          ...errorLane(tB, cmdB, actB, "B error °", B_ERROR_COLOR),
        ],
      },
    ]
  }

  if (kind === "gravity") {
    // Two sweeps of (usually) the same joint: overlay the residual-after
    // traces — the shape either fit failed to remove — against angle.
    const out: JointChart[] = []
    for (const [tag, run, color] of [
      ["A", a, ACTUAL_COLOR],
      ["B", b, B_COLOR],
    ] as const) {
      const q = run.series.q
      const measured = run.series.measured
      const after = run.series.model_after
      if (!q || !measured || !after) continue
      out.push({
        joint: `${tag}: ${run.meta.joint ?? "joint"}`,
        unit: "Nm",
        xUnit: "°",
        series: [
          { label: `measured (${tag})`, color, x: degSeries(q), data: measured },
          {
            label: `model fitted (${tag})`,
            color: COMMANDED_COLOR,
            x: degSeries(q),
            data: after,
          },
        ],
        sub: [
          {
            label: `residual ${tag} Nm`,
            color: tag === "A" ? ERROR_COLOR : B_ERROR_COLOR,
            x: degSeries(q),
            data: measured.map((v, i) => {
              const m = after[i]
              return v == null || m == null ? null : v - m
            }),
          },
        ],
      })
    }
    return out
  }

  // motion / filter: channels matched across the runs by column name.
  const refKey = kind === "motion" ? "target" : "clean"
  const outKey = kind === "motion" ? "actual" : "filtered"
  const refLabel = kind === "motion" ? "commanded" : "clean"
  const colsA = (a.meta.params.columns as string[] | undefined) ?? []
  const colsB = (b.meta.params.columns as string[] | undefined) ?? []
  const idxB = new Map(colsB.map((n, i) => [n, i]))
  const out: JointChart[] = []
  for (let i = 0; i < colsA.length; i++) {
    const name = colsA[i]
    if (arm != null && !name?.startsWith(`${arm}.`)) continue
    const j = idxB.get(name)
    const refA = a.series[`${refKey}/${i}`]
    const outA = a.series[`${outKey}/${i}`]
    const refB = j != null ? b.series[`${refKey}/${j}`] : undefined
    const outB = j != null ? b.series[`${outKey}/${j}`] : undefined
    if (!refA || !outA || !refB || !outB) continue
    let min = Infinity
    let max = -Infinity
    for (const v of refA) {
      if (v == null) continue
      if (v < min) min = v
      if (v > max) max = v
    }
    if (max - min < 0.017) continue
    out.push({
      joint: arm != null ? name.slice(arm.length + 1) : name,
      series: [
        { label: refLabel, color: COMMANDED_COLOR, x: tA, data: degSeries(refA) },
        { label: "A", color: ACTUAL_COLOR, x: tA, data: degSeries(outA) },
        { label: "B", color: B_COLOR, x: tB, data: degSeries(outB) },
      ],
      sub: [
        ...errorLane(tA, refA, outA, "A error °"),
        ...errorLane(tB, refB, outB, "B error °", B_ERROR_COLOR),
      ],
    })
  }
  return out
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
  { key: "hz", label: "loop Hz", digits: 0 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
  { key: "pos_ripple", label: "ripple", digits: 4 },
  { key: "holder_peak_deg", label: "holder wobble °", digits: 2, warn: 0.2, bad: 0.5 },
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

const GRAVITY_COLS: ScoreCol[] = [
  { key: "rms_before", label: "residual before Nm", digits: 3 },
  { key: "rms_after", label: "residual after Nm", digits: 3, warn: 0.15, bad: 0.4 },
  { key: "droop_before_deg", label: "droop before °", digits: 3 },
  { key: "droop_after_deg", label: "droop after °", digits: 3, warn: 0.1, bad: 0.3 },
  { key: "fo", label: "Fo Nm", digits: 3 },
]

const STEP_COLS: ScoreCol[] = [
  { key: "settling_s", label: "settling s", warn: 0.4, bad: 0.8 },
  { key: "overshoot", label: "overshoot °", deg: true, digits: 3, warn: 1, bad: 3 },
  { key: "ss_rms", label: "steady-state RMS °", deg: true, digits: 3, warn: 0.3, bad: 0.8 },
  { key: "ring_hz", label: "ring Hz", digits: 1 },
  { key: "hz", label: "loop Hz", digits: 0 },
  { key: "torque_hf", label: "torque chatter Nm", digits: 3 },
  { key: "holder_peak_deg", label: "holder wobble °", digits: 2, warn: 0.2, bad: 0.5 },
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
    "runs by. loop Hz = the command rate actually sustained — if it sits " +
    "below the requested rate, CAN round trips saturated the loop. torque " +
    "chatter / ripple = high-frequency roughness. holder " +
    "wobble = how far the other joints (held stiff in firmware position " +
    "mode) moved during the test — past ~0.5° the structure was flexing " +
    "and part of the error is not this joint's fault.",
  step:
    "settling = time to stay within 5% of the step. overshoot = travel past " +
    "the target. ring Hz = post-step oscillation frequency, if any. loop Hz " +
    "= the command rate actually sustained — if it sits below the requested " +
    "rate, CAN round trips saturated the loop. holder " +
    "wobble = how far the other joints (held stiff in firmware position " +
    "mode) moved — past ~0.5° the structure was flexing and part of the " +
    "ring came from a neighbour. score folds settling, overshoot, and " +
    "steady-state error — lower is better.",
  gravity:
    "residual = friction-cancelled measured torque minus the gravity model, " +
    "shape only (before) and everything (after the CoM fit, including the " +
    "refit Fo). droop = the parked position error that torque error causes " +
    "through the kp spring at this joint's config kp — the number the fix " +
    "actually buys you. Fo = friction offset refit against the corrected " +
    "model (saved with the CoM).",
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
  if (meta.kind === "sine" || meta.kind === "step" || meta.kind === "gravity") {
    return {
      cols: meta.kind === "sine" ? SINE_COLS : meta.kind === "step" ? STEP_COLS : GRAVITY_COLS,
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
  const joints = MAP_JOINT_ORDER.filter((j) => sides.some((s) => `${s}.${j}` in perJoint))
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
 * colors cells amber/red past the same thresholds. Tick two runs of the
 * same kind in the run list to compare them: charts overlay both outputs
 * (A yellow, B blue) with both error traces, a verdict line names the
 * better run by its headline score, and the score table pairs every metric
 * with the better value highlighted.
 */
export function TuningWorkbench({
  enabled,
  commands,
  activeCommand,
  busy,
  disabled,
  liveLines = [],
  onLaunch,
  onStop,
}: {
  enabled: boolean
  commands: CommandSpec[]
  /** Command id of the diagnostics run in flight, if any. */
  activeCommand: string | null
  busy: boolean
  disabled: boolean
  /** The active session's streamed log lines (live probe samples ride them). */
  liveLines?: string[]
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
  // Compare mode: tick two runs of the same kind and the detail area
  // overlays them (A = first ticked, yellow; B = blue). Ticking a third
  // swaps out the older pick; unticking drops back to the single-run view.
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [compareData, setCompareData] = useState<Record<string, TuningRunData>>({})
  // Which arm's joints to chart for a motion run — mirrors the live
  // telemetry arm toggle (and starts from the same remembered choice).
  const [arm, setArm] = useState<string>(() => localStorage.getItem("axolDiagArm") ?? "left")

  const tuningCommandIds = useMemo(() => new Set(TABS.map((t) => t.command)), [])
  const runningOurs = activeCommand != null && tuningCommandIds.has(activeCommand)
  const runningThisTab = activeCommand === tab.command
  // The in-flight probe, charted live as its samples stream in over the
  // session log (sine/step only — the other runs have no single trace).
  const live = useMemo(
    () => (runningOurs ? parseLiveProbe(liveLines) : null),
    [runningOurs, liveLines]
  )

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

  // Clicking a run (vs programmatic selection after a launch) also re-arms
  // the launcher: switch to the run's tab and replace that tab's form with
  // the settings that produced it, ready to rerun or nudge one knob.
  const openRun = useCallback(
    (meta: TuningRunMeta) => {
      select(meta.id)
      const form = runFormValues(meta)
      if (!form) return
      setTabKey(meta.kind)
      setValues((prev) => ({ ...prev, [meta.kind]: form }))
    },
    [select]
  )

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

  const toggleCompare = useCallback((id: string) => {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev.slice(-1), id]
    )
  }, [])

  // Fetch full run data for compare picks (cached; picks change rarely).
  useEffect(() => {
    let active = true
    for (const id of compareIds) {
      if (compareData[id]) continue
      fetchTuningRun(id)
        .then((r) => {
          if (active) setCompareData((prev) => ({ ...prev, [id]: r }))
        })
        .catch((e) => toast.error(String(e)))
    }
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- compareData is the cache being filled; toast is stable
  }, [compareIds])

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

  // Sine and step probe the same joint with the same gains, so their shared
  // fields (arm, joint, kp/kd/kd_host/…, amp, rate, …) behave as one set:
  // switching between the two tabs carries the current values across —
  // including cleared boxes — while tab-specific fields (freq/duration vs
  // hold) keep their own state.
  const switchTab = useCallback(
    (next: string) => {
      const prev = tabKey
      const paired = new Set(["sine", "step"])
      if (prev !== next && paired.has(prev) && paired.has(next)) {
        const prevTab = TABS.find((t) => t.key === prev)
        const nextTab = TABS.find((t) => t.key === next)
        if (prevTab && nextTab) {
          const nextKeys = new Set(nextTab.fields.map((f) => f.key))
          setValues((v) => {
            const src = v[prev] ?? {}
            const dst = { ...(v[next] ?? {}) }
            for (const f of prevTab.fields) {
              if (!nextKeys.has(f.key)) continue
              if (src[f.key] !== undefined) dst[f.key] = src[f.key]
              else delete dst[f.key]
            }
            return { ...v, [next]: dst }
          })
        }
      }
      setTabKey(next)
    },
    [tabKey]
  )

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
    const single = run.meta.kind === "gravity" ? gravityJointChart(run) : pidJointChart(run)
    return single ? [single] : []
  }, [run, arm, armed])
  const scores = meta ? scoreRows(meta, armed ? arm : null) : null
  const legend = meta ? SCORE_LEGEND[meta.kind] : null
  const perJoint = (meta?.metrics as Record<string, unknown> | undefined)?.per_joint as
    Record<string, Record<string, unknown>> | undefined

  /* --- compare mode ------------------------------------------------ */
  const comparing = compareIds.length === 2
  const cmpA = compareData[compareIds[0] ?? ""] ?? null
  const cmpB = compareData[compareIds[1] ?? ""] ?? null
  const cmpReady = comparing && cmpA != null && cmpB != null
  // The kind of the first pick gates the other rows' checkboxes: comparing
  // a step against a filter run has no meaning.
  const compareKind =
    compareIds.length > 0 ? (runs.find((r) => r.id === compareIds[0])?.kind ?? null) : null
  const cmpArms = useMemo(
    () => (cmpReady ? Array.from(new Set([...runArms(cmpA), ...runArms(cmpB)])) : []),
    [cmpReady, cmpA, cmpB]
  )
  const cmpArm = cmpArms.length > 0 ? (cmpArms.includes(arm) ? arm : cmpArms[0]) : null
  const cmpCharts = useMemo(
    () => (cmpReady ? compareJointCharts(cmpA, cmpB, cmpArm) : []),
    [cmpReady, cmpA, cmpB, cmpArm]
  )
  const cmpVerdict = useMemo(() => {
    if (!cmpReady) return null
    const va = headlineNum(cmpA.meta)
    const vb = headlineNum(cmpB.meta)
    if (va == null || vb == null) return null
    const label = headline(cmpA.meta)?.label ?? "score"
    const kind = cmpA.meta.kind
    const fmt = (v: number) =>
      kind === "sine" || kind === "step"
        ? fmtNum(v, 3)
        : kind === "gravity"
          ? `${fmtNum(v, 3)}°`
          : `${fmtNum(toDeg(v))}°`
    if (va === vb)
      return { better: null as string | null, text: `${label}: dead even at ${fmt(va)}` }
    const better = va < vb ? "A" : "B"
    const pct = Math.round((1 - Math.min(va, vb) / Math.max(va, vb)) * 100)
    return {
      better,
      text: `${label} ${fmt(va)} (A) vs ${fmt(vb)} (B) — ${better} is ${pct}% better`,
    }
  }, [cmpReady, cmpA, cmpB])
  const cmpScores = useMemo(() => {
    if (!cmpReady) return null
    const sa = scoreRows(cmpA.meta, cmpArm)
    const sb = scoreRows(cmpB.meta, cmpArm)
    if (!sa && !sb) return null
    const cols = (sa ?? sb)!.cols
    const mapA = new Map((sa?.rows ?? []).map((r) => [r.joint, r.values]))
    const mapB = new Map((sb?.rows ?? []).map((r) => [r.joint, r.values]))
    const joints = Array.from(new Set([...mapA.keys(), ...mapB.keys()]))
    return { cols, joints, mapA, mapB }
  }, [cmpReady, cmpA, cmpB, cmpArm])

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
        setCompareIds((prev) => prev.filter((x) => x !== id))
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
      setCompareIds([])
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
                  switchTab(t.key)
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
                  {cfg != null && <span className="text-white/25"> · config {fmtNum(cfg)}</span>}
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
                    <option value="">{tab.required.includes(f.key) ? "select…" : "default"}</option>
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
                    const sliderVal = Number.isFinite(first) ? first : (cfg ?? f.slider.min)
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
            {live && live.t.length > 1
              ? "Running — tracking live; the full-resolution charts and scores land below when it finishes."
              : "Running — the charts below update when it finishes."}
          </p>
        )}
      </Card>

      {/* Live view of the probe in flight: commanded vs actual streamed from
          the runner (decimated ~25 Hz), full-resolution artifact follows. */}
      {runningOurs && live && live.t.length > 1 && (
        <RunChart
          title={`live · ${live.joint} — ${live.mode}`}
          unit="°"
          series={[
            { label: "commanded", color: COMMANDED_COLOR, x: live.t, data: degSeries(live.target) },
            { label: "actual", color: ACTUAL_COLOR, x: live.t, data: degSeries(live.actual) },
          ]}
          sub={errorLane(live.t, live.target, live.actual)}
          height={240}
        />
      )}

      {/* Compare mode: the two ticked runs, overlaid per joint. */}
      {comparing && (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="neutral">compare</Badge>
            {compareIds.map((id, i) => {
              const r = runs.find((x) => x.id === id)
              if (!r) return null
              const head = headline(r)
              return (
                <span
                  key={id}
                  className="flex flex-wrap items-center gap-x-2 gap-y-0.5 rounded-md border border-white/10 bg-white/[0.03] px-2 py-1"
                >
                  <span
                    className="size-2 shrink-0 rounded-full"
                    style={{ background: i === 0 ? ACTUAL_COLOR : B_COLOR }}
                  />
                  <span className="font-semibold text-white/75">{i === 0 ? "A" : "B"}</span>
                  <span className="text-white/60">
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
                  <span className="text-white/35">{fmtWhen(r.startedAt)}</span>
                </span>
              )
            })}
            {cmpArms.length > 1 && (
              <span className="flex overflow-hidden rounded-md border border-white/10">
                {cmpArms.map((a) => (
                  <button
                    key={a}
                    type="button"
                    onClick={() => setArm(a)}
                    className={cn(
                      "px-3 py-1 text-xs capitalize transition-colors",
                      cmpArm === a
                        ? "bg-[#eff483]/15 text-[#eff483]"
                        : "text-white/50 hover:bg-white/[0.05]"
                    )}
                  >
                    {a} arm
                  </button>
                ))}
              </span>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto text-white/50"
              onClick={() => setCompareIds([])}
            >
              <X /> Exit compare
            </Button>
          </div>

          {cmpVerdict && (
            <p className="text-xs">
              <span
                className="rounded-md bg-white/[0.04] px-2.5 py-1.5 font-medium"
                style={{
                  color:
                    cmpVerdict.better === "B"
                      ? B_COLOR
                      : cmpVerdict.better === "A"
                        ? ACTUAL_COLOR
                        : "rgba(255,255,255,0.7)",
                }}
              >
                {cmpVerdict.text}
              </span>
            </p>
          )}

          {!cmpReady && <p className="text-xs text-white/40">Loading both runs…</p>}
          {cmpReady && cmpCharts.length > 0 && (
            <div className={cn("grid grid-cols-1 gap-4", cmpCharts.length > 1 && "xl:grid-cols-2")}>
              {cmpCharts.map((c) => (
                <RunChart
                  key={c.joint}
                  id={`cmp-chart-${c.joint}`}
                  title={c.joint}
                  unit={c.unit ?? "°"}
                  xUnit={c.xUnit}
                  series={c.series}
                  sub={c.sub}
                  height={cmpCharts.length > 1 ? 260 : 310}
                />
              ))}
            </div>
          )}
          {cmpReady && cmpCharts.length === 0 && (
            <p className="text-xs text-white/40">
              These two runs share no moving joints to overlay.
            </p>
          )}

          {cmpScores && (
            <Card className="gap-3 p-4">
              <h3 className="font-heading text-sm font-semibold">
                Scores — A vs B{cmpArm ? ` — ${cmpArm} arm` : ""}
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-white/40">
                      <th rowSpan={2} className="py-1 pr-4 align-bottom font-normal">
                        joint
                      </th>
                      {cmpScores.cols.map((c) => (
                        <th key={c.key} colSpan={2} className="py-1 pr-4 font-normal">
                          {c.label}
                        </th>
                      ))}
                    </tr>
                    <tr className="text-left">
                      {cmpScores.cols.map((c) => (
                        <Fragment key={c.key}>
                          <th className="py-0.5 pr-2 font-normal" style={{ color: ACTUAL_COLOR }}>
                            A
                          </th>
                          <th className="py-0.5 pr-4 font-normal" style={{ color: B_COLOR }}>
                            B
                          </th>
                        </Fragment>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {cmpScores.joints.map((joint) => (
                      <tr key={joint} className="border-t border-white/[0.06]">
                        <td className="py-1 pr-4 font-sans text-white/55">{joint}</td>
                        {cmpScores.cols.map((c) => {
                          const rowA = cmpScores.mapA.get(joint)
                          const rowB = cmpScores.mapB.get(joint)
                          const va = rowA ? scoreValue(rowA, c) : null
                          const vb = rowB ? scoreValue(rowB, c) : null
                          // Lower is better on every ranked column; ring_hz
                          // is descriptive, not a quality score.
                          const ranked =
                            c.key !== "ring_hz" && va != null && vb != null && va !== vb
                          return (
                            <Fragment key={c.key}>
                              <td
                                className={cn(
                                  "py-1 pr-2",
                                  ranked && va! < vb!
                                    ? "font-semibold text-emerald-300"
                                    : "text-white/70"
                                )}
                              >
                                {va == null ? "–" : fmtNum(va, c.digits ?? 2)}
                              </td>
                              <td
                                className={cn(
                                  "py-1 pr-4",
                                  ranked && vb! < va!
                                    ? "font-semibold text-emerald-300"
                                    : "text-white/70"
                                )}
                              >
                                {vb == null ? "–" : fmtNum(vb, c.digits ?? 2)}
                              </td>
                            </Fragment>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="max-w-3xl text-[0.65rem] leading-relaxed text-white/35">
                Green marks the better (lower) value of each pair. Sine/step charts are rebased to
                each run's starting position so runs probed at different centers still overlay; the
                error lanes are unaffected.
              </p>
            </Card>
          )}
        </>
      )}

      {/* Selected run: what it is, arm tabs, per-joint graphs, scores. */}
      {!comparing && meta && (
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
      {!comparing && meta && perJoint && (
        <FailureMap perJoint={perJoint} kind={meta.kind} arm={arm} onPick={pickJoint} />
      )}

      {/* Commanded vs actual position + error lane, one chart per joint. */}
      {!comparing && run && jointCharts.length > 0 && (
        <div className={cn("grid grid-cols-1 gap-4", jointCharts.length > 1 && "xl:grid-cols-2")}>
          {jointCharts.map((c) => (
            <RunChart
              key={c.joint}
              id={`joint-chart-${c.joint}`}
              title={c.joint}
              unit={c.unit ?? "°"}
              xUnit={c.xUnit}
              series={c.series}
              sub={c.sub}
              height={jointCharts.length > 1 ? 260 : 310}
            />
          ))}
        </div>
      )}
      {!comparing && run && jointCharts.length === 0 && (
        <p className="text-xs text-white/40">
          No joint moved more than 1° in this run — nothing to chart.
        </p>
      )}

      {/* Per-joint scores under the graphs. */}
      {!comparing && scores && (
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
              const cmpIdx = compareIds.indexOf(r.id)
              const cmpBlocked = compareKind != null && r.kind !== compareKind && cmpIdx < 0
              return (
                <div
                  key={r.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => (selected ? select(null) : openRun(r))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      if (selected) select(null)
                      else openRun(r)
                    }
                  }}
                  className={cn(
                    "flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
                    selected ? "bg-[#eff483]/10 ring-1 ring-[#eff483]/30" : "hover:bg-white/[0.04]"
                  )}
                >
                  <input
                    type="checkbox"
                    checked={cmpIdx >= 0}
                    disabled={cmpBlocked}
                    title={
                      cmpBlocked
                        ? `compare needs another ${compareKind} run`
                        : "tick two runs to compare them"
                    }
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => toggleCompare(r.id)}
                    className="accent-[#eff483] disabled:opacity-30"
                    style={
                      cmpIdx >= 0
                        ? { accentColor: cmpIdx === 0 ? ACTUAL_COLOR : B_COLOR }
                        : undefined
                    }
                  />
                  {cmpIdx >= 0 && (
                    <span
                      className="font-semibold"
                      style={{ color: cmpIdx === 0 ? ACTUAL_COLOR : B_COLOR }}
                    >
                      {cmpIdx === 0 ? "A" : "B"}
                    </span>
                  )}
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
