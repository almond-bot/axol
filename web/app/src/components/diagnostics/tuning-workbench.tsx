import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, RefreshCw, Square, Trash2, X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"
import { type FirmwareVendor, shownForJoint } from "@/lib/firmware-loop"
import { RunChart, type RunChartSeries } from "@/components/diagnostics/run-chart"
import type { CommandSpec, FormValue } from "@/lib/supervisor"
import {
  clearTuningRuns,
  deleteTuningRun,
  fetchTuningGains,
  fetchTuningMotions,
  fetchTuningRecordings,
  fetchTuningRun,
  fetchTuningRuns,
  type TuningGains,
  type TuningMotion,
  type TuningRecording,
  type TuningRunData,
  type TuningRunMeta,
  type TuningWireModes,
} from "@/lib/tuning"
import { fetchMotorDetails } from "@/lib/telemetry"
import {
  MYACTUATOR_JOINTS,
  SIDES,
  effectiveWireMode,
  parseA4Tokens,
  toggleA4Token,
} from "@/lib/wire-mode"

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
// offline-analysis artifacts) is hidden rather than half-rendered. "filter"
// artifacts (the retired offline-only tab) still render for old runs.
const KNOWN_KINDS = new Set(["sine", "step", "motion", "gravity", "filter", "build", "kinematics"])

/* ------------------------------------------------------------------ */
/* Inline launcher: what to run, its parameters, and the Run button   */
/* ------------------------------------------------------------------ */

interface WbField {
  key: string
  label: string
  type: "number" | "text" | "select" | "boolean" | "overrides" | "pose" | "wire"
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
  /**
   * Key into the selected motor's *firmware* loop gains (`position_kp`,
   * `speed_kp`, …), read live from the motor over the idle link: the field
   * shows that value as its baseline and an empty box runs with it.
   */
  fwGainKey?: string
  /** Render a slider next to the value box, over this range. */
  slider?: { min: number; max: number; step: number }
  /**
   * The motor vendors whose firmware loop has this knob; the field hides for
   * a joint of any other vendor (see `lib/firmware-loop`). Unset = always.
   */
  vendors?: readonly FirmwareVendor[]
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

/**
 * The firmware loop gains of the Firmware-loop tab. Each shows the selected
 * motor's *live* value ("motor N", read over the idle link when arm and joint
 * are picked); an empty box runs with the motor's value. Plain number boxes,
 * no sliders: the two vendors' gains live on different scales (a MyActuator
 * position_kp near 1, a Damiao KP_APR in the hundreds), so no one range fits.
 * Fields tagged with `vendors` show only for a joint on that vendor's motor —
 * the Damiao wrists have no position D, no exposed current loop.
 */
const FW_GAIN_FIELDS: WbField[] = [
  {
    key: "position_kp",
    label: "position_kp",
    type: "text",
    fwGainKey: "position_kp",
    hint:
      "position loop P — lag ∝ 1/kp. MyActuator: config 1.0 (elbow 1.4), stock 0.008 " +
      "on the X8 shoulders. Damiao KP_APR: config 400",
  },
  {
    key: "position_ki",
    label: "position_ki",
    type: "text",
    fwGainKey: "position_ki",
    hint: "position loop I (Damiao KI_APR)",
  },
  {
    key: "position_kd",
    label: "position_kd",
    type: "text",
    fwGainKey: "position_kd",
    vendors: ["myactuator"],
    hint: "position loop D — measured inert in the 0xA4 loop on the X8-P20",
  },
  {
    key: "speed_kp",
    label: "speed_kp",
    type: "text",
    fwGainKey: "speed_kp",
    hint:
      "speed loop P (Damiao KP_ASR) — on MyActuator the only damping term and the " +
      "buzz knob; 0.1 vibrated on shoulder_1 (stock 0.03)",
  },
  {
    key: "speed_ki",
    label: "speed_ki",
    type: "text",
    fwGainKey: "speed_ki",
    hint: "speed loop I (Damiao KI_ASR) — what pushes through stiction",
  },
  {
    key: "current_kp",
    label: "current_kp",
    type: "text",
    fwGainKey: "current_kp",
    vendors: ["myactuator"],
    hint: "current loop P — leave unless the vendor says otherwise",
  },
  {
    key: "current_ki",
    label: "current_ki",
    type: "text",
    fwGainKey: "current_ki",
    vendors: ["myactuator"],
    hint: "current loop I",
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
      {
        key: "pose",
        label: "pose — hold other joints (°)",
        type: "pose",
        hint:
          "hold other joints at an angle during the test — reflected inertia " +
          "and gravity load change with pose, so probe the worst case; " +
          "ramped back afterwards",
      },
      { key: "freq", label: "freq (Hz)", type: "number", placeholder: "1.0" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "5" },
      {
        key: "rate",
        label: "rate (Hz)",
        type: "number",
        placeholder: "240",
        hint: "command-loop rate — 240 is the production control rate; the loop Hz score shows what was actually sustained",
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
      {
        key: "pose",
        label: "pose — hold other joints (°)",
        type: "pose",
        hint:
          "hold other joints at an angle during the test — reflected inertia " +
          "and gravity load change with pose, so probe the worst case; " +
          "ramped back afterwards",
      },
      { key: "hold", label: "hold (s)", type: "number", placeholder: "2" },
      {
        key: "rate",
        label: "rate (Hz)",
        type: "number",
        placeholder: "240",
        hint: "command-loop rate — 240 is the production control rate; the loop Hz score shows what was actually sustained",
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
    key: "a4",
    label: "Firmware loop",
    command: "tune.a4",
    description:
      "Tune a joint's own firmware position loop — 0xA4 on the MyActuator joints " +
      "(wire_mode a4), position-velocity on the Damiao wrists (pv) — with a sine or " +
      "a constant-speed triangle; the gain boxes follow the joint's motor. Firmware " +
      "gains are written to RAM for the run and restored afterwards (persist " +
      "writes ROM, keep leaves them); planner acceleration must be 0 for the " +
      "joint to follow a stream. A buzz guard restores the previous gains on " +
      "any high-frequency motion. Compare runs on velocity ripple (MIT " +
      "stick-slip ≈ 0.8, smooth < 0.2), stuck windows, lag and the 1–4 Hz band. " +
      "The joint holds stiffly and does not yield to a hand: clear the space.",
    presets: { save_run: true },
    fields: [
      { key: "arm", label: "arm", type: "select", options: ["left", "right"] },
      {
        key: "joint",
        label: "joint",
        type: "select",
        options: ARM_JOINT_OPTIONS,
      },
      { key: "mode", label: "wave", type: "select", options: ["triangle", "sine"] },
      {
        key: "center",
        label: "center (°)",
        type: "number",
        placeholder: "mid",
        hint: "joint-frame centre (0 = rest); probe under gravity load, e.g. -35 on shoulder_1",
      },
      { key: "amp", label: "half-travel (°)", type: "number", placeholder: "10" },
      {
        key: "pose",
        label: "pose — hold other joints (°)",
        type: "pose",
        hint:
          "hold other joints at an angle during the run (overrides the sweep's own " +
          "clearance pose for that joint). A firmware loop that is well damped with the " +
          "arm hanging can oscillate with it extended — right shoulder_2 did, held during " +
          "a shoulder_3 sweep — so tune the worst-case pose too; the held joints are " +
          "sampled during the wave and scored",
      },
      { key: "speed", label: "triangle speed (°/s)", type: "number", placeholder: "3" },
      { key: "freq", label: "sine freq (Hz)", type: "number", placeholder: "0.3" },
      { key: "duration", label: "duration (s)", type: "number", placeholder: "12" },
      { key: "rate", label: "rate (Hz)", type: "number", placeholder: "400" },
      { key: "cap", label: "speed cap (°/s)", type: "number", placeholder: "60" },
      {
        key: "cap_track",
        label: "cap tracks speed ×",
        type: "number",
        placeholder: "0",
        hint:
          "0 = fixed cap. With planner accel 60000 a fixed cap lets the planner burst " +
          "through each 200 Hz step at the cap and idle the rest of the tick (4× the " +
          "current spread on the elbow); 1.1–1.2 sets the per-command cap to that " +
          "multiple of the commanded speed so the joint moves continuously",
      },
      {
        key: "cap_floor",
        label: "cap floor (°/s)",
        type: "number",
        placeholder: "1",
        hint: "lowest cap the tracking cap may set, so a stationary target still corrects",
      },
      {
        key: "dm_acc",
        label: "ACC/DEC (rad/s²)",
        type: "number",
        placeholder: "stored",
        vendors: ["damiao"],
        hint:
          "wrist_2 / wrist_3 only: the position-velocity profiler's acceleration (and " +
          "-deceleration), written to the registers for the run and restored afterwards " +
          "unless kept. Found at 2 rad/s² (~115 °/s²), far too slow to follow a stream",
      },
      {
        key: "accel",
        label: "planner accel (dps/s)",
        type: "text",
        fwGainKey: "planner_accel",
        vendors: ["myactuator"],
        width: "w-24",
        hint:
          "shows what the motor stores; 0 = direct PI tracking (required to follow the " +
          "stream). Written for the run and restored afterwards unless kept",
      },
      ...FW_GAIN_FIELDS,
      {
        key: "held_gain",
        label: "held joint gains",
        type: "text",
        width: "w-72",
        placeholder: "shoulder_2.position_kp=0.5 wrist_2.position_kp=200",
        hint:
          "space-separated JOINT.GAIN=VALUE for the joints *held* during the wave, in " +
          "RAM and restored afterwards unless kept. The held loops are what feed a ring " +
          "they all share — the run's power table names the ones putting energy in; " +
          "the gain boxes above set the test joint only. Damiao wrists: position/speed " +
          "kp/ki only",
      },
      { key: "buzz_abort", label: "buzz abort (°)", type: "number", placeholder: "0.3" },
      {
        key: "iq_abort",
        label: "current abort (A)",
        type: "number",
        placeholder: "30",
        hint:
          "a loaded X8 shoulder holds ~10 A of gravity alone at -55°; keep this above the " +
          "pose's static current",
      },
      { key: "persist", label: "persist gains to ROM", type: "boolean" },
      { key: "keep", label: "keep gains + planner after run", type: "boolean" },
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
      "path and score joint-space tracking per joint. Gain overrides apply " +
      "for this run only — run once plain, once with overrides, and compare " +
      "the scores in the run list. To test the teleop filter stack on real " +
      "hardware, inject noise (network jitter/outliers/stalls and/or IK " +
      "churn/jumps, deterministic per seed) and toggle the filter stack: " +
      "the arm physically shows what the filters remove and what they cost " +
      "in lag — the charts overlay the clean reference, the stream actually " +
      "sent, and the measured position.",
    presets: {},
    fields: [
      { key: "motion", label: "motion", type: "select", options: [] },
      {
        key: "arms",
        label: "arms",
        type: "select",
        options: ["both", "left", "right"],
        placeholder: "both",
        hint:
          "which arm(s) to bring up and drive; the other arm's channel is left " +
          "untouched, so a single-arm run does not need the other arm powered",
      },
      {
        key: "controller",
        label: "controller",
        type: "select",
        options: ["impedance", "position"],
        placeholder: "impedance",
        width: "w-40",
        hint:
          "impedance (240 Hz) is the production MIT frame: host gravity, " +
          "friction, inertia and damping feed-forward around the firmware PD, " +
          "compliant. position (400 Hz) hands every joint to its motor's own " +
          "position loop — MyActuator 0xA4, Damiao position-velocity, the gains " +
          "on the Firmware-loop tab — streamed at 400 Hz, where the loop's " +
          "target staircase is gone: stiff, no host feed-forward, NaN torque " +
          "on the MyActuator joints (contact watchdog blind there). Same " +
          "motion, same scoring, so the two controllers compare directly.",
      },
      { key: "stiffness", label: "stiffness s", type: "number", placeholder: "1" },
      {
        key: "loop_hz",
        label: "core loop (Hz)",
        type: "number",
        placeholder: "auto",
        hint:
          "realtime-core tick rate override; auto follows the wire modes (240 all " +
          "impedance, 400 all firmware loops, 480 mixed — impedance joints on " +
          "alternate ticks). With any arm joint on impedance only 240 or 480 is " +
          "accepted: impedance runs at 240 Hz only",
      },
      {
        key: "record",
        label: "record",
        type: "text",
        width: "w-32",
        placeholder: "prefix",
        hint:
          "flight-recorder prefix: measured joints to PREFIX_meas.npz and the " +
          "realtime core's per-tick trace to PREFIX_rt.npz in the recordings " +
          "directory, for diag.teleop-jitter or offline analysis",
      },
      {
        key: "hold",
        label: "hold joints steady",
        type: "text",
        width: "w-56",
        placeholder: "right.elbow right.wrist_2=10",
        hint:
          "space-separated SIDE.JOINT[=DEG]: held at the motion's start angle (or the " +
          "given one) instead of following it, same controller and gains, scored as " +
          "parked. Only the approach is collision-checked — watch the first pass",
      },
      {
        key: "repeat",
        label: "repeat",
        type: "number",
        placeholder: "1",
        hint:
          "replay the motion this many times back to back (0 = until stopped), each " +
          "pass scored and saved as its own run [k/N] — for soak runs and catching an " +
          "intermittent buzz",
      },
      { key: "gain", label: "gains — edit a cell to override it for this run", type: "overrides" },
      {
        key: "a4",
        label: "controller per joint — click a cell to put that joint on the firmware loop",
        type: "wire",
        hint:
          "inside the impedance controller, single MyActuator joints can go on " +
          "their 0xA4 firmware loop for this run only (--a4 side.joint), the rest " +
          "staying on impedance — no compliance, no host feed-forward and NaN " +
          "torque telemetry on that joint. Everything else about the replay is " +
          "unchanged, so runs compare directly. The Damiao wrists' firmware loop " +
          "(position-velocity) comes with the position controller above, which " +
          "puts every joint on its firmware loop at 400 Hz. A joint already " +
          "configured wire_mode a4 is pinned.",
      },
      {
        key: "ik",
        label: "run as IK",
        type: "boolean",
        hint:
          "drive the run through the IK solver: the motion's cartesian " +
          "end-effector path (FK of the reference, with elbow hints) is " +
          "re-solved to joints like teleop's pose→joints loop and the arms " +
          "execute the solver's output — still scored against the clean " +
          "reference, so IK reconstruction error and tracking show together",
      },
      {
        key: "noise",
        label: "inject noise",
        type: "select",
        options: ["none", "network", "ik", "combined"],
        hint:
          "corrupt the motion before streaming, at each source's real " +
          "pipeline entry point: network = jitter/outliers/stalls, ik = " +
          "solver churn/jumps — same seed, same corrupted stream",
      },
      {
        key: "filter",
        label: "filter stack",
        type: "boolean",
        hint:
          "replay the (possibly corrupted) stream through the production " +
          "teleop filters (pose low-pass → EMA → trapezoid) before sending " +
          "— off streams it raw, so noise hits the arm unsoftened",
      },
      { key: "seed", label: "seed", type: "number", placeholder: "0" },
      {
        key: "torque_threshold",
        label: "contact Nm",
        type: "number",
        placeholder: "8",
        hint:
          "contact watchdog: a sustained joint torque residual (measured minus " +
          "modeled gravity) above this aborts playback — the arm is pushing on " +
          "something that isn't in the plan; 0 disables",
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
    key: "factory",
    label: "Factory",
    command: "tune.factory",
    description:
      "Full-robot factory calibration: friction + gravity for all 14 joints " +
      "(both arms, distal→proximal) in one run. Every fit is saved to this " +
      "robot's calibration as it lands, and the whole document is uploaded " +
      "to the cloud keyed by the hub adapter serial — when Supabase " +
      "credentials are configured (AXOL_SUPABASE_URL / AXOL_SUPABASE_KEY); " +
      "without them the run calibrates locally only. Expect it to take a " +
      "while — 7 joints per arm, several sweep velocities each.",
    presets: {},
    fields: [
      { key: "arms", label: "arms", type: "select", options: ["both", "left", "right"] },
      {
        key: "velocities",
        label: "velocities (°/s)",
        type: "text",
        placeholder: "7.2 18 36 54 72",
        width: "w-44",
        hint: "fewer velocities = quicker run, coarser friction fit",
      },
      {
        key: "hub_serial",
        label: "hub serial",
        type: "text",
        placeholder: "auto-detect",
        width: "w-52",
        hint: "robot id for the cloud upload — leave empty to use the attached hub adapter's",
      },
    ],
    required: [],
    drivesMotors: true,
  },
  {
    key: "build",
    label: "Build motion",
    command: "motion.build",
    description:
      "Turn a recorded session into a reference motion: clip to the " +
      "engaged span (teleop) or trim the still ends (gravity comp), " +
      "resample, smooth, and project through the collision solver. The " +
      "motion saves into the package's committed motions directory and " +
      "becomes selectable under Recorded motion — commit it to git to run " +
      "it on other robots. Record first via teleop (axol teleop " +
      "--teleop.record NAME) or by hand-guiding the arms in gravity comp " +
      "(set the recording name on its operation panel, or axol " +
      "gravity-comp --record NAME); bare names land in " +
      "~/.almond/recordings/. Leave the recording picker on newest to " +
      "build from the most recent one.",
    presets: {},
    fields: [
      {
        key: "name",
        label: "motion name",
        type: "text",
        placeholder: "same as recording",
        hint:
          "what the motion is saved (and picked under Recorded motion) as — " +
          "empty names it after the recording",
      },
      {
        key: "prefix",
        label: "recording",
        type: "select",
        placeholder: "newest recording",
        width: "w-56",
        hint:
          "which recording to convert — captured via teleop " +
          "(--teleop.record) or hand-guided in gravity comp (--record); " +
          "leave on newest to build from the most recent one",
      },
      {
        key: "cutoff",
        label: "cutoff (Hz)",
        type: "number",
        placeholder: "6",
        hint:
          "zero-phase low-pass for the smoothing pass — keeps the operator's " +
          "deliberate motion, drops hand tremor and network jitter above it",
      },
      {
        key: "rate",
        label: "rate (Hz)",
        type: "number",
        placeholder: "240",
        hint:
          "the motion's uniform sample rate — replay commands at this rate, " +
          "so 240 matches the production control loop",
      },
      {
        key: "time_scale",
        label: "time scale",
        type: "number",
        placeholder: "1.0",
      },
      { key: "notes", label: "notes", type: "text", placeholder: "provenance", width: "w-40" },
    ],
    required: [],
    drivesMotors: false,
  },
  {
    key: "ik",
    label: "IK",
    command: "diag.offline",
    description:
      "Analyze the IK solver over a teleop recording (offline, no " +
      "hardware): the world EE target the solver was asked to reach vs the " +
      "FK of what it solved, per axis — plus per-tick solve time and " +
      "per-joint churn (a restless null space adds joint motion the hand " +
      "never made). Record a session first with axol teleop --teleop.record " +
      "NAME; leave the recording box empty to analyze the newest one.",
    presets: { suite: "kinematics", save_run: true },
    fields: [
      {
        key: "prefix",
        label: "recording",
        type: "text",
        placeholder: "newest recording",
        width: "w-44",
        hint:
          "which teleop recording to analyze — a bare --teleop.record name, " +
          "or a full path prefix; empty uses the newest in ~/.almond/recordings/",
      },
      { key: "label", label: "label", type: "text", placeholder: "note", width: "w-40" },
    ],
    required: [],
    drivesMotors: false,
  },
]

/** Run kinds → the launcher tab that produces them (for re-arming on click). */
const KIND_TABS: Record<string, string> = {
  sine: "sine",
  step: "step",
  motion: "motion",
  gravity: "gravity",
  build: "build",
  kinematics: "ik",
}

/**
 * A `tune.a4` run: saved as kind `sine` (it shares the sine/triangle charts)
 * but tagged `wire: "a4"` — it belongs to the Firmware-loop tab, carries the
 * firmware gains instead of impedance gains, and is scored on creep
 * smoothness rather than the impedance score.
 */
function isA4Run(meta: TuningRunMeta): boolean {
  return meta.kind === "sine" && meta.params?.wire === "a4"
}

/** The launcher tab a saved run re-arms, or null for kinds without one. */
function runTab(meta: TuningRunMeta): string | null {
  if (isA4Run(meta)) return "a4"
  return KIND_TABS[meta.kind] ?? null
}

/** What to call a run in badges: its kind, except firmware-loop runs. */
function runKindLabel(meta: TuningRunMeta): string {
  return isA4Run(meta) ? "a4" : meta.kind
}

/* ------------------------------------------------------------------ */
/* Gain-override editor (Recorded motion tab)                          */
/* ------------------------------------------------------------------ */

// Matches tune.motion's --gain fields (see _GAIN_FIELDS there).
const OVERRIDE_FIELDS = [
  "kp",
  "kd",
  "kd_host",
  "kd_host_hz",
  "kd_host_q",
  "j_eff",
  "stiction_gain",
  "stiction_load_gain",
  "dither_nm",
  "stribeck_gain",
]

/** Format a config gain for seeding/comparison (trims float32 noise). */
function fmtGain(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return ""
  return String(Number(v.toFixed(3)))
}

/**
 * Per-joint controller picker for tune.motion: one row per MyActuator joint,
 * one cell per arm, each a two-way toggle between the MIT impedance frame
 * and the firmware position loop (`--a4 side.joint`). Cells the robot's
 * config already pins to `wire_mode a4` show as firmware and cannot be
 * switched back — a run can only add `--a4` joints. Serializes to the token
 * string the CLI takes, so the launch path and run re-arming stay generic.
 */
function WireModeEditor({
  value,
  onChange,
  disabled,
  configModes,
}: {
  value: string
  onChange: (v: string) => void
  disabled: boolean
  configModes: TuningWireModes | null
}) {
  const picked = parseA4Tokens(value)
  return (
    <div className="flex flex-col gap-1.5 overflow-x-auto">
      <table className="w-fit border-separate border-spacing-0">
        <thead>
          <tr>
            <th />
            {SIDES.map((side) => (
              <th
                key={side}
                className="px-1 pb-1 text-left text-[0.65rem] font-normal text-white/40"
              >
                {side}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MYACTUATOR_JOINTS.map((joint) => (
            <tr key={joint}>
              <td className="pr-2 text-xs text-white/55">{joint}</td>
              {SIDES.map((side) => {
                const mode = effectiveWireMode(value, configModes, side, joint)
                const pinned =
                  (configModes?.[side]?.[joint] ?? "mit").toLowerCase() === "a4" &&
                  !picked.has(`${side}.${joint}`)
                const dirty = picked.has(`${side}.${joint}`)
                return (
                  <td key={side} className="p-0.5">
                    <button
                      type="button"
                      disabled={disabled || pinned}
                      title={
                        pinned
                          ? "wire_mode a4 in this robot's config — the run cannot put it back on impedance"
                          : mode === "a4"
                            ? "firmware position loop (0xA4) for this run — click for impedance"
                            : "MIT impedance frame — click to run this joint on the firmware loop"
                      }
                      onClick={() => onChange(toggleA4Token(value, side, joint))}
                      className={cn(
                        "h-7 w-24 rounded-md border px-2 text-left font-mono text-[0.7rem] outline-none disabled:cursor-not-allowed",
                        dirty
                          ? "border-[#eff483]/60 bg-[#eff483]/10 text-[#eff483]"
                          : mode === "a4"
                            ? "border-white/10 bg-[#1c1c1c] text-white/55"
                            : "border-white/10 bg-[#1c1c1c] text-white/70 hover:border-white/25"
                      )}
                    >
                      {mode === "a4" ? "firmware" : "impedance"}
                      {pinned && <span className="text-white/30"> · config</span>}
                    </button>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {picked.size > 0 && (
        <div className="flex items-center gap-2 text-[0.65rem] text-white/40">
          <span>
            {picked.size} joint{picked.size === 1 ? "" : "s"} on the firmware loop this run
          </span>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange("")}
            className="text-white/55 underline-offset-2 hover:underline disabled:opacity-40"
          >
            reset
          </button>
        </div>
      )}
    </div>
  )
}

/**
 * Full gain table for tune.motion's per-run overrides: one row per joint,
 * one column per gain field, every cell pre-filled with this robot's
 * current config value (defaults + calibration). Overrides always apply to
 * **both arms**: edit a cell and it is sent as a `joint.field=value` token
 * (highlighted); a cell left at (or retyped to) its config value sends
 * nothing. Serializes to the same token string the CLI takes, so the
 * launch path and run-metadata re-arming are unchanged — sided tokens from
 * a re-armed older run collapse onto the shared cell.
 */
function GainOverrideEditor({
  value,
  onChange,
  disabled,
  gains,
}: {
  value: string
  onChange: (v: string) => void
  disabled: boolean
  gains: TuningGains | null
}) {
  // Config seed per cell. When the two arms' configs disagree the cell
  // seeds empty with a "left|right" placeholder — typing any value there
  // overrides both arms to it.
  const seeds = useMemo(() => {
    const out: Record<string, { text: string; placeholder: string }> = {}
    for (const joint of ARM_JOINT_OPTIONS) {
      for (const field of OVERRIDE_FIELDS) {
        const l = fmtGain(gains?.["left"]?.[joint]?.[field])
        const r = fmtGain(gains?.["right"]?.[joint]?.[field])
        out[`${joint}.${field}`] =
          l === r ? { text: l, placeholder: "" } : { text: "", placeholder: `${l}|${r}` }
      }
    }
    return out
  }, [gains])

  const build = useCallback(
    (tokens: string) => {
      const cells: Record<string, string> = {}
      for (const key of Object.keys(seeds)) cells[key] = seeds[key].text
      for (const tok of tokens.split(/\s+/).filter(Boolean)) {
        const [path = "", v = ""] = tok.split("=")
        const parts = path.split(".")
        const key = parts.length === 3 ? `${parts[1]}.${parts[2]}` : path
        if (key in cells) cells[key] = v
      }
      return cells
    },
    [seeds]
  )

  const serialize = useCallback(
    (cells: Record<string, string>) => {
      const toks: string[] = []
      for (const joint of ARM_JOINT_OPTIONS) {
        for (const field of OVERRIDE_FIELDS) {
          const key = `${joint}.${field}`
          const text = (cells[key] ?? "").trim()
          if (text !== "" && text !== seeds[key].text) toks.push(`${key}=${text}`)
        }
      }
      return toks.join(" ")
    },
    [seeds]
  )

  const [cells, setCells] = useState<Record<string, string>>(() => build(value))
  // Reseed when the config gains arrive/refresh (overrides are re-applied
  // from `value`, so nothing typed is lost).
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reseed on gains load
    setCells(build(value))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reseed only when the seeds change
  }, [seeds])
  // Resync on external changes (re-arm from a clicked run) — while the
  // operator is mid-edit, in-progress cells must survive serialization.
  useEffect(() => {
    setCells((prev) => (serialize(prev) === value ? prev : build(value)))
  }, [value, build, serialize])

  const edit = (key: string, text: string) => {
    const next = { ...cells, [key]: text }
    setCells(next)
    onChange(serialize(next))
  }

  const dirtyCount = value.split(/\s+/).filter(Boolean).length
  return (
    <div className="flex flex-col gap-1.5 overflow-x-auto">
      <table className="w-fit border-separate border-spacing-0">
        <thead>
          <tr>
            <th />
            {OVERRIDE_FIELDS.map((f) => (
              <th key={f} className="px-1 pb-1 text-left text-[0.65rem] font-normal text-white/40">
                {f}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ARM_JOINT_OPTIONS.map((joint) => (
            <tr key={joint}>
              <td className="pr-2 text-xs text-white/55">{joint}</td>
              {OVERRIDE_FIELDS.map((field) => {
                const key = `${joint}.${field}`
                const text = (cells[key] ?? "").trim()
                const dirty = text !== "" && text !== seeds[key].text
                return (
                  <td key={field} className="p-0.5">
                    <input
                      type="text"
                      inputMode="decimal"
                      value={cells[key] ?? ""}
                      placeholder={seeds[key].placeholder}
                      onChange={(e) => edit(key, e.target.value)}
                      onBlur={() => {
                        // An emptied cell means "no override" — snap it back
                        // to the config value so the table never shows blanks.
                        if (text === "" && seeds[key].text !== "") {
                          setCells((prev) => ({ ...prev, [key]: seeds[key].text }))
                        }
                      }}
                      disabled={disabled}
                      className={cn(
                        "h-7 w-16 rounded border bg-[#1c1c1c] px-1.5 font-mono text-xs outline-none placeholder:text-white/25 focus:border-[#eff483]/40",
                        dirty
                          ? "border-[#eff483]/50 text-[#eff483]"
                          : "border-white/10 text-white/60"
                      )}
                    />
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {dirtyCount > 0 && (
        <button
          type="button"
          onClick={() => {
            setCells(build(""))
            onChange("")
          }}
          disabled={disabled}
          className="flex items-center gap-1 self-start text-[0.65rem] text-white/40 transition-colors hover:text-red-300"
        >
          <X className="h-3 w-3" /> reset {dirtyCount} override{dirtyCount > 1 ? "s" : ""}
        </button>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Pose editor (sine/step tabs)                                        */
/* ------------------------------------------------------------------ */

interface PoseRow {
  joint: string
  deg: string
}

/** Parse the CLI's `joint=deg` tokens into editor rows. */
function parsePose(text: string): PoseRow[] {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((tok) => {
      const [joint = "", deg = ""] = tok.split("=")
      return { joint, deg }
    })
}

/** Serialize complete rows back to `joint=deg` tokens (incomplete rows stay editor-local). */
function serializePose(rows: PoseRow[]): string {
  return rows
    .filter((r) => r.joint && r.deg.trim() !== "")
    .map((r) => `${r.joint}=${r.deg.trim()}`)
    .join(" ")
}

/**
 * Structured editor for the sine/step tests' hold pose: rows of joint +
 * angle instead of hand-typed `elbow=90` tokens. Serializes to exactly
 * those tokens (each becomes its own --pose flag), so the launch path and
 * run-metadata re-arming are unchanged. The joint under test is excluded
 * from the options — the CLI rejects posing it.
 */
function PoseEditor({
  value,
  onChange,
  disabled,
  excludeJoint,
}: {
  value: string
  onChange: (v: string) => void
  disabled: boolean
  excludeJoint: string
}) {
  const [rows, setRows] = useState<PoseRow[]>(() => parsePose(value))
  // Resync only on external changes (re-arm from a clicked run) — while the
  // operator is mid-edit, incomplete rows must survive serialization.
  useEffect(() => {
    setRows((prev) => (value === serializePose(prev) ? prev : parsePose(value)))
  }, [value])
  const update = (next: PoseRow[]) => {
    setRows(next)
    onChange(serializePose(next))
  }
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((row, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <select
            value={row.joint}
            onChange={(e) =>
              update(rows.map((r, j) => (j === i ? { ...r, joint: e.target.value } : r)))
            }
            disabled={disabled}
            className="h-8 w-28 rounded-md border border-white/10 bg-[#1c1c1c] px-1.5 text-xs text-white/85 outline-none focus:border-[#eff483]/40"
          >
            <option value="">joint…</option>
            {ARM_JOINT_OPTIONS.filter((o) => o !== excludeJoint || o === row.joint).map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
          <input
            type="text"
            inputMode="decimal"
            value={row.deg}
            placeholder="deg"
            onChange={(e) =>
              update(rows.map((r, j) => (j === i ? { ...r, deg: e.target.value } : r)))
            }
            disabled={disabled}
            className="h-8 w-16 rounded-md border border-white/10 bg-[#1c1c1c] px-2 font-mono text-xs text-white/85 outline-none placeholder:text-white/25 focus:border-[#eff483]/40"
          />
          <button
            type="button"
            onClick={() => update(rows.filter((_, j) => j !== i))}
            disabled={disabled}
            className="text-white/30 transition-colors hover:text-red-300"
            title="remove pose hold"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={() => update([...rows, { joint: "", deg: "" }])}
        disabled={disabled}
        className="h-8 w-fit rounded-md border border-dashed border-white/15 px-2 text-xs text-white/45 transition-colors hover:border-[#eff483]/40 hover:text-[#eff483]"
      >
        + hold joint
      </button>
    </div>
  )
}

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
  if (isA4Run(meta)) {
    put("arm", meta.side)
    put("joint", meta.joint)
    put("mode", p.mode)
    put("center", p.center_deg)
    put("amp", p.amp_deg)
    put("speed", p.speed_dps)
    put("freq", p.freq_hz)
    put("duration", p.duration_s)
    put("rate", p.rate_hz)
    put("cap", p.cap_dps)
    if (typeof p.cap_track === "number" && p.cap_track > 0) put("cap_track", p.cap_track)
    if (typeof p.cap_track === "number" && p.cap_track > 0) put("cap_floor", p.cap_floor_dps)
    if (Array.isArray(p.accel) && typeof p.accel[0] === "number") out["accel"] = String(p.accel[0])
    if (Array.isArray(p.dm_acc) && typeof p.dm_acc[0] === "number")
      out["dm_acc"] = String(p.dm_acc[0])
    if (Array.isArray(p.pose) && p.pose.length > 0) out["pose"] = p.pose.join(" ")
    // Held joints' gains, as run: {joint: {gain: value}} → "joint.gain=value …".
    if (p.held_gains && typeof p.held_gains === "object") {
      const held = Object.entries(p.held_gains as Record<string, Record<string, unknown>>)
        .flatMap(([j, gains]) =>
          Object.entries(gains ?? {})
            .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
            .map(([n, v]) => `${j}.${n}=${fmtFwGain(v)}`)
        )
        .join(" ")
      if (held) out["held_gain"] = held
    }
    for (const k of ["position_kp", "position_ki", "position_kd", "speed_kp", "speed_ki"]) {
      const v = g[k]
      if (typeof v === "number" && Number.isFinite(v)) out[k] = fmtFwGain(v)
    }
    if (p.persist === true) out["persist"] = "true"
    return out
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
      if (Array.isArray(p.pose) && p.pose.length > 0) out["pose"] = p.pose.join(" ")
      put("freq", p.freq)
      put("duration", p.duration)
      put("hold", p.hold)
      put("rate", p.rate)
      put("ff", p.ff)
      put("stiffness", p.stiffness)
      put("target_noise", p.target_noise_deg)
      break
    case "motion": {
      put("motion", p.motion)
      put("controller", p.controller)
      put("stiffness", p.stiffness)
      put("noise", p.noise)
      if (p.ik === true) out["ik"] = "true"
      if (p.filter === true) out["filter"] = "true"
      put("seed", p.seed)
      const overrides = Object.entries(g)
        .map(([k, v]) => `${k}=${v}`)
        .join(" ")
      if (overrides) out["gain"] = overrides
      if (Array.isArray(p.a4) && p.a4.length > 0) {
        out["a4"] = p.a4.filter((t): t is string => typeof t === "string").join(" ")
      }
      if (Array.isArray(p.hold) && p.hold.length > 0) {
        out["hold"] = p.hold.filter((t): t is string => typeof t === "string").join(" ")
      }
      if (p.arms === "left" || p.arms === "right") out["arms"] = p.arms
      break
    }
    case "gravity":
      put("arm", meta.side)
      put("joint", meta.joint)
      put("velocity", p.velocity_deg_s)
      break
    case "build":
      put("name", p.name)
      put("prefix", p.prefix)
      put("rate", p.rate)
      put("cutoff", p.cutoff)
      put("time_scale", p.time_scale)
      put("notes", meta.label)
      break
    case "kinematics":
      put("prefix", p.prefix)
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

/** Firmware loop gains span 0.0001 … 1: four significant digits, no padding. */
function fmtFwGain(v: unknown): string {
  if (v == null || typeof v !== "number" || !Number.isFinite(v)) return "–"
  return String(Number(v.toPrecision(4)))
}

/** The baseline a gain box falls back to: the motor's live value, or config. */
function baselineText(f: WbField, cfg: number | null): string {
  if (cfg == null) return f.fwGainKey ? "motor" : "config"
  return f.fwGainKey ? fmtFwGain(cfg) : fmtNum(cfg)
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
  if (meta.kind === "build") {
    const v = num(m.dev_max_deg)
    return v == null ? null : { label: "max change", value: `${fmtNum(v)}°` }
  }
  if (meta.kind === "kinematics") {
    const ee = m.ee_rms_mm as Record<string, unknown> | undefined
    const vals = ee ? Object.values(ee).filter((v): v is number => typeof v === "number") : []
    if (vals.length === 0) return null
    return { label: "EE RMS", value: `${fmtNum(Math.max(...vals))} mm` }
  }
  return null
}

/** One per-joint chart: commanded vs actual position for a single joint. */
interface JointChart {
  joint: string
  /**
   * Shown after the joint in the chart title: "held" for a joint the run
   * held steady (`tune.motion --hold`), "parked" for one it never moved.
   */
  note?: string
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
/** `side.joint` columns a motion run held steady (`--hold SIDE.JOINT[=DEG]`). */
function heldColumns(run: TuningRunData): Set<string> {
  const hold = run.meta.params.hold
  return new Set(
    (Array.isArray(hold) ? hold : [])
      .filter((h): h is string => typeof h === "string")
      .map((h) => h.split("=")[0] ?? h)
  )
}

/** Whether a commanded series moves less than ~1° (0.017 rad) end to end. */
function isStationary(values: (number | null)[]): boolean {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v == null) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  return max - min < 0.017
}

/**
 * Commanded vs actual per joint for a motion run. Joints that moved come
 * first; joints the run held steady (`--hold`) or never moved follow, noted
 * as such — a parked joint still buzzes or sags, which is worth seeing.
 */
function motionJointCharts(run: TuningRunData, arm: string): JointChart[] {
  const columns = (run.meta.params.columns as string[] | undefined) ?? []
  const t = run.series.t ?? []
  const held = heldColumns(run)
  const out: JointChart[] = []
  const still: JointChart[] = []
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i]
    if (!name?.startsWith(`${arm}.`)) continue
    const commanded = run.series[`target/${i}`]
    const actual = run.series[`actual/${i}`]
    const sent = run.series[`sent/${i}`]
    if (!commanded || !actual || !actual.some((v) => v != null)) continue
    const note = held.has(name) ? "held" : isStationary(commanded) ? "parked" : undefined
    const series: RunChartSeries[] = [
      { label: "commanded", color: COMMANDED_COLOR, x: t, data: degSeries(commanded) },
    ]
    if (sent) {
      // Noise/filter runs: the stream actually sent to the arm, between the
      // clean reference and what the joint measured.
      series.push({ label: "sent", color: NOISY_COLOR, x: t, data: degSeries(sent) })
    }
    series.push({ label: "actual", color: ACTUAL_COLOR, x: t, data: degSeries(actual) })
    ;(note ? still : out).push({
      joint: name.slice(arm.length + 1),
      note,
      series,
      sub: errorLane(t, commanded, actual),
    })
  }
  return [...out, ...still]
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

/** Linear interpolation of a (possibly gappy) series onto another x grid. */
function interpOnto(
  xSrc: (number | null)[],
  ySrc: (number | null)[],
  xDst: (number | null)[]
): (number | null)[] {
  const xs: number[] = []
  const ys: number[] = []
  const n = Math.min(xSrc.length, ySrc.length)
  for (let i = 0; i < n; i++) {
    const x = xSrc[i]
    const y = ySrc[i]
    if (x != null && y != null) {
      xs.push(x)
      ys.push(y)
    }
  }
  if (xs.length < 2) return xDst.map(() => null)
  let j = 0
  return xDst.map((x) => {
    if (x == null) return null
    if (x <= xs[0]) return ys[0]
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1]
    while (j < xs.length - 2 && xs[j + 1] < x) j++
    const f = (x - xs[j]) / (xs[j + 1] - xs[j])
    return ys[j] + f * (ys[j + 1] - ys[j])
  })
}

/**
 * Before/after charts for a motion.build run: the clipped raw command
 * recording vs the built motion (resampled + smoothed + collision-projected),
 * per joint of `arm`. The lane shows built − recorded — exactly what the
 * postprocessing changed.
 */
function buildJointCharts(run: TuningRunData, arm: string): JointChart[] {
  const columns = (run.meta.params.columns as string[] | undefined) ?? []
  const tRaw = run.series.t_raw ?? []
  const tBuilt = run.series.t ?? []
  const out: JointChart[] = []
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i]
    if (!name?.startsWith(`${arm}.`)) continue
    const raw = run.series[`raw/${i}`]
    const built = run.series[`built/${i}`]
    if (!raw || !built) continue
    let min = Infinity
    let max = -Infinity
    for (const v of raw) {
      if (v == null) continue
      if (v < min) min = v
      if (v > max) max = v
    }
    if (max - min < 0.017) continue
    const builtAtRaw = interpOnto(tBuilt, built, tRaw)
    out.push({
      joint: name.slice(arm.length + 1),
      series: [
        { label: "recorded", color: NOISY_COLOR, x: tRaw, data: degSeries(raw) },
        { label: "built", color: ACTUAL_COLOR, x: tBuilt, data: degSeries(built) },
      ],
      sub: errorLane(tRaw, raw, builtAtRaw, "built − recorded °"),
    })
  }
  return out
}

const EE_AXES = ["x", "y", "z"] as const

/** Series in meters → millimeters for display (nulls pass through). */
function mmSeries(data: (number | null)[]): (number | null)[] {
  return data.map((v) => (v == null ? null : v * 1000))
}

/**
 * Charts for an IK (kinematics) run: EE pose set vs actual per axis for one
 * arm — the world target the solver was asked to reach vs the FK of what it
 * solved — plus the per-tick solve time. Everything else the solver did
 * wrong shows in the per-joint churn score table.
 */
function kinematicsCharts(run: TuningRunData, arm: string): JointChart[] {
  const s = arm === "left" ? "l" : "r"
  const t = run.series.t ?? []
  const out: JointChart[] = []
  for (let ax = 0; ax < 3; ax++) {
    const set = run.series[`ee_tgt_${s}/${ax}`]
    const actual = run.series[`ee_fk_${s}/${ax}`]
    if (!set || !actual || !set.some((v) => v != null)) continue
    const setMm = mmSeries(set)
    const actMm = mmSeries(actual)
    out.push({
      joint: `EE ${EE_AXES[ax]}`,
      unit: "mm",
      series: [
        { label: "set (target)", color: COMMANDED_COLOR, x: t, data: setMm },
        { label: "actual (FK)", color: ACTUAL_COLOR, x: t, data: actMm },
      ],
      sub: [
        {
          label: "error mm",
          color: ERROR_COLOR,
          x: t,
          data: setMm.map((v, i) => {
            const a = actMm[i]
            return v == null || a == null ? null : a - v
          }),
        },
      ],
    })
  }
  const solve = run.series.solve_ms
  const solveT = run.series.q_t
  if (solve && solveT && out.length > 0) {
    out.push({
      joint: "solve time",
      unit: "ms",
      series: [{ label: "solve ms", color: ACTUAL_COLOR, x: solveT, data: solve }],
      sub: [],
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
  if (run.meta.kind === "build") {
    return ["left", "right"].filter((arm) => buildJointCharts(run, arm).length > 0)
  }
  if (run.meta.kind === "kinematics") {
    return ["left", "right"].filter((arm) => kinematicsCharts(run, arm).length > 0)
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
  // build / kinematics runs have no overlay story — compare their scores.
  if (kind === "build" || kind === "kinematics") return []
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
  const heldA = kind === "motion" ? heldColumns(a) : new Set<string>()
  const heldB = kind === "motion" ? heldColumns(b) : new Set<string>()
  const out: JointChart[] = []
  const still: JointChart[] = []
  for (let i = 0; i < colsA.length; i++) {
    const name = colsA[i]
    if (arm != null && !name?.startsWith(`${arm}.`)) continue
    const j = idxB.get(name)
    const refA = a.series[`${refKey}/${i}`]
    const outA = a.series[`${outKey}/${i}`]
    const refB = j != null ? b.series[`${refKey}/${j}`] : undefined
    const outB = j != null ? b.series[`${outKey}/${j}`] : undefined
    if (!refA || !outA || !refB || !outB) continue
    // Motion runs keep held / parked joints (after the moving ones); a filter
    // channel that never moves has nothing to compare.
    const stationary = isStationary(refA)
    if (stationary && kind !== "motion") continue
    const note =
      kind !== "motion"
        ? undefined
        : heldA.has(name) || heldB.has(name)
          ? `held (${[heldA.has(name) && "A", heldB.has(name) && "B"].filter(Boolean).join(", ")})`
          : stationary
            ? "parked"
            : undefined
    ;(note ? still : out).push({
      joint: arm != null ? name.slice(arm.length + 1) : name,
      note,
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
  return [...out, ...still]
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
  { key: "buzz", label: "buzz °", deg: true, digits: 3, warn: 0.01, bad: 0.02 },
  { key: "buzz_hz", label: "buzz Hz", digits: 0 },
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

// tune.a4's creep-smoothness scorecard (see a4_metrics): the MIT frame's
// stick-slip sits near 0.8 velocity ripple, smooth is under 0.2.
const A4_COLS: ScoreCol[] = [
  { key: "rms", label: "tracking RMS °", deg: true, digits: 3, warn: 0.5, bad: 2.0 },
  { key: "max", label: "max err °", deg: true, digits: 3, warn: 1.5, bad: 5 },
  { key: "lag_ms", label: "lag ms", digits: 0, warn: 100, bad: 300 },
  { key: "v_ripple", label: "vel ripple", digits: 2, warn: 0.3, bad: 0.8 },
  { key: "stuck_frac", label: "stuck", digits: 2, warn: 0.05, bad: 0.3 },
  { key: "band_1_4", label: "1–4 Hz °", deg: true, digits: 3, warn: 0.1, bad: 0.3 },
  { key: "buzz", label: ">10 Hz buzz °", deg: true, digits: 3, warn: 0.02, bad: 0.1 },
  { key: "iq_mode", label: "3–8 Hz mode A", digits: 2, warn: 0.5, bad: 1.0 },
  { key: "iq_sd", label: "current spread A", digits: 2, warn: 1.3, bad: 1.8 },
  { key: "iq_rms", label: "current RMS A", digits: 2 },
  { key: "iq_max", label: "peak A", digits: 1 },
  { key: "hz", label: "loop Hz", digits: 0 },
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

// motion.build before/after: values are already in display units (degrees).
const BUILD_COLS: ScoreCol[] = [
  { key: "dev_rms_deg", label: "change RMS °", digits: 3 },
  { key: "dev_max_deg", label: "change max °", digits: 2, warn: 3, bad: 8 },
  { key: "peak_vel_raw_dps", label: "peak vel in °/s", digits: 0 },
  { key: "peak_vel_built_dps", label: "peak vel out °/s", digits: 0, warn: 120, bad: 180 },
]

// IK per-joint churn (values already in degrees / Hz).
const KIN_COLS: ScoreCol[] = [
  { key: "churn_deg_min", label: "churn °/min", digits: 0, warn: 200, bad: 600 },
  { key: "mid_band_deg", label: "3–15 Hz °", digits: 3, warn: 0.15, bad: 0.4 },
  { key: "peak_hz", label: "peak Hz", digits: 1 },
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

const A4_LEGEND =
  "firmware position loop (0xA4). tracking RMS / max / lag = how the stream was " +
  "followed. vel ripple = std of measured minus commanded velocity over the " +
  "commanded speed — the MIT frame's stick-slip sits near 0.8, smooth is under 0.2. " +
  "stuck = fraction of the pass with the joint not moving. 1–4 Hz = the stick-slip " +
  "band in the error; >10 Hz buzz = high-frequency position motion. 3–8 Hz mode = " +
  "the position loop's own mode in the current — the shudder felt at speed (a 12 deg/s " +
  "triangle's reversals kick it to ~1.9 A at position_kp 0.7; 0.2 A is quiet); current " +
  "spread = all current variation, the gravity hold removed. Anything above 100 Hz is " +
  "invisible to the 200 Hz stream, so an audible buzz can leave every column clean."

const SCORE_LEGEND: Record<string, string> = {
  motion:
    "tracking RMS = average distance from the commanded joint position " +
    "(lower is better; the run-list number is the all-joint mean). lag = " +
    "command→measurement delay. jitter = 3–15 Hz vibration in the error — " +
    "what the operator feels. ringing ×>1 = the joint oscillates more than " +
    "commanded. torque chatter = cycle-to-cycle torque noise. buzz = " +
    "sustained ≥20 Hz motion — what you hear: healthy joints sit near " +
    "0.005°, an audible limit cycle reads 2–5× that at its frequency.",
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
  build:
    "change = built minus recorded, evaluated on the recording's own " +
    "timestamps — how much the resample + smoothing + collision projection " +
    "moved the motion. A few degrees is the smoothing doing its job on " +
    "jittery capture; tens of degrees means the projection slid waypoints " +
    "off a limit or the torso — check that region of the graph. peak vel " +
    "out is what tune.motion will actually command.",
  kinematics:
    "churn = total joint travel per minute — a restless null space shows " +
    "up as churn without EE motion (an IK settings problem, not a motor " +
    "problem). 3–15 Hz = mid-band jitter this joint carries; if it's in a " +
    "joint but not in the EE target charts above, the solver is injecting " +
    "it. Solve time spikes (chart above) show up as teleop lurches — the " +
    "target falls behind the hand and catches up.",
}

/** Per-joint score rows for one run, in display units. */
function scoreRows(
  meta: TuningRunMeta,
  arm: string | null
): { cols: ScoreCol[]; rows: { joint: string; values: Record<string, unknown> }[] } | null {
  const m = meta.metrics as Record<string, unknown>
  const perJointKinds: Record<string, ScoreCol[]> = {
    motion: MOTION_COLS,
    filter: FILTER_COLS,
    build: BUILD_COLS,
    kinematics: KIN_COLS,
  }
  if (meta.kind in perJointKinds) {
    const perJoint = m.per_joint as Record<string, Record<string, unknown>> | undefined
    if (!perJoint) return null
    const rows = Object.entries(perJoint)
      .filter(([name]) => arm == null || name.startsWith(`${arm}.`))
      .map(([name, values]) => ({
        joint: arm == null ? name : name.slice(arm.length + 1),
        values,
      }))
    return rows.length > 0 ? { cols: perJointKinds[meta.kind], rows } : null
  }
  if (meta.kind === "sine" || meta.kind === "step" || meta.kind === "gravity") {
    return {
      cols: isA4Run(meta)
        ? A4_COLS
        : meta.kind === "sine"
          ? SINE_COLS
          : meta.kind === "step"
            ? STEP_COLS
            : GRAVITY_COLS,
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
  build: {
    metric: "dev_max_deg",
    deg: false,
    label: "built − recorded max",
    unit: "°",
    warn: 3,
    bad: 8,
  },
  kinematics: {
    metric: "mid_band_deg",
    deg: false,
    label: "solver 3–15 Hz jitter",
    unit: "°",
    warn: 0.15,
    bad: 0.4,
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
 * The tuning workbench: pick what to run (sine / step / recorded motion —
 * optionally with injected noise and the teleop filter stack — motion
 * building, IK analysis), set the numbers inline, hit Run — and the result lands
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
  // Buildable flight recordings (teleop _cmd / gravity-comp _gc), newest
  // first — the Build-motion recording picker's options.
  const [recordings, setRecordings] = useState<TuningRecording[]>([])
  // Effective per-joint config gains (defaults + calibration): the slider
  // baselines and "config N" labels on the gain fields.
  const [gains, setGains] = useState<TuningGains | null>(null)
  const [wireModes, setWireModes] = useState<TuningWireModes | null>(null)
  // The selected motor's live firmware loop gains (Firmware-loop tab): read
  // from the motor over the idle link whenever arm/joint change or a run
  // ends, so the baselines are what the motor actually holds right now.
  const [fwGains, setFwGains] = useState<Record<string, number | null> | null>(null)

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

  const refreshRecordings = useCallback(() => {
    fetchTuningRecordings()
      .then(({ recordings }) => setRecordings(recordings))
      .catch(() => {})
  }, [])

  // Recordings are made by the teleop / gravity-comp operations, outside
  // this workbench — refetch whenever the Build tab is opened so a session
  // recorded since the page loaded shows up in the picker.
  useEffect(() => {
    if (enabled && tabKey === "build") refreshRecordings()
  }, [enabled, tabKey, refreshRecordings])

  const refreshGains = useCallback(() => {
    fetchTuningGains()
      .then(({ gains, wire_modes }) => {
        setGains(gains)
        setWireModes(wire_modes ?? null)
      })
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
    refreshRecordings()
    refreshGains()
  }, [enabled, refreshRuns, refreshMotions, refreshRecordings, refreshGains])

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
      const tabFor = runTab(meta)
      if (!form || !tabFor) return
      setTabKey(tabFor)
      setValues((prev) => ({ ...prev, [tabFor]: form }))
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

  const fwArm = tabValues["arm"] ?? ""
  const fwJoint = tabValues["joint"] ?? ""
  useEffect(() => {
    if (tabKey !== "a4" || !fwArm || !fwJoint || runningOurs) return
    let stale = false
    setFwGains(null)
    fetchMotorDetails(fwArm, fwJoint.toUpperCase())
      .then((d) => {
        if (stale) return
        // Loop gains plus the planner acceleration, under one lookup so the
        // accel field gets the same "motor N" baseline as the gains.
        setFwGains({
          ...(d.gains ?? {}),
          planner_accel: d.planner?.accel ?? null,
          planner_decel: d.planner?.decel ?? null,
        })
      })
      .catch(() => {
        if (!stale) setFwGains(null)
      })
    return () => {
      stale = true
    }
  }, [tabKey, fwArm, fwJoint, runningOurs])

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
      // A value typed for the other vendor's knob stays in the form (it
      // comes back if the joint does) but is never sent: tune.a4 refuses it.
      if (!shownForJoint(f.vendors, tabValues["joint"])) continue
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
      if (f.fwGainKey) {
        const v = fwGains?.[f.fwGainKey]
        return typeof v === "number" && Number.isFinite(v) ? v : null
      }
      if (!f.gainKey || !gains) return null
      const side = tabValues["arm"]
      const joint = tabValues["joint"]
      if (!side || !joint) return null
      const v = gains[side]?.[joint]?.[f.gainKey]
      return typeof v === "number" && Number.isFinite(v) ? v : null
    },
    [gains, fwGains, tabValues]
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
    if (run.meta.kind === "build") return buildJointCharts(run, arm)
    if (run.meta.kind === "kinematics") return kinematicsCharts(run, arm)
    const single = run.meta.kind === "gravity" ? gravityJointChart(run) : pidJointChart(run)
    return single ? [single] : []
  }, [run, arm, armed])
  const scores = meta ? scoreRows(meta, armed ? arm : null) : null
  const legend = meta ? (isA4Run(meta) ? A4_LEGEND : SCORE_LEGEND[meta.kind]) : null
  const perJoint = (meta?.metrics as Record<string, unknown> | undefined)?.per_joint as
    | Record<string, Record<string, unknown>>
    | undefined

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
          {tab.key === "build" && recordings.length === 0 && (
            <span className="text-xs text-amber-200/70">
              no recordings yet — record via teleop or hand-guide in gravity comp first
            </span>
          )}
        </div>
        <p className="max-w-3xl text-xs leading-relaxed text-white/45">{tab.description}</p>

        <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
          {tab.fields
            .filter((f) => shownForJoint(f.vendors, tabValues["joint"]))
            .map((f) => {
              const cfg = configValue(f)
              return (
                <label key={f.key} className="flex flex-col gap-1">
                  <span className="text-[0.65rem] text-white/40">
                    {f.label}
                    {tab.required.includes(f.key) && <span className="text-[#eff483]/70"> *</span>}
                    {cfg != null && (
                      <span className="text-white/25">
                        {f.fwGainKey ? " · motor " : " · config "}
                        {baselineText(f, cfg)}
                      </span>
                    )}
                    {f.fwGainKey && cfg == null && fwArm && fwJoint && (
                      <span className="text-white/25"> · motor …</span>
                    )}
                  </span>
                  {f.type === "overrides" ? (
                    <GainOverrideEditor
                      value={tabValues[f.key] ?? ""}
                      onChange={(v) => setValue(f.key, v)}
                      disabled={runningOurs || busy}
                      gains={gains}
                    />
                  ) : f.type === "wire" ? (
                    <WireModeEditor
                      value={tabValues[f.key] ?? ""}
                      onChange={(v) => setValue(f.key, v)}
                      disabled={runningOurs || busy}
                      configModes={wireModes}
                    />
                  ) : f.type === "pose" ? (
                    <PoseEditor
                      value={tabValues[f.key] ?? ""}
                      onChange={(v) => setValue(f.key, v)}
                      disabled={runningOurs || busy}
                      excludeJoint={tabValues["joint"] ?? ""}
                    />
                  ) : f.type === "boolean" ? (
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
                        {tab.required.includes(f.key) ? "select…" : (f.placeholder ?? "default")}
                      </option>
                      {(f.key === "motion"
                        ? motions.map((m) => ({ value: m.name, label: m.name }))
                        : f.key === "prefix" && tab.key === "build"
                          ? recordings.map((r) => ({
                              value: r.name,
                              label:
                                `${r.name} — ` +
                                (r.kind === "gravity-comp" ? "hand-guided" : "teleop") +
                                (r.durationS != null ? ` · ${Math.round(r.durationS)}s` : ""),
                            }))
                          : (f.options ?? []).map((o) => ({ value: o, label: o }))
                      ).map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
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
                            placeholder={baselineText(f, cfg)}
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
                      placeholder={
                        f.placeholder ??
                        (f.fwGainKey || f.gainKey ? baselineText(f, cfg) : undefined)
                      }
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
                      (r.params.name as string) ?? null,
                      (r.params.source as string) ?? null,
                      r.params.ik === true ? "IK" : null,
                      r.params.noise && r.params.noise !== "none"
                        ? `${r.params.noise as string} noise`
                        : null,
                      r.params.filter === true ? "filtered" : null,
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
                  title={c.note ? `${c.joint} · ${c.note}` : c.joint}
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
          <Badge variant="neutral">{runKindLabel(meta)}</Badge>
          <span className="text-white/60">
            {meta.joint ? `${meta.side} ${meta.joint}` : ""}
            {meta.params.motion ? `${meta.params.motion as string}` : ""}
            {meta.params.source ? `${meta.params.source as string}` : ""}
            {meta.params.name ? `${meta.params.name as string}` : ""}
            {meta.params.ik === true ? " · IK" : ""}
            {meta.params.noise && meta.params.noise !== "none"
              ? ` · ${meta.params.noise as string} noise`
              : ""}
            {meta.params.filter === true ? " · filtered" : ""}
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
              title={c.note ? `${c.joint} · ${c.note}` : c.joint}
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
                  <Badge variant="neutral">{runKindLabel(r)}</Badge>
                  <span className="text-white/70">
                    {[
                      r.side,
                      r.joint,
                      (r.params.motion as string) ?? null,
                      (r.params.name as string) ?? null,
                      (r.params.source as string) ?? null,
                      r.params.ik === true ? "IK" : null,
                      r.params.noise && r.params.noise !== "none"
                        ? `${r.params.noise as string} noise`
                        : null,
                      r.params.filter === true ? "filtered" : null,
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
