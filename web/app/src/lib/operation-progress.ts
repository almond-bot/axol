import { useEffect, useState } from "react"
import type { OperationProgress } from "@/lib/supervisor"

/**
 * An operation's startup as the operator reads it: what it is doing now and
 * how far along it is. The host reports step keys in order (see
 * almond_axol/progress.py); the labels live here.
 *
 * Each step carries a rough duration that sets its share of the percentage.
 * Within a step the percentage creeps toward that share and never reaches it,
 * so a long step (the IK warm-up, opening the dataset) keeps moving without
 * claiming to be done before the host says so.
 */
const STEPS: Record<string, { label: string; seconds: number }> = {
  headset: { label: "Opening the headset connection", seconds: 1 },
  arms: { label: "Connecting to {device}", seconds: 4 },
  jelly: { label: "Connecting to Jelly", seconds: 3 },
  triggers: { label: "Waiting for both triggers to be released", seconds: 2 },
  grippers: { label: "Calibrating the grippers", seconds: 5 },
  ik: { label: "Warming up IK", seconds: 15 },
  ik_reset: { label: "Warming up IK", seconds: 15 },
  cameras: { label: "Starting cameras", seconds: 8 },
  dataset: { label: "Opening the dataset", seconds: 20 },
  policy: { label: "Loading the policy", seconds: 15 },
  home: { label: "Moving to the start pose", seconds: 8 },
}

const UNKNOWN_STEP = { label: "Starting", seconds: 5 }

export interface StartupView {
  label: string
  /** 0-99 while starting; the run is ready at 100. */
  percent: number
}

export interface StartupContext {
  /** The device the run drives: "Axol" or "Mantis". */
  device: string
  /** A sim run starts the simulator instead of connecting to the arms. */
  sim: boolean
}

function stepInfo(key: string, context: StartupContext) {
  if (key === "arms" && context.sim) return { label: "Starting the simulator", seconds: 2 }
  const info = STEPS[key] ?? UNKNOWN_STEP
  return { ...info, label: info.label.replace("{device}", context.device) }
}

/**
 * The startup label and percentage at `nowMs`, or null when there is none to
 * show: the run is ready, or the host reports no steps (an older host, or an
 * operation that doesn't report them).
 */
export function startupView(
  progress: OperationProgress | null | undefined,
  context: StartupContext,
  nowMs: number
): StartupView | null {
  if (!progress || progress.ready || progress.steps.length === 0) return null
  const steps = progress.steps.map((key) => stepInfo(key, context))
  const total = steps.reduce((sum, s) => sum + s.seconds, 0)
  if (progress.current < 0) return { label: "Starting", percent: 0 }
  const current = steps[Math.min(progress.current, steps.length - 1)]
  const done = steps.slice(0, progress.current).reduce((sum, s) => sum + s.seconds, 0)
  const elapsed =
    progress.stepStartedAt == null ? 0 : Math.max(0, nowMs / 1000 - progress.stepStartedAt)
  const within = Math.min(0.95, 1 - Math.exp(-elapsed / current.seconds))
  const percent = Math.floor((100 * (done + current.seconds * within)) / total)
  return { label: current.label, percent: Math.max(0, Math.min(99, percent)) }
}

/** `startupView`, ticking twice a second while the run is starting. */
export function useStartupView(
  progress: OperationProgress | null | undefined,
  context: StartupContext
): StartupView | null {
  const starting = !!progress && !progress.ready && progress.steps.length > 0
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!starting) return
    const timer = window.setInterval(() => setNow(Date.now()), 500)
    return () => window.clearInterval(timer)
  }, [starting])
  return startupView(progress, context, now)
}
