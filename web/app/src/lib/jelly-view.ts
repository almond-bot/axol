import {
  liftFaultLabel,
  wheelFaults,
  type CanDevicePresence,
  type JellyDevice,
  type JellyLiftStatus,
  type JellyStatus,
  type RobotState,
  type WheelMotorHealth,
} from "./supervisor"

/** Status-dot colour, shared by the connection tiles and the Jelly panels. */
export type StatusDot = "ok" | "busy" | "warn" | "err" | "idle"

export const STATUS_DOT_CLASS: Record<StatusDot, string> = {
  ok: "bg-emerald-400",
  busy: "bg-sky-400",
  warn: "bg-amber-400",
  err: "bg-red-400",
  idle: "bg-white/30",
}

export function wheelName(m: WheelMotorHealth): string {
  return m.name.replace(/_/g, " ")
}

/** The lift's one-line state: height and homed state, or why it is silent. */
export function liftSummaryText(status: JellyLiftStatus): string {
  const board = status.status
  if (status.reachable == null) return "unknown (a task owns the bus)"
  if (status.reachable === false) return "board not answering"
  if (board == null) return "waiting for status…"
  return [
    board.homing
      ? "homing…"
      : board.homed
        ? board.heightPercent != null
          ? `${Math.round(board.heightPercent)}% height`
          : "homed"
        : "not homed",
    board.moving ? "moving" : null,
    board.atLower ? "at lower stop" : board.atUpper ? "at upper stop" : null,
  ]
    .filter(Boolean)
    .join(" · ")
}

export interface JellyDeviceView {
  state: RobotState
  dot: StatusDot
  label: string
  /** A fault worth a red dot while connected, else null. */
  fault: string | null
}

/**
 * One Jelly device's tile state: the link state when the host reports one,
 * else what the CAN inventory says about its interface. `supported` is false
 * on a host too old to expose the Jelly links at all.
 */
export function jellyDeviceView(
  device: JellyDevice,
  jelly: JellyStatus | null | undefined,
  presence: CanDevicePresence | null | undefined,
  supported: boolean
): JellyDeviceView {
  const status = jelly?.[device]
  const state: RobotState = status?.state ?? "disconnected"
  const detected = presence?.present ?? false
  let fault: string | null = null
  if (device === "wheels") {
    const faults = wheelFaults(jelly?.wheels)
    if (faults.length > 0) {
      const first = faults[0]
      const problem = first.reachable
        ? (first.status ?? "error").replace(/_/g, " ").toLowerCase()
        : "unreachable"
      const more = faults.length > 1 ? ` +${faults.length - 1}` : ""
      fault = `${wheelName(first)} ${problem}${more}`
    }
  } else {
    fault = liftFaultLabel(jelly?.lift)
  }
  if (!supported) {
    return { state, dot: "idle", label: "Not available on this host", fault: null }
  }
  const dot: StatusDot =
    state === "connected"
      ? fault
        ? "err"
        : "ok"
      : state === "busy"
        ? "busy"
        : state === "connecting"
          ? "warn"
          : state === "error"
            ? "err"
            : detected
              ? "warn"
              : "idle"
  const label =
    state === "connected"
      ? (fault ?? "Connected")
      : state === "busy"
        ? "In use by task"
        : state === "connecting"
          ? "Connecting…"
          : state === "error"
            ? status?.error || "Error"
            : detected
              ? "CAN detected"
              : presence
                ? "Not detected"
                : "Disconnected"
  return { state, dot, label, fault }
}
