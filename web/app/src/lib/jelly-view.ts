import {
  liftFaultLabel,
  wheelFaults,
  type CanDevicePresence,
  type JellyBattery,
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

/** Below these the battery shows amber, then red (LiFePO4 falls off a cliff under ~10%). */
export const BATTERY_LOW_PERCENT = 25
export const BATTERY_CRITICAL_PERCENT = 10

/**
 * The battery reading worth showing: only while the lift link is up or lent
 * to a task (a stale reading is still labelled with its age), never from a
 * link the operator disconnected.
 */
export function jellyBattery(jelly: JellyStatus | null | undefined): JellyBattery | null {
  const lift = jelly?.lift
  if (!lift || (lift.state !== "connected" && lift.state !== "busy")) return null
  return lift.battery ?? null
}

export type BatteryLevel = "charging" | "ok" | "low" | "critical"

export function batteryLevel(battery: JellyBattery): BatteryLevel {
  if (battery.charging) return "charging"
  if (battery.percent <= BATTERY_CRITICAL_PERCENT) return "critical"
  if (battery.percent <= BATTERY_LOW_PERCENT) return "low"
  return "ok"
}

/** "12 s", "4 min", "2 h 5 min" */
export function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m ? `${h} h ${m} min` : `${h} h`
}

/** Short headline: "62%", "Charging", "~40%" for a reading taken under load. */
export function batteryText(battery: JellyBattery): string {
  if (battery.charging) return "Charging"
  const pct = `${Math.round(battery.percent)}%`
  return battery.underLoad ? `~${pct}` : pct
}

/** Hover detail: the voltage behind the number and how far to trust it. */
export function batteryTooltip(battery: JellyBattery): string {
  const lines = [
    battery.charging
      ? `Battery charging · ${battery.voltage.toFixed(2)} V`
      : `Battery ${Math.round(battery.percent)}% · ${battery.voltage.toFixed(2)} V`,
  ]
  if (!battery.live) lines.push(`Last reading ${formatAge(battery.ageSeconds)} ago`)
  if (battery.underLoad)
    lines.push("Measured while the lift or wheels were moving, so it reads low")
  lines.push("Estimated from the lift board's 24 V rail (2× LiFePO4, 100 Ah)")
  return lines.join("\n")
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
