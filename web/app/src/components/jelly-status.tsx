import {
  BatteryCharging,
  BatteryFull,
  BatteryLow,
  BatteryMedium,
  BatteryWarning,
} from "lucide-react"
import { useId } from "react"
import {
  batteryLevel,
  batteryText,
  batteryTooltip,
  wheelName,
  type BatteryLevel,
} from "@/lib/jelly-view"
import {
  wheelMotorHealthy,
  type JellyBattery,
  type JellyWheelsStatus,
  type WheelMotorHealth,
} from "@/lib/supervisor"
import { cn } from "@/lib/utils"

const WHEEL_SHORT: Record<string, string> = {
  front_left: "FL",
  front_right: "FR",
  back_left: "BL",
  back_right: "BR",
}

const SQUARE = {
  ok: "bg-emerald-400/80",
  err: "bg-red-400/70",
  unknown: "bg-white/25",
}

/**
 * The four wheel motors as one status row (FL FR BL BR), mirroring the arm
 * tile's MotorGrid: unknown while a task owns the bus, otherwise healthy or
 * faulted (unreachable counts as a fault).
 */
export function WheelGrid({ status }: { status: JellyWheelsStatus }) {
  if (!status.motors.length) return null
  const color = (m: WheelMotorHealth) =>
    m.reachable == null ? "unknown" : wheelMotorHealthy(m) ? "ok" : "err"
  const tip = (m: WheelMotorHealth) => {
    if (m.reachable == null) return `${wheelName(m)}: unknown (a task owns the bus)`
    if (!m.reachable) return `${wheelName(m)}: unreachable`
    const state = (m.status ?? "OK").replace(/_/g, " ").toLowerCase()
    const temp = m.temperature != null ? ` · ${Math.round(m.temperature)}°C` : ""
    const volt = m.voltage != null ? ` · ${m.voltage.toFixed(1)} V` : ""
    return `${wheelName(m)}: ${state}${temp}${volt}`
  }
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {status.motors.map((m, index, motors) => {
        const tooltip = `wheel-${m.id}-tooltip`
        const first = index === 0
        const last = index === motors.length - 1
        return (
          <div key={m.id} className="flex items-center gap-1">
            <span className="font-mono text-[0.6rem] text-white/35">
              {WHEEL_SHORT[m.name] ?? m.id}
            </span>
            <span
              tabIndex={0}
              aria-describedby={tooltip}
              className="group/motor relative inline-flex rounded-[3px] outline-none focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#111]"
            >
              <span className={cn("size-2.5 rounded-[3px]", SQUARE[color(m)])} />
              <span
                id={tooltip}
                role="tooltip"
                className={cn(
                  "pointer-events-none absolute top-full z-30 mt-2 w-max max-w-56 rounded-md border border-white/15 bg-[#181818] px-2.5 py-1.5 text-center text-xs leading-snug font-normal text-white/85 opacity-0 shadow-lg transition-opacity duration-75 group-hover/motor:opacity-100 group-focus-visible/motor:opacity-100",
                  first ? "left-0" : last ? "right-0" : "left-1/2 -translate-x-1/2"
                )}
              >
                {tip(m)}
              </span>
            </span>
          </div>
        )
      })}
    </div>
  )
}

const BATTERY_TONE: Record<BatteryLevel, string> = {
  charging: "text-sky-300",
  ok: "text-emerald-300",
  low: "text-amber-300",
  critical: "text-red-300",
}

function BatteryIcon({ battery, level }: { battery: JellyBattery; level: BatteryLevel }) {
  const props = { className: "size-3.5", "aria-hidden": true }
  if (level === "charging") return <BatteryCharging {...props} />
  if (level === "critical") return <BatteryWarning {...props} />
  if (level === "low") return <BatteryLow {...props} />
  return battery.percent >= 70 ? <BatteryFull {...props} /> : <BatteryMedium {...props} />
}

/**
 * Jelly's battery as icon + percent (+ volts), with the reading's caveats on
 * hover. A last-known reading (a task owns the lift bus) is dimmed.
 */
export function BatteryIndicator({
  battery,
  align = "right",
  className,
}: {
  battery: JellyBattery
  align?: "left" | "right"
  className?: string
}) {
  const level = batteryLevel(battery)
  const tip = batteryTooltip(battery)
  const tooltipId = useId()
  return (
    <span
      tabIndex={0}
      aria-describedby={tooltipId}
      className={cn(
        "group/battery relative inline-flex shrink-0 items-center gap-1 rounded-md whitespace-nowrap outline-none focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#111]",
        battery.live ? BATTERY_TONE[level] : "text-white/40",
        className
      )}
    >
      <BatteryIcon battery={battery} level={level} />
      <span className="font-mono text-xs tabular-nums">{batteryText(battery)}</span>
      {!battery.charging && (
        <span className="font-mono text-[0.65rem] text-white/35 tabular-nums">
          {battery.voltage.toFixed(1)} V
        </span>
      )}
      <span
        id={tooltipId}
        role="tooltip"
        className={cn(
          "pointer-events-none absolute top-full z-30 mt-2 w-max max-w-64 rounded-md border border-white/15 bg-[#181818] px-2.5 py-1.5 text-left text-xs leading-snug font-normal whitespace-pre-line text-white/85 opacity-0 shadow-lg transition-opacity duration-75 group-hover/battery:opacity-100 group-focus-visible/battery:opacity-100",
          align === "right" ? "right-0" : "left-0"
        )}
      >
        {tip}
      </span>
    </span>
  )
}
