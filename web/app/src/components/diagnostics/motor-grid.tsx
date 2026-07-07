import { useEffect, useState } from "react"
import { AlertTriangle, CircleCheck, CircleOff, CircleX, Loader2, X } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  JOINTS,
  JOINT_COLORS,
  fetchMotorDetails,
  jointLabel,
  motorKey,
  type ArmSide,
  type MotorDetails,
  type SlowReading,
  type TelemetryFrame,
} from "@/lib/telemetry"

// Advisory display thresholds only (the motors enforce their own protection
// limits in firmware): tint the temperature readout as it climbs.
const TEMP_WARN_C = 60
const TEMP_HOT_C = 75

type StatusKind = "ok" | "disabled" | "warning" | "error" | "unknown"

function statusKind(reading: SlowReading | undefined): StatusKind {
  if (!reading) return "unknown"
  if (!reading.reachable) return "error"
  switch (reading.status) {
    case "OK":
      return "ok"
    case "DISABLED":
      return "disabled"
    case null:
      return "unknown"
    default:
      return "error"
  }
}

/** Status is never color-alone: each kind pairs an icon with its label. */
const STATUS_META: Record<StatusKind, { icon: typeof CircleCheck; className: string }> = {
  ok: { icon: CircleCheck, className: "text-emerald-300" },
  disabled: { icon: CircleOff, className: "text-white/45" },
  warning: { icon: AlertTriangle, className: "text-amber-300" },
  error: { icon: CircleX, className: "text-red-300" },
  unknown: { icon: CircleOff, className: "text-white/30" },
}

function fmt(v: number | null | undefined, digits: number, suffix: string): string {
  return v == null ? "–" : `${v.toFixed(digits)}${suffix}`
}

/**
 * Per-motor status tiles for one arm: status (icon + label), temperature and
 * bus voltage from the 1 Hz health ping. A tile opens the full `motor.info`
 * readout (model, firmware, mode, gains) fetched over the idle link.
 */
export function MotorGrid({
  arm,
  slow,
  frames,
  version,
  canInspect,
}: {
  arm: ArmSide
  slow: Record<string, SlowReading>
  frames: TelemetryFrame[]
  version: number
  canInspect: boolean
}) {
  const [inspecting, setInspecting] = useState<string | null>(null)
  void version // tiles re-render with the stream tick so position stays fresh

  const latestFrame = frames.length > 0 ? frames[frames.length - 1] : null

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {JOINTS.map((joint) => {
        const key = motorKey(arm, joint)
        const reading = slow[key]
        const kind = statusKind(reading)
        const Meta = STATUS_META[kind]
        const temp = reading?.temperature ?? null
        const pos = latestFrame?.m[key]?.[0] ?? null
        return (
          <button
            key={joint}
            type="button"
            disabled={!canInspect}
            onClick={() => setInspecting(joint)}
            title={canInspect ? "Motor details" : undefined}
            className={cn(
              "flex flex-col gap-1 rounded-lg border border-white/10 bg-white/[0.02] p-2.5 text-left transition-colors",
              canInspect && "hover:border-white/25 hover:bg-white/[0.05]"
            )}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block size-2 shrink-0 rounded-full"
                style={{ background: JOINT_COLORS[joint] }}
              />
              <span className="truncate text-xs font-medium text-white/85 capitalize">
                {jointLabel(joint)}
              </span>
            </div>
            <div className={cn("flex items-center gap-1 text-xs", Meta.className)}>
              <Meta.icon className="size-3.5 shrink-0" />
              <span className="truncate font-mono text-[0.65rem] tracking-wide">
                {reading?.status ?? (kind === "error" ? "UNREACHABLE" : "—")}
              </span>
            </div>
            <div className="flex items-baseline gap-2 font-mono text-[0.65rem] text-white/45 tabular-nums">
              <span
                className={cn(
                  temp != null && temp >= TEMP_HOT_C && "text-red-300",
                  temp != null && temp >= TEMP_WARN_C && temp < TEMP_HOT_C && "text-amber-300"
                )}
              >
                {fmt(temp, 0, "°C")}
              </span>
              <span>{fmt(reading?.voltage, 1, "V")}</span>
              <span className="ml-auto">{fmt(pos, 2, "")}</span>
            </div>
          </button>
        )
      })}
      {inspecting && (
        <MotorDetailsDialog arm={arm} joint={inspecting} onClose={() => setInspecting(null)} />
      )}
    </div>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <span className="text-xs text-white/45">{label}</span>
      <span className="font-mono text-xs text-white/85 tabular-nums">{value}</span>
    </div>
  )
}

function MotorDetailsDialog({
  arm,
  joint,
  onClose,
}: {
  arm: ArmSide
  joint: string
  onClose: () => void
}) {
  const [details, setDetails] = useState<MotorDetails | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    fetchMotorDetails(arm, joint)
      .then((d) => {
        if (active) setDetails(d)
      })
      .catch((e) => {
        if (active) setError(String(e).replace(/^Error:\s*/, ""))
      })
    return () => {
      active = false
    }
  }, [arm, joint])

  const rad = (v: number | null) =>
    v == null ? "–" : `${v.toFixed(3)} rad (${((v * 180) / Math.PI).toFixed(1)}°)`

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <Card
        className="w-full max-w-sm gap-3 bg-[#1a1a1a] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{ background: JOINT_COLORS[joint as keyof typeof JOINT_COLORS] }}
          />
          <h3 className="font-heading text-sm font-semibold capitalize">
            {arm} {jointLabel(joint)}
          </h3>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto size-7"
            onClick={onClose}
            aria-label="Close"
          >
            <X />
          </Button>
        </div>
        {error ? (
          <p className="text-sm text-red-300">{error}</p>
        ) : !details ? (
          <div className="flex items-center gap-2 py-4 text-sm text-white/40">
            <Loader2 className="size-4 animate-spin" /> Reading motor…
          </div>
        ) : (
          <div className="flex flex-col divide-y divide-white/[0.06]">
            <DetailRow label="Status" value={details.status ?? "–"} />
            <DetailRow label="Control mode" value={details.mode ?? "–"} />
            <DetailRow label="Model" value={details.model ?? "–"} />
            <DetailRow label="Firmware" value={details.firmware?.toString() ?? "–"} />
            <DetailRow label="Position" value={rad(details.position)} />
            <DetailRow label="Velocity" value={fmt(details.velocity, 3, " rad/s")} />
            <DetailRow label="Torque" value={fmt(details.torque, 3, " Nm")} />
            <DetailRow label="Temperature" value={fmt(details.temperature, 1, " °C")} />
            <DetailRow label="Bus voltage" value={fmt(details.voltage, 1, " V")} />
            {details.gains &&
              Object.entries(details.gains)
                .filter(([, v]) => v != null)
                .map(([k, v]) => (
                  <DetailRow key={k} label={`Gain ${k}`} value={String(v)} />
                ))}
          </div>
        )}
      </Card>
    </div>
  )
}
