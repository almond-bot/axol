import { AlertTriangle, Cpu, Loader2, Plug, Server, Power } from "lucide-react"
import type { ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import { motorFaultLabel, type RobotStatus } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

type Dot = "ok" | "busy" | "warn" | "err" | "idle"

const DOT_CLASS: Record<Dot, string> = {
  ok: "bg-emerald-400",
  busy: "bg-sky-400",
  warn: "bg-amber-400",
  err: "bg-red-400",
  idle: "bg-white/30",
}

function Tile({
  icon,
  title,
  dot,
  label,
  pulse,
  children,
  extra,
}: {
  icon: ReactNode
  title: string
  dot: Dot
  label: string
  pulse?: boolean
  children?: ReactNode
  extra?: ReactNode
}) {
  return (
    <div className="group relative flex min-w-0 flex-col gap-2.5 overflow-hidden rounded-xl border border-white/10 bg-white/[0.02] p-4">
      {/* title — its own line */}
      <div className="flex items-center gap-2 text-xs tracking-widest text-white/40 uppercase">
        {icon}
        <span className="font-mono">{title}</span>
      </div>
      {/* status — its own line, full width */}
      <div className="flex items-center gap-2 text-sm">
        <span
          className={cn("size-2 shrink-0 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")}
        />
        <span className="min-w-0 truncate text-white/75" title={label}>
          {label}
        </span>
      </div>
      {/* optional extra detail (e.g. the Axol motor grid + fault list) */}
      {extra}
      {/* action — pinned to the bottom so the buttons align across the row */}
      {children && <div className="mt-auto flex justify-end pt-1">{children}</div>}
    </div>
  )
}

/**
 * The two connection tiles: the Axol Host (the machine running `axol serve`)
 * and the Axol robot itself, with live per-motor health and any active motor
 * faults called out (a fault blocks every hardware operation from starting).
 * Cameras and Quest USB live in the Settings tabs below.
 */
export function ConnectionsBar({
  conn,
  host,
  hostName,
  version,
  onOpenSetup,
  onHostDisconnect,
  robot,
  robotBusy,
  onRobotConnect,
  onRobotDisconnect,
}: {
  conn: ConnState
  host: string
  hostName?: string
  /** Installed release version of the serve host, e.g. "0.1.2". */
  version?: string | null
  onOpenSetup: () => void
  onHostDisconnect: () => void
  robot: RobotStatus | null
  robotBusy: boolean
  onRobotConnect: () => void
  onRobotDisconnect: () => void
}) {
  const online = conn === "ok"

  // -- axol host --
  const wsDot: Dot =
    conn === "ok" ? "ok" : conn === "err" ? "err" : conn === "idle" ? "idle" : "warn"
  const wsLabel =
    conn === "ok"
      ? hostName || host || "Connected"
      : conn === "err"
        ? "Offline"
        : conn === "idle"
          ? "Not connected"
          : "Connecting…"

  // -- robot --
  const rs = robot?.state ?? "disconnected"
  const faults = robot?.faults ?? []
  const robotDot: Dot =
    rs === "connected"
      ? faults.length > 0
        ? "err"
        : "ok"
      : rs === "busy"
        ? "busy"
        : rs === "connecting"
          ? "warn"
          : rs === "error"
            ? "err"
            : "idle"
  const robotLabel =
    rs === "connected"
      ? faults.length > 0
        ? `${faults.length} motor ${faults.length > 1 ? "faults" : "fault"}`
        : `${robot?.reachableCount ?? 0}/${robot?.motorCount ?? 16} motors healthy`
      : rs === "busy"
        ? "In use by task"
        : rs === "connecting"
          ? "Connecting…"
          : rs === "error"
            ? robot?.error || "Error"
            : "Disconnected"

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <Tile
        icon={<Server className="size-3.5" />}
        title="Axol Host"
        dot={wsDot}
        label={wsLabel}
        pulse={conn === "loading"}
        extra={
          online && version ? (
            <span className="font-mono text-[0.7rem] text-white/35" title={`v${version}`}>
              axol v{version}
            </span>
          ) : undefined
        }
      >
        {online ? (
          <Button
            variant="outline"
            size="icon"
            onClick={onHostDisconnect}
            aria-label="Disconnect Axol Host"
            className="size-8"
          >
            <Power />
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={onOpenSetup}>
            <Plug />
            Connect
          </Button>
        )}
      </Tile>

      <Tile
        icon={<Cpu className="size-3.5" />}
        title="Axol"
        dot={robotDot}
        label={robotLabel}
        pulse={rs === "connecting"}
        extra={
          robot && (rs === "connected" || rs === "busy") ? (
            <div className="flex flex-col gap-2">
              <MotorGrid robot={robot} />
              {faults.length > 0 && <MotorFaults robot={robot} />}
            </div>
          ) : undefined
        }
      >
        {rs === "connected" || rs === "busy" ? (
          <Button
            variant="outline"
            size="icon"
            onClick={onRobotDisconnect}
            disabled={robotBusy}
            aria-label="Disconnect Axol"
            className="size-8"
          >
            <Power />
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={onRobotConnect}
            disabled={!online || robotBusy}
          >
            {rs === "connecting" || robotBusy ? <Loader2 className="animate-spin" /> : <Plug />}
            Connect
          </Button>
        )}
      </Tile>
    </div>
  )
}

/**
 * Compact 16-dot motor health (two clusters of 8 dots, prefixed with a faint
 * L / R). A dot is red for any fault — unreachable *or* an error status —
 * and its tooltip carries the details (status, temperature, voltage).
 */
export function MotorGrid({ robot }: { robot: RobotStatus }) {
  if (!robot.motors.length) return null
  const arms = ["left", "right"]
  return (
    <div className="flex flex-wrap items-center justify-start gap-x-2 gap-y-1">
      {arms.map((arm) => (
        <div key={arm} className="flex items-center gap-1">
          <span className="font-mono text-[0.6rem] text-white/35">{arm[0].toUpperCase()}</span>
          <div className="flex gap-[3px]">
            {robot.motors
              .filter((m) => m.arm === arm)
              .map((m) => {
                const healthy = m.reachable && (m.status === "OK" || m.status === "DISABLED")
                const details = [
                  m.reachable ? (m.status ?? "unknown") : "unreachable",
                  m.temperature != null ? `${Math.round(m.temperature)}°C` : null,
                  m.voltage != null ? `${m.voltage.toFixed(1)}V` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")
                return (
                  <span
                    key={m.joint}
                    title={`${m.arm} ${m.joint.replace(/_/g, " ").toLowerCase()}: ${details}`}
                    className={cn(
                      "size-2 rounded-[2px]",
                      healthy ? "bg-emerald-400/80" : "bg-red-400/60"
                    )}
                  />
                )
              })}
          </div>
        </div>
      ))}
    </div>
  )
}

/** The active motor faults, spelled out — these block every hardware task. */
function MotorFaults({ robot }: { robot: RobotStatus }) {
  return (
    <div className="flex flex-col gap-1 rounded-md border border-red-400/25 bg-red-400/[0.06] px-2.5 py-2">
      {(robot.faults ?? []).map((f) => (
        <div
          key={`${f.arm}:${f.joint}`}
          className="flex items-center gap-1.5 text-xs text-red-300/90"
        >
          <AlertTriangle className="size-3 shrink-0" />
          <span className="truncate">{motorFaultLabel(f)}</span>
        </div>
      ))}
      <span className="text-[0.65rem] text-white/40">
        Operations are blocked until every motor fault is cleared.
      </span>
    </div>
  )
}
