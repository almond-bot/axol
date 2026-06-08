import { Cpu, Loader2, Plug, Camera, Server, Power } from "lucide-react"
import type { ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import type { RobotStatus, ZedLinkStatus } from "@/lib/supervisor"
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
  headerRight,
}: {
  icon: ReactNode
  title: string
  dot: Dot
  label: string
  pulse?: boolean
  children?: ReactNode
  headerRight?: ReactNode
}) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs tracking-widest text-white/40 uppercase">
          {icon}
          <span className="font-mono">{title}</span>
        </div>
        {headerRight}
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2 text-sm">
          <span
            className={cn("size-2 shrink-0 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")}
          />
          <span className="truncate text-white/75">{label}</span>
        </span>
        {children}
      </div>
    </div>
  )
}

export function ConnectionsBar({
  conn,
  host,
  onOpenSetup,
  robot,
  robotBusy,
  onRobotConnect,
  onRobotDisconnect,
  zed,
  onZedConnect,
  onZedDisconnect,
}: {
  conn: ConnState
  host: string
  onOpenSetup: () => void
  robot: RobotStatus | null
  robotBusy: boolean
  onRobotConnect: () => void
  onRobotDisconnect: () => void
  zed: ZedLinkStatus | null
  onZedConnect: () => void
  onZedDisconnect: () => void
}) {
  const online = conn === "ok"

  // -- workstation --
  const wsDot: Dot = conn === "ok" ? "ok" : conn === "err" ? "err" : "warn"
  const wsLabel = conn === "ok" ? host || "Connected" : conn === "err" ? "Offline" : "Connecting…"

  // -- robot --
  const rs = robot?.state ?? "disconnected"
  const robotDot: Dot =
    rs === "connected"
      ? robot && robot.reachableCount < robot.motorCount
        ? "warn"
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
      ? `${robot?.reachableCount ?? 0}/${robot?.motorCount ?? 16} motors`
      : rs === "busy"
        ? "In use by task"
        : rs === "connecting"
          ? "Connecting…"
          : rs === "error"
            ? robot?.error || "Error"
            : "Disconnected"

  // -- zed --
  const zedConnected = !!zed?.connected
  const zedDot: Dot = zedConnected ? "ok" : zed?.error ? "err" : "idle"
  const zedLabel = zedConnected
    ? zed?.info?.hostname || zed?.boxUrl || "Connected"
    : zed?.error
      ? "Unreachable"
      : "Not connected"

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <Tile
        icon={<Server className="size-3.5" />}
        title="Workstation"
        dot={wsDot}
        label={wsLabel}
        pulse={conn === "loading"}
      >
        <Button variant="outline" size="sm" onClick={onOpenSetup}>
          <Plug />
          Setup
        </Button>
      </Tile>

      <Tile
        icon={<Cpu className="size-3.5" />}
        title="Axol"
        dot={robotDot}
        label={robotLabel}
        pulse={rs === "connecting"}
        headerRight={
          robot && (rs === "connected" || rs === "busy") ? <MotorGrid robot={robot} /> : undefined
        }
      >
        {rs === "connected" || rs === "busy" ? (
          <Button
            variant="outline"
            size="sm"
            onClick={onRobotDisconnect}
            disabled={robotBusy || rs === "busy"}
          >
            <Power />
            Disconnect
          </Button>
        ) : (
          <Button size="sm" onClick={onRobotConnect} disabled={!online || robotBusy}>
            {rs === "connecting" || robotBusy ? <Loader2 className="animate-spin" /> : <Power />}
            Connect
          </Button>
        )}
      </Tile>

      <Tile icon={<Camera className="size-3.5" />} title="ZED box" dot={zedDot} label={zedLabel}>
        {zedConnected ? (
          <Button variant="outline" size="sm" onClick={onZedDisconnect}>
            <Power />
            Disconnect
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={onZedConnect} disabled={!online}>
            <Plug />
            Connect
          </Button>
        )}
      </Tile>
    </div>
  )
}

/**
 * Compact 16-dot motor health, sized to sit inline in the Axol tile header
 * (two clusters of 8 dots, prefixed with a faint L / R) so the tile stays the
 * same height as the others.
 */
export function MotorGrid({ robot }: { robot: RobotStatus }) {
  if (!robot.motors.length) return null
  const arms = ["left", "right"]
  return (
    <div className="flex items-center gap-2">
      {arms.map((arm) => (
        <div key={arm} className="flex items-center gap-1">
          <span className="font-mono text-[0.6rem] text-white/35">{arm[0].toUpperCase()}</span>
          <div className="flex gap-[3px]">
            {robot.motors
              .filter((m) => m.arm === arm)
              .map((m) => (
                <span
                  key={m.joint}
                  title={`${m.joint}${m.status ? ` — ${m.status}` : ""}`}
                  className={cn(
                    "size-2 rounded-[2px]",
                    m.reachable ? "bg-emerald-400/80" : "bg-red-400/60"
                  )}
                />
              ))}
          </div>
        </div>
      ))}
    </div>
  )
}
