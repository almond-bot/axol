import { Cpu, Loader2, Plug, Camera, RotateCw, Server, Power } from "lucide-react"
import type { ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import type { PtpStatus, RobotStatus, StreamStatus, ZedLinkStatus } from "@/lib/supervisor"
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
    <div className="group relative flex min-w-0 flex-1 flex-col gap-2 rounded-xl border border-white/10 bg-white/[0.02] p-3">
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
  hostName,
  onOpenSetup,
  onHostDisconnect,
  robot,
  robotBusy,
  onRobotConnect,
  onRobotDisconnect,
  zed,
  zedBusy,
  onZedConnect,
  onZedDisconnect,
  onZedRestart,
}: {
  conn: ConnState
  host: string
  hostName?: string
  onOpenSetup: () => void
  onHostDisconnect: () => void
  robot: RobotStatus | null
  robotBusy: boolean
  onRobotConnect: () => void
  onRobotDisconnect: () => void
  zed: ZedLinkStatus | null
  zedBusy: boolean
  onZedConnect: () => void
  onZedDisconnect: () => void
  onZedRestart: () => void
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
        title="Axol Host"
        dot={wsDot}
        label={wsLabel}
        pulse={conn === "loading"}
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
        headerRight={
          robot && (rs === "connected" || rs === "busy") ? <MotorGrid robot={robot} /> : undefined
        }
      >
        {rs === "connected" || rs === "busy" ? (
          <Button
            variant="outline"
            size="icon"
            onClick={onRobotDisconnect}
            disabled={robotBusy || rs === "busy"}
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

      <Tile
        icon={<Camera className="size-3.5" />}
        title="ZED box"
        dot={zedDot}
        label={zedLabel}
        headerRight={
          zedConnected ? (
            <div className="flex items-center gap-2">
              <PtpBadge ptp={zed?.ptp} />
              <StreamBadge stream={zed?.stream} />
            </div>
          ) : undefined
        }
      >
        {zedConnected ? (
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="icon"
              onClick={onZedRestart}
              disabled={zedBusy}
              aria-label="Restart PTP clock sync and camera streams"
              className="size-8"
            >
              {zedBusy ? <Loader2 className="animate-spin" /> : <RotateCw />}
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={onZedDisconnect}
              disabled={zedBusy}
              aria-label="Disconnect the ZED box"
              className="size-8"
            >
              <Power />
            </Button>
          </div>
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
 * A labelled status light: short name + tiny dot, so the ZED box header stays
 * compact and never crowds the title.
 * Green = good, amber (pulsing) = loading/starting, red = error.
 */
function StatusDot({ name, dot, pulse }: { name: string; dot: Dot; pulse?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[0.6rem] tracking-wide text-white/35 uppercase">
      <span>{name}</span>
      <span className={cn("size-2 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")} />
    </span>
  )
}

/**
 * PTP clock-sync light for the ZED box header. The link comes up on connect, so
 * it settles (syncing → locked) before any task.
 */
function PtpBadge({ ptp }: { ptp?: PtpStatus }) {
  if (!ptp) return null
  const [dot, pulse] = ptp.locked
    ? (["ok", false] as const)
    : ptp.needsSudo
      ? (["warn", true] as const)
      : ptp.error
        ? (["err", false] as const)
        : (["warn", true] as const)
  return <StatusDot name="clock" dot={dot} pulse={pulse} />
}

/**
 * Camera-stream light for the ZED box header. Streaming starts after the clocks
 * lock for whatever serials were entered on connect; hidden when no cameras are
 * configured.
 */
function StreamBadge({ stream }: { stream?: StreamStatus }) {
  if (!stream) return null
  if (!stream.streaming && stream.cameras.length === 0 && !stream.error) return null
  const [dot, pulse] = stream.ready
    ? (["ok", false] as const)
    : stream.error
      ? (["err", false] as const)
      : (["warn", true] as const)
  return <StatusDot name="stream" dot={dot} pulse={pulse} />
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
