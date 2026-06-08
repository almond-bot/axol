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

  // -- workstation --
  const wsDot: Dot = conn === "ok" ? "ok" : conn === "err" ? "err" : "warn"
  const wsLabel =
    conn === "ok" ? hostName || host || "Connected" : conn === "err" ? "Offline" : "Connecting…"

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
          <div className="relative flex items-center">
            {/* Restart stays out of the layout (and off the hostname) until hover. */}
            <Button
              variant="outline"
              size="sm"
              onClick={onZedRestart}
              disabled={zedBusy}
              title="Restart PTP clock sync and camera streams"
              className="pointer-events-none absolute right-full mr-1.5 opacity-0 backdrop-blur-sm transition-opacity duration-150 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100"
            >
              {zedBusy ? <Loader2 className="animate-spin" /> : <RotateCw />}
              Restart
            </Button>
            <Button variant="outline" size="sm" onClick={onZedDisconnect} disabled={zedBusy}>
              <Power />
              Disconnect
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
 * A labelled status light: tiny dot, no text (the label is the hover tooltip),
 * so the ZED box header stays compact and never crowds the title.
 * Green = good, amber (pulsing) = loading/starting, red = error.
 */
function StatusDot({
  name,
  dot,
  label,
  pulse,
}: {
  name: string
  dot: Dot
  label: string
  pulse?: boolean
}) {
  return (
    <span
      className="flex items-center gap-1.5 text-[0.6rem] tracking-wide text-white/35"
      title={label}
    >
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
  const [dot, label, pulse] = ptp.locked
    ? (["ok", "Clock locked", false] as const)
    : ptp.needsSudo
      ? (["warn", "Clock sync needs sudo", true] as const)
      : ptp.error
        ? (["err", ptp.error || "Clock sync error", false] as const)
        : (["warn", "Syncing clocks…", true] as const)
  return <StatusDot name="clock" dot={dot} label={label} pulse={pulse} />
}

/**
 * Camera-stream light for the ZED box header. Streaming starts after the clocks
 * lock for whatever serials were entered on connect; hidden when no cameras are
 * configured.
 */
function StreamBadge({ stream }: { stream?: StreamStatus }) {
  if (!stream) return null
  if (!stream.streaming && stream.cameras.length === 0 && !stream.error) return null
  const [dot, label, pulse] = stream.ready
    ? (["ok", "Cameras live", false] as const)
    : stream.error
      ? (["err", stream.error || "Stream error", false] as const)
      : (["warn", "Starting cameras…", true] as const)
  return <StatusDot name="stream" dot={dot} label={label} pulse={pulse} />
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
