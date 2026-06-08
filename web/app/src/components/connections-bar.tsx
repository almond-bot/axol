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
  zedBusy,
  onZedConnect,
  onZedDisconnect,
  onZedRestart,
}: {
  conn: ConnState
  host: string
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

      <Tile
        icon={<Camera className="size-3.5" />}
        title="ZED box"
        dot={zedDot}
        label={zedLabel}
        headerRight={
          zedConnected ? (
            <div className="flex flex-col items-end gap-0.5">
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
              size="sm"
              onClick={onZedRestart}
              disabled={zedBusy}
              title="Restart PTP clock sync and camera streams"
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
 * Compact PTP clock-sync state for the ZED box header. The link comes up on
 * connect, so this shows the sync settling (syncing → locked) before any task.
 */
function PtpBadge({ ptp }: { ptp?: PtpStatus }) {
  if (!ptp) return null
  const [dot, text, pulse] = ptp.locked
    ? (["ok", "clock locked", false] as const)
    : ptp.needsSudo
      ? (["warn", "needs sudo", false] as const)
      : ptp.error
        ? (["err", "sync error", false] as const)
        : ptp.running
          ? (["warn", "syncing clocks…", true] as const)
          : (["idle", "clock idle", false] as const)
  return (
    <span
      className="flex items-center gap-1.5 text-[0.65rem] text-white/45"
      title={ptp.error ?? undefined}
    >
      <span className={cn("size-1.5 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")} />
      {text}
    </span>
  )
}

/**
 * Compact camera-stream state for the ZED box header. Streaming starts after
 * the clocks lock for whatever serials were entered on connect; hidden when no
 * cameras are configured so the header stays uncluttered.
 */
function StreamBadge({ stream }: { stream?: StreamStatus }) {
  if (!stream) return null
  if (!stream.streaming && stream.cameras.length === 0 && !stream.error) return null
  const n = stream.cameras.length
  const [dot, text, pulse] = stream.ready
    ? (["ok", `${n} camera${n === 1 ? "" : "s"} live`, false] as const)
    : stream.error
      ? (["err", "stream error", false] as const)
      : stream.streaming
        ? (["warn", "starting cameras…", true] as const)
        : (["idle", "cameras idle", false] as const)
  return (
    <span
      className="flex items-center gap-1.5 text-[0.65rem] text-white/45"
      title={stream.error ?? undefined}
    >
      <span className={cn("size-1.5 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")} />
      {text}
    </span>
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
