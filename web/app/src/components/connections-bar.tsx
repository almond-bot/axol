import { Cpu, Loader2, Plug, Camera, Settings2, Server, Power, Usb } from "lucide-react"
import type { ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import {
  cameraCount,
  missingCameraSerials,
  type CameraDevice,
  type CameraSpec,
  type RobotStatus,
  type UsbStatus,
} from "@/lib/supervisor"
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
      {/* optional extra detail (e.g. the Axol motor grid) */}
      {extra}
      {/* action — pinned to the bottom so the buttons align across the row */}
      {children && <div className="mt-auto flex justify-end pt-1">{children}</div>}
    </div>
  )
}

export function ConnectionsBar({
  conn,
  host,
  hostName,
  commit,
  onOpenSetup,
  onHostDisconnect,
  robot,
  robotBusy,
  onRobotConnect,
  onRobotDisconnect,
  cameras,
  cameraDevices,
  cameraDetectError,
  onConfigureCameras,
  usb,
  usbBusy,
  onUsbConnect,
}: {
  conn: ConnState
  host: string
  hostName?: string
  /** Installed git commit of the serve host (null for dev checkouts). */
  commit?: string | null
  onOpenSetup: () => void
  onHostDisconnect: () => void
  robot: RobotStatus | null
  robotBusy: boolean
  onRobotConnect: () => void
  onRobotDisconnect: () => void
  cameras: CameraSpec
  /** Detected ZED devices on the serve host (null until first detection). */
  cameraDevices: CameraDevice[] | null
  /** Why detection couldn't run (SDK/daemon issue), else null. */
  cameraDetectError: string | null
  onConfigureCameras: () => void
  usb: UsbStatus | null
  usbBusy: boolean
  onUsbConnect: () => void
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

  // -- cameras --
  // "Configured" only counts assigned serials; the green badge must also mean
  // those cameras are actually *connected*. Once we have a detection result we
  // cross-check the assigned serials against it: any that aren't physically
  // present flip the badge red (x/N connected) instead of a misleading N/3.
  const camCount = cameraCount(cameras)
  const camMissing =
    camCount > 0 && cameraDevices ? missingCameraSerials(cameras, cameraDevices) : []
  let camDot: Dot
  let camLabel: string
  if (camCount === 0) {
    camDot = "idle"
    camLabel = "Not configured"
  } else if (cameraDetectError) {
    // Detection itself couldn't run (SDK not installed, daemon hung): we can't
    // confirm the cameras, so warn rather than claim they're connected.
    camDot = "warn"
    camLabel = "Can't detect cameras"
  } else if (camMissing.length > 0) {
    camDot = "err"
    camLabel = `${camCount - camMissing.length}/${camCount} connected`
  } else {
    camDot = "ok"
    camLabel = `${camCount}/3 configured`
  }

  // -- quest usb (adb reverse pose tunnel) --
  const usbDot: Dot = !usb
    ? "idle"
    : !usb.installed
      ? "warn"
      : usb.ready
        ? "ok"
        : usb.state === "none"
          ? "idle"
          : "warn"
  const usbLabel = !usb
    ? "—"
    : !usb.installed
      ? "adb not installed"
      : usb.ready
        ? "Controller over USB"
        : usb.state === "device"
          ? "Headset ready"
          : usb.state === "none"
            ? "No headset"
            : usb.state === "unauthorized"
              ? "Authorize on headset"
              : usb.state

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Tile
        icon={<Server className="size-3.5" />}
        title="Axol Host"
        dot={wsDot}
        label={wsLabel}
        pulse={conn === "loading"}
        extra={
          online && commit ? (
            <span className="font-mono text-[0.7rem] text-white/35" title={commit}>
              axol @ {commit.slice(0, 7)}
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
          robot && (rs === "connected" || rs === "busy") ? <MotorGrid robot={robot} /> : undefined
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

      <Tile icon={<Camera className="size-3.5" />} title="Cameras" dot={camDot} label={camLabel}>
        <Button variant="outline" size="sm" onClick={onConfigureCameras} disabled={!online}>
          <Settings2 />
          Configure
        </Button>
      </Tile>

      <Tile icon={<Usb className="size-3.5" />} title="Quest USB" dot={usbDot} label={usbLabel}>
        <Button
          variant="outline"
          size="sm"
          onClick={onUsbConnect}
          disabled={!online || usbBusy || usb?.installed === false}
        >
          {usbBusy ? <Loader2 className="animate-spin" /> : <Plug />}
          {usb?.ready ? "Reconnect" : "Connect"}
        </Button>
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
    <div className="flex flex-wrap items-center justify-start gap-x-2 gap-y-1">
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
