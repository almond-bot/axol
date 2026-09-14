import { ArrowUpFromLine, CircleDot, Cpu, Loader2, Plug, Unplug } from "lucide-react"
import type { ReactNode } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { WheelGrid } from "@/components/jelly-status"
import {
  jellyDeviceView,
  liftSummaryText,
  STATUS_DOT_CLASS,
  wheelName,
  type StatusDot,
} from "@/lib/jelly-view"
import {
  JELLY_DEVICE_LABELS,
  JELLY_DEVICES,
  liftFaultLabel,
  motorFaultLabel,
  wheelMotorHealthy,
  type CanDeviceInventory,
  type CanProfileInventory,
  type HardwareProfile,
  type JellyDevice,
  type JellyLiftStatus,
  type JellyStatus,
  type JellyWheelsStatus,
  type RobotState,
  type RobotStatus,
} from "@/lib/supervisor"
import { cn } from "@/lib/utils"

const PROFILE_LABELS: Record<HardwareProfile, string> = { axol: "Axol", mantis: "Mantis" }
const PROFILES: readonly HardwareProfile[] = ["axol", "mantis"]

const STATE_BADGE: Record<
  RobotState,
  { variant: "success" | "warning" | "destructive" | "neutral"; text: string }
> = {
  connected: { variant: "success", text: "connected" },
  busy: { variant: "warning", text: "in use" },
  connecting: { variant: "neutral", text: "connecting" },
  disconnected: { variant: "neutral", text: "disconnected" },
  error: { variant: "destructive", text: "error" },
}

function HardwareCard({
  icon,
  title,
  state,
  dot,
  label,
  action,
  children,
}: {
  icon: ReactNode
  title: string
  state: RobotState
  dot: StatusDot
  label: string
  action: ReactNode
  children?: ReactNode
}) {
  const badge = STATE_BADGE[state]
  return (
    <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
      <div className="flex min-h-8 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-xs tracking-widest text-white/40 uppercase">
          {icon}
          <span className="truncate font-mono">{title}</span>
          <Badge variant={badge.variant}>{badge.text}</Badge>
        </div>
        <div className="shrink-0">{action}</div>
      </div>
      <div className="flex min-w-0 items-center gap-2 text-sm">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            STATUS_DOT_CLASS[dot],
            state === "connecting" && "animate-pulse"
          )}
        />
        <span className="min-w-0 flex-1 truncate text-white/75" title={label}>
          {label}
        </span>
      </div>
      {children}
    </div>
  )
}

function Row({ name, value, warn }: { name: string; value: ReactNode; warn?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-xs">
      <span className="text-white/40">{name}</span>
      <span className={cn("truncate font-mono", warn ? "text-red-300" : "text-white/70")}>
        {value}
      </span>
    </div>
  )
}

/** Per-wheel detail rows under the FL/FR/BL/BR grid. */
function WheelDetails({ status }: { status: JellyWheelsStatus }) {
  return (
    <div className="flex flex-col gap-2">
      <WheelGrid status={status} />
      <div className="flex flex-col gap-1">
        {status.motors.map((m) => {
          const unknown = m.reachable == null
          const healthy = wheelMotorHealthy(m)
          const problem = unknown
            ? "unknown"
            : !m.reachable
              ? "unreachable"
              : (m.status ?? "OK").replace(/_/g, " ").toLowerCase()
          const extra = [
            m.temperature != null ? `${Math.round(m.temperature)}°C` : null,
            m.voltage != null ? `${m.voltage.toFixed(1)} V` : null,
          ]
            .filter(Boolean)
            .join(" · ")
          return (
            <Row
              key={m.id}
              name={`${wheelName(m)} (id ${m.id})`}
              value={extra ? `${problem} · ${extra}` : problem}
              warn={!unknown && !healthy}
            />
          )
        })}
      </div>
    </div>
  )
}

/** The jelly_legs board's status frame, field by field. */
function LiftDetails({ status }: { status: JellyLiftStatus }) {
  const board = status.status
  const fault = liftFaultLabel(status)
  if (!board) {
    return <p className="text-xs text-white/45">{fault ?? liftSummaryText(status)}</p>
  }
  const yesNo = (v: boolean | null) => (v == null ? "—" : v ? "yes" : "no")
  return (
    <div className="flex flex-col gap-1">
      <Row name="Homed" value={board.homing ? "homing…" : yesNo(board.homed)} />
      <Row
        name="Height"
        value={board.heightPercent != null ? `${Math.round(board.heightPercent)}%` : "—"}
      />
      <Row
        name="Motion"
        value={
          board.moving
            ? "moving"
            : board.atLower
              ? "at lower stop"
              : board.atUpper
                ? "at upper stop"
                : "still"
        }
      />
      <Row name="Stall fault" value={yesNo(board.stallFault)} warn={board.stallFault} />
      {board.driversEnabled != null && (
        <Row name="Drivers enabled" value={yesNo(board.driversEnabled)} />
      )}
      {board.vmPresent != null && (
        <Row name="Motor power" value={yesNo(board.vmPresent)} warn={board.vmPresent === false} />
      )}
      {board.driverFaultMask != null && (
        <Row
          name="Driver faults"
          value={board.driverFaultMask ? `0x${board.driverFaultMask.toString(16)}` : "none"}
          warn={board.driverFaultMask !== 0}
        />
      )}
    </div>
  )
}

export interface HardwareOverviewProps {
  online: boolean
  robot: RobotStatus | null
  robotBusy: boolean
  canProfiles: CanProfileInventory | null
  /** A diagnostic, operation, or setup session owns the hardware. */
  hardwareBusy: boolean
  onRobotConnect: (profile: HardwareProfile) => void
  onRobotDisconnect: () => void
  jelly: JellyStatus | null
  jellySupported: boolean
  jellyBusy: Partial<Record<JellyDevice, boolean>>
  canDevices: CanDeviceInventory | null
  onJellyConnect: (device: JellyDevice) => void
  onJellyDisconnect: (device: JellyDevice) => void
}

/**
 * Every CAN-attached device the host knows about — Axol, Mantis, Jelly's
 * wheels, and Jelly's lift — with its link state, detection, per-motor /
 * per-board detail, and a Connect / Disconnect for each. Axol and Mantis are
 * the two profiles of the one telemetry link, so only one of them is ever
 * connected; the Jelly devices are independent links on their own adapters.
 */
export function HardwareOverview({
  online,
  robot,
  robotBusy,
  canProfiles,
  hardwareBusy,
  onRobotConnect,
  onRobotDisconnect,
  jelly,
  jellySupported,
  jellyBusy,
  canDevices,
  onJellyConnect,
  onJellyDisconnect,
}: HardwareOverviewProps) {
  const activeProfile = robot?.profile ?? "axol"
  const robotLive = robot?.state === "connected" || robot?.state === "busy"
  const busyTitle = "Wait for the active run or setup session to finish."

  const profileCard = (profile: HardwareProfile) => {
    const title = PROFILE_LABELS[profile]
    const active = activeProfile === profile
    const presence = canProfiles?.[profile]
    const detected = presence?.present ?? false
    const state: RobotState = active ? (robot?.state ?? "disconnected") : "disconnected"
    const faults = active ? (robot?.faults ?? []) : []
    const dot: StatusDot =
      state === "connected"
        ? faults.length > 0
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
        ? faults.length > 0
          ? motorFaultLabel(faults[0]) + (faults.length > 1 ? ` +${faults.length - 1}` : "")
          : `${robot?.reachableCount ?? 0}/${robot?.motorCount ?? 0} motors reachable`
        : state === "busy"
          ? "In use by task"
          : state === "connecting"
            ? "Connecting…"
            : state === "error"
              ? robot?.error || "Error"
              : detected
                ? "CAN detected"
                : canProfiles
                  ? "Not detected"
                  : "Disconnected"
    const channels = active ? robot?.channels : presence?.channels
    // The link shows one profile at a time; switching means disconnecting
    // the other first, so the page never silently swaps an open link.
    const otherLive = robotLive && !active
    const action =
      active && robotLive ? (
        <Button
          variant="outline"
          size="icon"
          className="size-8"
          onClick={onRobotDisconnect}
          disabled={robotBusy || hardwareBusy}
          aria-label={`Disconnect ${title}`}
          title={
            hardwareBusy ? busyTitle : `Release the ${title} link. The hardware stays powered.`
          }
        >
          <Unplug />
        </Button>
      ) : (
        <Button
          variant="outline"
          size="sm"
          onClick={() => onRobotConnect(profile)}
          disabled={!online || robotBusy || hardwareBusy || otherLive}
          title={
            otherLive
              ? `Disconnect ${PROFILE_LABELS[activeProfile]} first — the telemetry link shows one device at a time.`
              : hardwareBusy
                ? busyTitle
                : undefined
          }
        >
          {robotBusy && active ? <Loader2 className="animate-spin" /> : <Plug />}
          Connect
        </Button>
      )
    return (
      <HardwareCard
        key={profile}
        icon={<Cpu className="size-3.5" />}
        title={title}
        state={state}
        dot={dot}
        label={label}
        action={action}
      >
        {channels && (channels.left || channels.right) && (
          <div className="flex flex-col gap-1">
            <Row
              name={profile === "mantis" ? "Left gripper" : "Left arm"}
              value={channels.left ?? "—"}
            />
            <Row
              name={profile === "mantis" ? "Right gripper" : "Right arm"}
              value={channels.right ?? "—"}
            />
          </div>
        )}
      </HardwareCard>
    )
  }

  const jellyCard = (device: JellyDevice) => {
    const title = JELLY_DEVICE_LABELS[device]
    const view = jellyDeviceView(device, jelly, canDevices?.[device], jellySupported)
    const busy = jellyBusy[device] ?? false
    const live = view.state === "connected" || view.state === "busy"
    const channel = jelly?.[device]?.channel ?? canDevices?.[device]?.channel ?? null
    const action = live ? (
      <Button
        variant="outline"
        size="icon"
        className="size-8"
        onClick={() => onJellyDisconnect(device)}
        disabled={busy || hardwareBusy || view.state === "busy"}
        aria-label={`Disconnect ${title}`}
        title={
          hardwareBusy || view.state === "busy"
            ? busyTitle
            : `Release the ${title} link. The hardware stays powered.`
        }
      >
        <Unplug />
      </Button>
    ) : (
      <Button
        variant="outline"
        size="sm"
        onClick={() => onJellyConnect(device)}
        disabled={!online || !jellySupported || busy || hardwareBusy}
        title={
          !jellySupported
            ? "Update the serve host to connect Jelly's wheels and lift from the panel."
            : hardwareBusy
              ? busyTitle
              : undefined
        }
      >
        {busy ? <Loader2 className="animate-spin" /> : <Plug />}
        Connect
      </Button>
    )
    return (
      <HardwareCard
        key={device}
        icon={
          device === "wheels" ? (
            <CircleDot className="size-3.5" />
          ) : (
            <ArrowUpFromLine className="size-3.5" />
          )
        }
        title={title}
        state={view.state}
        dot={view.dot}
        label={view.label}
        action={action}
      >
        {channel && <Row name="CAN interface" value={channel} />}
        {jelly && live ? (
          device === "wheels" ? (
            <WheelDetails status={jelly.wheels} />
          ) : (
            <LiftDetails status={jelly.lift} />
          )
        ) : null}
      </HardwareCard>
    )
  }

  return (
    <div className="grid grid-cols-1 items-start gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {PROFILES.map(profileCard)}
      {JELLY_DEVICES.map(jellyCard)}
    </div>
  )
}
