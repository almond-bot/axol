import { ArrowUpFromLine, CircleDot, Cpu, Loader2, Plug, Unplug } from "lucide-react"
import type { ReactNode } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { BatteryIndicator, WheelGrid } from "@/components/jelly-status"
import {
  jellyBattery,
  jellyDeviceView,
  liftSummaryText,
  STATUS_DOT_CLASS,
  type StatusDot,
} from "@/lib/jelly-view"
import {
  JELLY_DEVICE_LABELS,
  JELLY_DEVICES,
  liftFaultLabel,
  motorFaultLabel,
  type CanDeviceInventory,
  type CanProfileInventory,
  type HardwareProfile,
  type JellyDevice,
  type JellyStatus,
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

/**
 * One device row: identity, link state, a one-line status (or the wheel
 * grid), its CAN interface(s), and the connect / disconnect control. Wraps
 * to two lines on narrow screens instead of squeezing.
 */
function HardwareRow({
  icon,
  title,
  state,
  dot,
  status,
  channels,
  action,
}: {
  icon: ReactNode
  title: string
  state: RobotState
  dot: StatusDot
  status: ReactNode
  channels: ReactNode
  action: ReactNode
}) {
  const badge = STATE_BADGE[state]
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3.5 py-2.5">
      <div className="flex w-40 shrink-0 items-center gap-2 text-xs tracking-widest text-white/45 uppercase">
        {icon}
        <span className="truncate font-mono">{title}</span>
      </div>
      <Badge variant={badge.variant} className="shrink-0">
        {badge.text}
      </Badge>
      <div className="flex min-w-48 flex-1 items-center gap-2 text-sm">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            STATUS_DOT_CLASS[dot],
            state === "connecting" && "animate-pulse"
          )}
        />
        {status}
      </div>
      <div className="hidden shrink-0 font-mono text-xs text-white/40 lg:block">{channels}</div>
      <div className="shrink-0">{action}</div>
    </div>
  )
}

function Text({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span className="min-w-0 flex-1 truncate text-white/75" title={title}>
      {children}
    </span>
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
 * wheels, and Jelly's lift — one row each, with link state, detection,
 * status detail, CAN interface, and Connect / Disconnect. Axol and Mantis
 * are the two profiles of the one telemetry link, so only one of them is
 * ever connected; the Jelly devices are independent links on their own
 * adapters. Per-wheel temperatures and voltages are in the wheel grid's
 * tooltips; the Axol / Mantis motors have the Motors section below.
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

  const profileRow = (profile: HardwareProfile) => {
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
    const channelText = channels
      ? [channels.left && `L ${channels.left}`, channels.right && `R ${channels.right}`]
          .filter(Boolean)
          .join(" · ")
      : ""
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
      <HardwareRow
        key={profile}
        icon={<Cpu className="size-3.5" />}
        title={title}
        state={state}
        dot={dot}
        status={<Text title={label}>{label}</Text>}
        channels={channelText || "—"}
        action={action}
      />
    )
  }

  const jellyRow = (device: JellyDevice) => {
    const title = JELLY_DEVICE_LABELS[device]
    const view = jellyDeviceView(device, jelly, canDevices?.[device], jellySupported)
    const busy = jellyBusy[device] ?? false
    const live = view.state === "connected" || view.state === "busy"
    const channel = jelly?.[device]?.channel ?? canDevices?.[device]?.channel ?? "—"
    const battery = device === "lift" && jellySupported ? jellyBattery(jelly) : null
    const summary =
      jelly && view.state === "connected" && !view.fault ? (
        device === "wheels" ? (
          <WheelGrid status={jelly.wheels} />
        ) : (
          <Text title={liftSummaryText(jelly.lift)}>{liftSummaryText(jelly.lift)}</Text>
        )
      ) : jelly && view.state === "connected" && device === "lift" ? (
        <Text title={liftFaultLabel(jelly.lift) ?? view.label}>
          {liftFaultLabel(jelly.lift) ?? view.label}
        </Text>
      ) : (
        <Text title={view.label}>{view.label}</Text>
      )
    const status = battery ? (
      <>
        {summary}
        <BatteryIndicator battery={battery} />
      </>
    ) : (
      summary
    )
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
        {busy ? <Loader2 className="animate-spin" /> : <Unplug />}
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
      <HardwareRow
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
        status={status}
        channels={channel}
        action={action}
      />
    )
  }

  return (
    <div className="divide-y divide-white/10 rounded-xl border border-white/10 bg-white/[0.02]">
      {PROFILES.map(profileRow)}
      {JELLY_DEVICES.map(jellyRow)}
    </div>
  )
}
