import {
  Bot,
  Check,
  CircleDot,
  Grab,
  Loader2,
  Plug,
  Power,
  RotateCcw,
  Server,
  Settings2,
  Unplug,
} from "lucide-react"
import { deviceErrorText } from "@/lib/hardware-status"
import { useCallback, useState, type ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import { BatteryIndicator, WheelGrid } from "@/components/jelly-status"
import {
  jellyBattery,
  jellyDeviceView,
  liftSummaryText,
  STATUS_DOT_CLASS,
  type StatusDot,
} from "@/lib/jelly-view"
import type { SettingsScope } from "@/lib/settings-scope"
import {
  JELLY_DEVICE_LABELS,
  JELLY_DEVICES,
  restartHost,
  shutdownHost,
  type CanDeviceInventory,
  type CanProfileInventory,
  type HardwareProfile,
  type JellyDevice,
  type JellyStatus,
  type MotorHealth,
  type RobotStatus,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"

type Dot = StatusDot

const DOT_CLASS = STATUS_DOT_CLASS

/**
 * One device on the Devices page: an icon, its name and status, the main
 * action at the top right, then what it reports and its other actions.
 */
function DeviceCard({
  icon,
  name,
  dot,
  status,
  statusTitle,
  pulse,
  badge,
  action,
  children,
  footer,
}: {
  icon: ReactNode
  name: string
  dot: Dot
  status: string
  /** The host's own words, for the tooltip. */
  statusTitle?: string | null
  pulse?: boolean
  /** A marker next to the name (the selected device). */
  badge?: ReactNode
  /** The main action: Connect or Disconnect. */
  action?: ReactNode
  /** What the device reports while connected. */
  children?: ReactNode
  /** Settings and the less frequent actions. */
  footer?: ReactNode
}) {
  const active = dot === "ok" || dot === "busy"
  return (
    <section className="flex min-w-0 flex-col rounded-2xl border border-white/10 bg-white/[0.02]">
      <div className="flex items-start gap-4 p-5">
        <span
          className={cn(
            "flex size-11 shrink-0 items-center justify-center rounded-xl border [&_svg]:size-5",
            active
              ? "border-[#eff483]/15 bg-[#eff483]/[0.07] text-[#eff483]"
              : "border-white/10 bg-white/[0.04] text-white/45"
          )}
        >
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-h-6 flex-wrap items-center gap-2">
            <h2 className="font-heading text-base font-semibold text-white">{name}</h2>
            {badge}
          </div>
          <p className="mt-1 flex min-w-0 items-center gap-2 text-sm">
            <span
              className={cn(
                "size-2 shrink-0 rounded-full",
                DOT_CLASS[dot],
                pulse && "animate-pulse"
              )}
            />
            <span
              className={cn(
                "min-w-0 truncate",
                dot === "err" ? "text-red-300/90" : "text-white/55"
              )}
              title={statusTitle ?? status}
            >
              {status}
            </span>
          </p>
        </div>
        {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
      </div>
      {children && <div className="border-t border-white/[0.06] px-5 py-4">{children}</div>}
      {footer && (
        <div className="mt-auto flex flex-wrap items-center gap-1 border-t border-white/[0.06] px-3 py-2">
          {footer}
        </div>
      )}
    </section>
  )
}

/** A label and its value, one line of a device card's details. */
function Detail({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-4 text-sm">
      <span className="shrink-0 text-white/45">{label}</span>
      <span className="min-w-0 truncate text-right text-white/80">{children}</span>
    </div>
  )
}

/** Opens a device's settings, from the card's footer. */
function SettingsButton({ onClick }: { onClick?: () => void }) {
  if (!onClick) return null
  return (
    <Button variant="ghost" size="sm" onClick={onClick} className="text-white/60">
      <Settings2 />
      Open settings
    </Button>
  )
}

/** The two host power actions offered on the connected Host Computer tile. */
type PowerAction = "shutdown" | "restart"

const POWER_ACTIONS: Record<
  PowerAction,
  { title: string; body: string; button: string; done: string }
> = {
  shutdown: {
    title: "Shut down the Host Computer",
    body:
      "Turn off the Host Computer. This panel disconnects right away, and " +
      "you need to be at the robot to turn it back on.",
    button: "Shut down",
    done: "The Host Computer is shutting down.",
  },
  restart: {
    title: "Restart the Host Computer",
    body:
      "Restart the Host Computer. This panel disconnects until it is back " +
      "up, usually within a minute.",
    button: "Restart",
    done: "The Host Computer is restarting.",
  },
}

/**
 * The system-wide device selection, shown on each hardware tile: the selected
 * device carries a highlighted marker, the other offers Select. One choice for
 * the whole panel (persisted with the shared settings) instead of a
 * per-operation Mantis toggle — teleop and data collection follow it, and
 * Axol-only operations wait until it is back on Axol.
 */
function SelectedBadge({ saving }: { saving: boolean }) {
  return (
    <span
      className="flex shrink-0 items-center gap-1 rounded-full bg-[#eff483]/15 px-2 py-0.5 text-xs text-[#eff483]"
      title="Every operation runs on this device"
    >
      {saving ? <Loader2 className="size-3 animate-spin" /> : <Check className="size-3" />}
      Selected
    </span>
  )
}

/**
 * Connection tiles for the Host Computer, the two hardware profiles, and Jelly's
 * two devices. Axol and Mantis share one idle telemetry link, so connecting
 * either hardware tile switches that link to its CAN interfaces and motor
 * set. Jelly's wheels and lift ride their own single-channel adapters and
 * have independent idle links (status only), so their tiles connect and
 * disconnect on their own. Clicking a tile's title opens that connection's
 * settings: Axol, Mantis, and Jelly each have their own, and the host tile
 * opens the general (shared) settings. The Axol/Mantis tiles also carry
 * the system-wide device selection: the selected one is marked, the other
 * offers "Use for operations".
 *
 * The host tile also carries the host power controls (restart / shut down,
 * each behind a confirmation) — the Disconnect button only drops this
 * browser's view and never touches the machine.
 */
export function ConnectionsBar({
  conn,
  host,
  hostName,
  version,
  onOpenSetup,
  onHostDisconnect,
  opRunning = false,
  hostInstalling = false,
  robot,
  robotBusy,
  canProfiles,
  onRobotConnect,
  onRobotDisconnect,
  jelly,
  jellySupported = true,
  jellyBusy,
  canDevices,
  onJellyConnect,
  onJellyDisconnect,
  selectedProfile,
  onSelectProfile,
  selectDisabled = false,
  selectDisabledReason,
  selectSaving = false,
  onOpenSettings,
}: {
  conn: ConnState
  host: string
  hostName?: string
  /** Installed release version of the serve host, e.g. "0.1.2". */
  version?: string | null
  onOpenSetup: () => void
  onHostDisconnect: () => void
  /** An operation or session is in flight — the server would refuse a power
   *  action (409), so the confirm button is disabled with an explanation. */
  opRunning?: boolean
  /** The host is reinstalling or provisioning (apt/dpkg); a power action now
   *  could leave its package manager half-configured, so the server refuses. */
  hostInstalling?: boolean
  robot: RobotStatus | null
  robotBusy: boolean
  /** Configured profiles whose CAN netdevs or exact persisted USB hub exist. */
  canProfiles?: CanProfileInventory | null
  onRobotConnect: (profile: HardwareProfile) => void
  onRobotDisconnect: () => void
  /** Idle-link status of Jelly's wheels and lift (null until fetched). */
  jelly?: JellyStatus | null
  /** False on a serve host too old to expose the Jelly links. */
  jellySupported?: boolean
  /** A connect/disconnect is in flight for that device. */
  jellyBusy?: Partial<Record<JellyDevice, boolean>>
  /** Presence of the wheel/lift CAN interfaces from the host inventory. */
  canDevices?: CanDeviceInventory | null
  onJellyConnect?: (device: JellyDevice) => void
  onJellyDisconnect?: (device: JellyDevice) => void
  /** The system-wide device selection, marked on its tile. */
  selectedProfile?: HardwareProfile
  /** Switches the system-wide device selection (offered on the other tile). */
  onSelectProfile?: (profile: HardwareProfile) => void
  selectDisabled?: boolean
  selectDisabledReason?: string | null
  /** The selection is being written to the host. */
  selectSaving?: boolean
  /** Opens the settings for a connection (only offered while online). */
  onOpenSettings?: (scope: SettingsScope) => void
}) {
  const toast = useToast()
  const online = conn === "ok"

  // Host power (shutdown -h/-r now), behind a confirmation dialog. The server
  // refuses while an operation or session is running.
  const [powerOpen, setPowerOpen] = useState<PowerAction | null>(null)
  const [powerBusy, setPowerBusy] = useState(false)
  const confirmPower = useCallback(
    async (action: PowerAction) => {
      setPowerBusy(true)
      try {
        await (action === "shutdown" ? shutdownHost() : restartHost())
        setPowerOpen(null)
        toast.success(POWER_ACTIONS[action].done)
      } catch (e) {
        toast.error(String(e))
      } finally {
        setPowerBusy(false)
      }
    },
    [toast]
  )

  // -- Host Computer --
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

  // Axol and Mantis are two profiles of the same server-owned telemetry link.
  // Older hosts omit profile and are necessarily the original Axol profile.
  const activeProfile = robot?.profile ?? "axol"
  const hardwareCard = (profile: HardwareProfile, title: string) => {
    const active = activeProfile === profile
    const detected = canProfiles?.[profile].present ?? false
    const state = active ? (robot?.state ?? "disconnected") : "disconnected"
    const faults = active ? (robot?.faults ?? []) : []
    const dot: Dot =
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
        ? "Connected"
        : state === "busy"
          ? "Used by a running test or operation"
          : state === "connecting"
            ? "Connecting…"
            : state === "error"
              ? deviceErrorText(robot?.error, title)
              : detected
                ? "Detected"
                : canProfiles
                  ? "Not detected"
                  : "Disconnected"

    const live = active && (state === "connected" || state === "busy")
    const selected = selectedProfile === profile
    return (
      <DeviceCard
        icon={profile === "axol" ? <Bot /> : <Grab />}
        name={title}
        dot={dot}
        status={label}
        statusTitle={state === "error" ? robot?.error : null}
        pulse={state === "connecting"}
        badge={selectedProfile && selected ? <SelectedBadge saving={selectSaving} /> : undefined}
        action={
          live ? (
            <Button
              variant="outline"
              onClick={onRobotDisconnect}
              disabled={robotBusy || opRunning}
              title={
                opRunning
                  ? "Wait for the active operation or setup session to finish."
                  : `Disconnect ${title}. It stays powered on.`
              }
            >
              <Unplug />
              Disconnect
            </Button>
          ) : (
            <Button
              variant={detected ? "default" : "outline"}
              onClick={() => onRobotConnect(profile)}
              disabled={!online || robotBusy || opRunning}
              title={
                opRunning ? "Wait for the active operation or setup session to finish." : undefined
              }
            >
              {robotBusy ? <Loader2 className="animate-spin" /> : <Plug />}
              Connect
            </Button>
          )
        }
        footer={
          online && (onOpenSettings || (selectedProfile && !selected && onSelectProfile)) ? (
            <>
              <SettingsButton
                onClick={onOpenSettings ? () => onOpenSettings(profile) : undefined}
              />
              {selectedProfile && !selected && onSelectProfile && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onSelectProfile(profile)}
                  disabled={selectDisabled}
                  title={selectDisabled ? (selectDisabledReason ?? undefined) : undefined}
                  className="ml-auto text-white/60"
                >
                  <Check />
                  Use for operations
                </Button>
              )}
            </>
          ) : undefined
        }
      >
        {live && robot && robot.motors.length > 0 ? <MotorSummary robot={robot} /> : undefined}
      </DeviceCard>
    )
  }

  // Jelly's wheels and lift: two independent idle links on their own
  // adapters, shown as two rows of one tile (each with its own status and
  // connect / disconnect). Neither takes part in the Axol / Mantis device
  // selection — Jelly is driven alongside Axol, not instead of it.
  const jellyRow = (device: JellyDevice) => {
    const title = JELLY_DEVICE_LABELS[device]
    const short = device === "wheels" ? "Wheels" : "Lift"
    const view = jellyDeviceView(device, jelly, canDevices?.[device], jellySupported)
    const busy = jellyBusy?.[device] ?? false
    const live = view.state === "connected" || view.state === "busy"
    const detail =
      jelly && view.state === "connected" && !view.fault ? (
        device === "wheels" ? (
          <WheelGrid status={jelly.wheels} />
        ) : (
          <span
            className="min-w-0 flex-1 truncate text-white/75"
            title={liftSummaryText(jelly.lift)}
          >
            {liftSummaryText(jelly.lift)}
          </span>
        )
      ) : (
        <span className="min-w-0 flex-1 truncate text-white/75" title={view.label}>
          {view.label}
        </span>
      )
    return (
      <div key={device} className="flex min-h-9 min-w-0 items-center gap-3 text-sm">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            DOT_CLASS[view.dot],
            view.state === "connecting" && "animate-pulse"
          )}
        />
        <span className="w-12 shrink-0 font-medium text-white/80">{short}</span>
        {detail}
        {live ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onJellyDisconnect?.(device)}
            disabled={busy || opRunning || view.state === "busy" || !onJellyDisconnect}
            title={
              opRunning || view.state === "busy"
                ? "Wait for the active operation or setup session to finish."
                : `Disconnect ${title}. It stays powered on.`
            }
            className="shrink-0"
          >
            {busy ? <Loader2 className="animate-spin" /> : <Unplug />}
            Disconnect
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onJellyConnect?.(device)}
            disabled={!online || !jellySupported || busy || opRunning || !onJellyConnect}
            title={
              !jellySupported
                ? "Update the software on the Host Computer to connect Jelly."
                : opRunning
                  ? "Wait for the active operation or setup session to finish."
                  : undefined
            }
            className="shrink-0"
          >
            {busy ? <Loader2 className="animate-spin" /> : <Plug />}
            Connect
          </Button>
        )}
      </div>
    )
  }
  // The card's headline dot: the worse of the two devices.
  const jellyViews = JELLY_DEVICES.map((device) =>
    jellyDeviceView(device, jelly, canDevices?.[device], jellySupported)
  )
  const jellyDot: Dot = jellyViews.some((v) => v.dot === "err")
    ? "err"
    : jellyViews.some((v) => v.dot === "warn")
      ? "warn"
      : jellyViews.some((v) => v.dot === "busy")
        ? "busy"
        : jellyViews.some((v) => v.dot === "ok")
          ? "ok"
          : "idle"
  const jellyLive = jellyViews.filter((v) => v.state === "connected" || v.state === "busy").length
  const jellyLabel = !jellySupported
    ? "Update the Host Computer to use Jelly"
    : jellyLive === 2
      ? "Wheels and lift connected"
      : jellyLive === 1
        ? "One of two connected"
        : jellyViews.some((v) => v.label === "Detected")
          ? "Detected"
          : "Disconnected"
  const battery = jellySupported ? jellyBattery(jelly) : null

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <DeviceCard
        icon={<Server />}
        name="Host Computer"
        dot={wsDot}
        status={wsLabel}
        pulse={conn === "loading"}
        action={
          online ? (
            <Button
              variant="outline"
              onClick={onHostDisconnect}
              title="Disconnect this panel. The Host Computer keeps running."
            >
              <Unplug />
              Disconnect
            </Button>
          ) : (
            <Button onClick={onOpenSetup}>
              <Plug />
              Connect
            </Button>
          )
        }
        footer={
          online ? (
            <>
              <SettingsButton
                onClick={onOpenSettings ? () => onOpenSettings("general") : undefined}
              />
              <span className="ml-auto" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPowerOpen("restart")}
                className="text-white/60"
              >
                <RotateCcw />
                Restart
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPowerOpen("shutdown")}
                className="text-red-300 hover:bg-red-400/10 hover:text-red-200"
              >
                <Power />
                Shut down
              </Button>
            </>
          ) : undefined
        }
      >
        {online && (host || version) ? (
          <div className="flex flex-col gap-2">
            {host && <Detail label="Address">{host}</Detail>}
            {version && <Detail label="Software">Axol v{version}</Detail>}
          </div>
        ) : undefined}
      </DeviceCard>

      {hardwareCard("axol", "Axol")}
      {hardwareCard("mantis", "Mantis")}

      <DeviceCard
        icon={<CircleDot />}
        name="Jelly"
        dot={jellyDot}
        status={jellyLabel}
        action={battery ? <BatteryIndicator battery={battery} /> : undefined}
        footer={
          online && onOpenSettings ? (
            <SettingsButton onClick={() => onOpenSettings("jelly")} />
          ) : undefined
        }
      >
        {jellySupported ? (
          <div className="flex flex-col gap-2">{JELLY_DEVICES.map(jellyRow)}</div>
        ) : undefined}
      </DeviceCard>

      {/* Host power confirmation (shutdown / restart) */}
      {powerOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setPowerOpen(null)}
        >
          <Card
            className="w-full max-w-sm gap-4 bg-[#1a1a1a] p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col gap-1">
              <h3 className="font-heading text-base font-semibold">
                {POWER_ACTIONS[powerOpen].title}
              </h3>
              <p className="text-sm leading-relaxed text-white/45">
                {POWER_ACTIONS[powerOpen].body}
              </p>
            </div>
            {opRunning ? (
              <p className="text-xs text-amber-200/70">An operation is running. Stop it first.</p>
            ) : (
              hostInstalling && (
                <p className="text-xs text-amber-200/70">
                  The robot is installing updates. Wait for it to finish.
                </p>
              )
            )}
            <div className="flex items-center justify-end gap-2 pt-1">
              <Button variant="ghost" size="sm" onClick={() => setPowerOpen(null)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => confirmPower(powerOpen)}
                disabled={powerBusy || opRunning || hostInstalling}
              >
                {powerBusy ? (
                  <Loader2 className="animate-spin" />
                ) : powerOpen === "shutdown" ? (
                  <Power />
                ) : (
                  <RotateCcw />
                )}{" "}
                {POWER_ACTIONS[powerOpen].button}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

/** The motor squares with how many are healthy, on a connected device's card. */
function MotorSummary({ robot }: { robot: RobotStatus }) {
  const known = robot.motors.filter((m) => m.reachable != null)
  const healthy = known.filter(motorHealthy).length
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-sm">
      <span className="text-white/45">
        Motors
        {known.length > 0 && (
          <span
            className={cn("ml-2", healthy < known.length ? "text-red-300/90" : "text-white/80")}
          >
            {healthy} of {robot.motors.length} OK
          </span>
        )}
      </span>
      <MotorGrid robot={robot} size="md" />
    </div>
  )
}

/** Healthy = reachable on CAN and reporting no error status. */
function motorHealthy(m: MotorHealth): boolean {
  return m.reachable === true && (m.status === "OK" || m.status === "DISABLED" || m.status == null)
}

const jointName = (m: MotorHealth) => `${m.arm} ${m.joint.replace(/_/g, " ").toLowerCase()}`

/**
 * Per-motor health as one status row. CAN reachability and the reported motor
 * error are one result: an unreachable motor is an error, not a separate state.
 */
export function MotorGrid({ robot, size = "sm" }: { robot: RobotStatus; size?: "sm" | "md" }) {
  if (!robot.motors.length) return null
  // Unknown is its own square: while a task owns the bus nothing reads these
  // motors, and painting the last-known state would keep showing a motor as
  // healthy long after its power was cut.
  const color = (m: MotorHealth) =>
    m.reachable == null ? "unknown" : motorHealthy(m) ? "ok" : "err"
  const tip = (m: MotorHealth) => {
    if (m.reachable == null) return `${jointName(m)}: unknown while in use`
    if (!m.reachable) return `${jointName(m)}: unreachable`
    const status = (m.status ?? "OK").replace(/_/g, " ").toLowerCase()
    const temp = m.temperature != null ? ` · ${Math.round(m.temperature)}°C` : ""
    return `${jointName(m)}: ${status}${temp}`
  }
  const arms = ["left", "right"]
  const SQUARE = {
    ok: "bg-emerald-400/80",
    err: "bg-red-400/70",
    unknown: "bg-white/25",
  }
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {arms.map((arm) => (
        <div key={arm} className="flex items-center gap-1">
          <span className="font-mono text-[0.6rem] text-white/35">{arm[0].toUpperCase()}</span>
          <div className={cn("flex", size === "md" ? "gap-1" : "gap-0.5")}>
            {robot.motors
              .filter((m) => m.arm === arm)
              .map((m, index, motors) => {
                const tooltip = `motor-${m.arm}-${m.joint}-tooltip`
                const first = index === 0
                const last = index === motors.length - 1
                return (
                  <span
                    key={m.joint}
                    tabIndex={0}
                    aria-describedby={tooltip}
                    className="group/motor relative inline-flex rounded-[3px] outline-none focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#111]"
                  >
                    <span
                      className={cn(
                        "rounded-[3px]",
                        size === "md" ? "size-3.5" : "size-2.5",
                        SQUARE[color(m)]
                      )}
                    />
                    <span
                      id={tooltip}
                      role="tooltip"
                      className={cn(
                        "pointer-events-none absolute top-full z-30 mt-2 w-max max-w-56 rounded-md border border-white/15 bg-[#181818] px-2.5 py-1.5 text-center text-xs leading-snug font-normal text-white/85 opacity-0 shadow-lg transition-opacity duration-75 group-hover/motor:opacity-100 group-focus-visible/motor:opacity-100",
                        first ? "left-0" : last ? "right-0" : "left-1/2 -translate-x-1/2"
                      )}
                    >
                      {tip(m)}
                      <span
                        className={cn(
                          "absolute bottom-full size-1.5 translate-y-1/2 rotate-45 border-t border-l border-white/15 bg-[#181818]",
                          first ? "left-1" : last ? "right-1" : "left-1/2 -translate-x-1/2"
                        )}
                      />
                    </span>
                  </span>
                )
              })}
          </div>
        </div>
      ))}
    </div>
  )
}
