import { Cpu, Loader2, Plug, Power, RotateCcw, Server, Settings2, Unplug } from "lucide-react"
import { useCallback, useState, type ReactNode } from "react"
import type { ConnState } from "@/components/setup-dialog"
import type { SettingsScope } from "@/lib/settings-scope"
import {
  restartHost,
  shutdownHost,
  type CanProfileInventory,
  type HardwareProfile,
  type MotorHealth,
  type RobotStatus,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
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
  statusEnd,
  statusContent,
  badge,
  onOpenSettings,
}: {
  icon: ReactNode
  title: string
  dot: Dot
  label: string
  pulse?: boolean
  children?: ReactNode
  statusEnd?: ReactNode
  statusContent?: ReactNode
  /** Small marker next to the title (e.g. the selected device). */
  badge?: ReactNode
  /** Clicking the tile's title opens this connection's settings. */
  onOpenSettings?: () => void
}) {
  const heading = (
    <>
      {icon}
      <span className="font-mono">{title}</span>
    </>
  )
  return (
    <div
      className={cn(
        "group relative flex h-fit min-w-0 flex-col gap-2 overflow-visible rounded-xl border border-white/10 bg-white/[0.02] p-3.5",
        onOpenSettings && "transition-colors hover:border-white/20"
      )}
    >
      <div className="flex min-h-8 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {onOpenSettings ? (
            <button
              type="button"
              onClick={onOpenSettings}
              title={`Open ${title} settings`}
              aria-label={`Open ${title} settings`}
              className="flex items-center gap-2 rounded-md text-xs tracking-widest text-white/40 uppercase transition-colors hover:text-white/80"
            >
              {heading}
              <Settings2 className="size-3.5 text-white/30 transition-colors group-hover:text-white/60" />
            </button>
          ) : (
            <div className="flex items-center gap-2 text-xs tracking-widest text-white/40 uppercase">
              {heading}
            </div>
          )}
          {badge}
        </div>
        {children && <div className="shrink-0">{children}</div>}
      </div>
      {statusContent ?? (
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <span
            className={cn("size-2 shrink-0 rounded-full", DOT_CLASS[dot], pulse && "animate-pulse")}
          />
          <span className="min-w-0 flex-1 truncate text-white/75" title={label}>
            {label}
          </span>
          {statusEnd && <div className="shrink-0">{statusEnd}</div>}
        </div>
      )}
    </div>
  )
}

/** The two host power actions offered on the connected Axol Host tile. */
type PowerAction = "shutdown" | "restart"

const POWER_ACTIONS: Record<
  PowerAction,
  { title: string; command: string; body: string; button: string; done: string }
> = {
  shutdown: {
    title: "Shut down host",
    command: "shutdown -h now",
    body:
      "This panel loses its connection immediately, and turning the robot " +
      "back on requires physical access to it.",
    button: "Shut down",
    done: "Host is shutting down.",
  },
  restart: {
    title: "Restart host",
    command: "shutdown -r now",
    body:
      "This panel loses its connection while the host reboots — reconnect " +
      "once it's back up (typically under a minute).",
    button: "Restart",
    done: "Host is restarting.",
  },
}

const PROFILE_LABEL: Record<HardwareProfile, string> = { axol: "Axol", mantis: "Mantis" }

/**
 * The system-wide device selection: which hardware every operation runs on.
 * One switch for the whole panel (persisted with the shared settings) instead
 * of a per-operation Mantis toggle — teleop and data collection follow it, and
 * Axol-only operations wait until it is back on Axol.
 */
export function DeviceSwitch({
  value,
  onChange,
  disabled = false,
  disabledReason,
  saving = false,
}: {
  value: HardwareProfile
  onChange: (profile: HardwareProfile) => void
  disabled?: boolean
  disabledReason?: string | null
  /** The selection is being written to the host. */
  saving?: boolean
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.02] px-3.5 py-2.5">
      <div className="flex min-w-0 flex-col">
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">Device</span>
        <span className="text-xs text-white/45">
          Every operation runs on the selected hardware; Axol-only operations wait for Axol.
        </span>
      </div>
      <div className="flex items-center gap-2">
        {saving && <Loader2 className="size-3.5 animate-spin text-white/40" />}
        <div
          role="radiogroup"
          aria-label="Device"
          title={disabled ? (disabledReason ?? undefined) : undefined}
          className="flex rounded-lg border border-white/10 bg-white/[0.02] p-0.5"
        >
          {(["axol", "mantis"] as const).map((profile) => {
            const active = profile === value
            return (
              <button
                key={profile}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={disabled}
                onClick={() => {
                  if (!active) onChange(profile)
                }}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                  active
                    ? "bg-[#eff483]/15 text-[#eff483]"
                    : "text-white/60 hover:bg-white/[0.05] hover:text-white/85"
                )}
              >
                <Cpu className="size-3.5" />
                {PROFILE_LABEL[profile]}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/**
 * Connection tiles for the Axol Host and the two hardware profiles. Axol and
 * Mantis share one idle telemetry link, so connecting either hardware tile
 * switches that link to its CAN interfaces and motor set. Clicking a tile's
 * title opens that connection's settings: Axol and Mantis each have their
 * own, and the host tile opens the general (shared) settings.
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
  robot,
  robotBusy,
  canProfiles,
  onRobotConnect,
  onRobotDisconnect,
  selectedProfile,
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
  robot: RobotStatus | null
  robotBusy: boolean
  /** Configured profiles whose CAN netdevs or exact persisted USB hub exist. */
  canProfiles?: CanProfileInventory | null
  onRobotConnect: (profile: HardwareProfile) => void
  onRobotDisconnect: () => void
  /** The system-wide device selection, marked on its tile. */
  selectedProfile?: HardwareProfile
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

  // -- axol host --
  const wsDot: Dot =
    conn === "ok" ? "ok" : conn === "err" ? "err" : conn === "idle" ? "idle" : "warn"
  const wsLabel =
    conn === "ok"
      ? hostName || host || "Connected"
      : conn === "err"
        ? "Offline"
        : conn === "migration"
          ? "Installer migration required"
          : conn === "idle"
            ? "Not connected"
            : "Connecting…"

  // Axol and Mantis are two profiles of the same server-owned telemetry link.
  // Older hosts omit profile and are necessarily the original Axol profile.
  const activeProfile = robot?.profile ?? "axol"
  const hardwareTile = (profile: HardwareProfile, title: string) => {
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

    return (
      <Tile
        icon={<Cpu className="size-3.5" />}
        title={title}
        dot={dot}
        label={label}
        pulse={state === "connecting"}
        badge={
          selectedProfile === profile ? (
            <span
              className="shrink-0 rounded-full bg-[#eff483]/15 px-1.5 py-0.5 font-mono text-[0.6rem] tracking-wider text-[#eff483] uppercase"
              title="Operations run on this device"
            >
              selected
            </span>
          ) : undefined
        }
        onOpenSettings={online && onOpenSettings ? () => onOpenSettings(profile) : undefined}
        statusContent={
          active &&
          robot &&
          robot.motors.length > 0 &&
          (state === "connected" || state === "busy") ? (
            <MotorGrid robot={robot} />
          ) : undefined
        }
      >
        {active && (state === "connected" || state === "busy") ? (
          <Button
            variant="outline"
            size="icon"
            onClick={onRobotDisconnect}
            disabled={robotBusy || opRunning}
            aria-label={`Disconnect ${title}`}
            title={
              opRunning
                ? "Wait for the active operation or setup session to finish."
                : `Release the ${title} link (CAN). The hardware stays powered.`
            }
            className="size-8"
          >
            <Unplug />
          </Button>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => onRobotConnect(profile)}
            disabled={!online || robotBusy || opRunning}
            title={
              opRunning ? "Wait for the active operation or setup session to finish." : undefined
            }
          >
            {robotBusy ? <Loader2 className="animate-spin" /> : <Plug />}
            Connect
          </Button>
        )}
      </Tile>
    )
  }

  return (
    <div className="grid grid-cols-1 items-start gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <Tile
        icon={<Server className="size-3.5" />}
        title="Axol Host"
        dot={wsDot}
        label={wsLabel}
        pulse={conn === "loading"}
        onOpenSettings={online && onOpenSettings ? () => onOpenSettings("general") : undefined}
        statusEnd={
          online && version ? (
            <span className="font-mono text-[0.7rem] text-white/35" title={`v${version}`}>
              axol v{version}
            </span>
          ) : undefined
        }
      >
        {online ? (
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setPowerOpen("restart")}
              aria-label="Restart host"
              title="Reboot the robot host (shutdown -r now)."
              className="size-8"
            >
              <RotateCcw />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setPowerOpen("shutdown")}
              aria-label="Shut down host"
              title="Power off the robot host (shutdown -h now)."
              className="size-8 text-red-300 hover:bg-red-400/10"
            >
              <Power />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={onHostDisconnect}
              aria-label="Disconnect Axol Host"
              title="Disconnect this panel from the host. The host keeps running."
              className="size-8"
            >
              <Unplug />
            </Button>
          </div>
        ) : (
          <Button variant="outline" size="sm" onClick={onOpenSetup}>
            <Plug />
            Connect
          </Button>
        )}
      </Tile>

      {hardwareTile("axol", "Axol")}
      {hardwareTile("mantis", "Mantis")}

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
                {powerOpen === "shutdown" ? "Power off" : "Reboot"} the robot host (
                <span className="font-mono">{POWER_ACTIONS[powerOpen].command}</span>).{" "}
                {POWER_ACTIONS[powerOpen].body}
              </p>
            </div>
            {opRunning && (
              <p className="text-xs text-amber-200/70">
                A run or operation is in flight — stop it first.
              </p>
            )}
            <div className="flex items-center justify-end gap-2 pt-1">
              <Button variant="ghost" size="sm" onClick={() => setPowerOpen(null)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => confirmPower(powerOpen)}
                disabled={powerBusy || opRunning}
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

/** Healthy = reachable on CAN and reporting no error status. */
function motorHealthy(m: MotorHealth): boolean {
  return m.reachable && (m.status === "OK" || m.status === "DISABLED" || m.status == null)
}

const jointName = (m: MotorHealth) => `${m.arm} ${m.joint.replace(/_/g, " ").toLowerCase()}`

/**
 * Per-motor health as one status row. CAN reachability and the reported motor
 * error are one result: an unreachable motor is an error, not a separate state.
 */
export function MotorGrid({ robot }: { robot: RobotStatus }) {
  if (!robot.motors.length) return null
  const color = (m: MotorHealth) => (motorHealthy(m) ? "ok" : "err")
  const tip = (m: MotorHealth) => {
    if (!m.reachable) return `${jointName(m)}: unreachable`
    const status = (m.status ?? "OK").replace(/_/g, " ").toLowerCase()
    const temp = m.temperature != null ? ` · ${Math.round(m.temperature)}°C` : ""
    return `${jointName(m)}: ${status}${temp}`
  }
  const arms = ["left", "right"]
  const SQUARE = {
    ok: "bg-emerald-400/80",
    err: "bg-red-400/70",
  }
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {arms.map((arm) => (
        <div key={arm} className="flex items-center gap-1">
          <span className="font-mono text-[0.6rem] text-white/35">{arm[0].toUpperCase()}</span>
          <div className="flex gap-0.5">
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
                    <span className={cn("size-2.5 rounded-[3px]", SQUARE[color(m)])} />
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
