import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { Link } from "react-router-dom"
import {
  ArrowDown,
  ArrowUp,
  Hand,
  Loader2,
  OctagonX,
  Play,
  RotateCcw,
  RotateCw,
  Square,
  Undo2,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectOption } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { AppShell } from "@/components/app-shell"
import { OperationIcon } from "@/components/operation-launcher"
import { DrivePad, HoldButton } from "@/components/jog/hold-controls"
import { RobotStatus } from "@/components/jog/robot-status"
import { MarionetteStage } from "@/components/marionette/stage"
import { ROSTER } from "@/lib/axol-roster"
import { useHostSummary } from "@/lib/host-summary"
import { catalogEntry, rememberOperation } from "@/lib/operation-catalog"
import { cn } from "@/lib/utils"
import {
  IDLE_HOLD,
  storedSimRobot,
  storeSimRobot,
  useJogLink,
  type JogArm,
  type JogHold,
  type JogState,
  type Vec3,
} from "@/lib/jog"
import {
  CALIBRATION_LABELS,
  DEFAULT_GATEWAY,
  GATEWAY_STORAGE_KEY,
  TRACKING_LABELS,
  desktopHold,
  driveKeyOf,
  fetchMarionettePairing,
  frameLabel,
  isTextEntry,
  phoneActivity,
  phoneConnectHint,
  qrImageSource,
  resetMarionetteArm,
  setMarionetteStop,
  type DriveKey,
  type MarionetteArm,
  type MarionettePairing,
  type MarionettePhone,
  type MarionetteState,
} from "@/lib/marionette"
import {
  fetchOpStatus,
  fetchSettings,
  parseAxolModel,
  parseGripper,
  setServerBase,
  startOperation,
  stopOperation,
  wsBaseUrl,
  type AxolModel,
  type GripperType,
  type OpStatus,
} from "@/lib/supervisor"

const JogViewer = lazy(() => import("@/components/jog/jog-viewer"))

const OPERATION = catalogEntry("marionette")

type Mode = "sim" | "hardware"

const ARMS: readonly JogArm[] = ["left", "right"]
const ignore = () => undefined

/** The key legend, laid out like the keyboard. */
const KEY_ROWS: { key: DriveKey; action: string }[][] = [
  [
    { key: "q", action: "Turn left" },
    { key: "w", action: "Forward" },
    { key: "e", action: "Turn right" },
    { key: "z", action: "Raise lift" },
  ],
  [
    { key: "a", action: "Left" },
    { key: "s", action: "Back" },
    { key: "d", action: "Right" },
    { key: "x", action: "Lower lift" },
  ],
]

function storedGateway(): string {
  try {
    return localStorage.getItem(GATEWAY_STORAGE_KEY) ?? DEFAULT_GATEWAY
  } catch {
    return DEFAULT_GATEWAY
  }
}

// The side panels are denser than the app's default cards.
const PANEL = "gap-3 p-4"
// How often the pairing details are read again: a restarted gateway, or one
// that went away, shows within this long.
const PAIRING_REFRESH_MS = 15000

/**
 * The QR code and pairing code of the gateway at `address`, or null while
 * they can't be read (no gateway there, say). A changed address is read
 * after a pause, since typing in Gateway changes it on every key.
 */
function usePairing(address: string): MarionettePairing | null {
  // Kept with the address it was read for, so another gateway's code never shows.
  const [read, setRead] = useState<{ address: string; pairing: MarionettePairing | null }>()
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      let pairing: MarionettePairing | null = null
      try {
        pairing = await fetchMarionettePairing(address)
      } catch {
        // No gateway there, or it can't say: the page shows the address only.
      }
      if (cancelled) return
      setRead({ address, pairing })
      timer = setTimeout(load, PAIRING_REFRESH_MS)
    }
    timer = setTimeout(load, 400)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [address])
  return read?.address === address ? read.pairing : null
}

type Tone = "neutral" | "good" | "warn" | "stop" | "active"
const TONES: Record<Tone, string> = {
  neutral: "border-white/10 text-white/55",
  good: "border-emerald-400/25 text-emerald-300",
  warn: "border-amber-400/30 text-amber-300",
  stop: "border-red-400/40 bg-red-400/10 text-red-300",
  active: "border-[#eff483]/40 bg-[#eff483]/10 text-[#eff483]",
}

function Chip({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={cn("rounded border px-1.5 py-0.5 text-[11px] leading-tight", TONES[tone])}>
      {children}
    </span>
  )
}

function Dot({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="size-2.5 shrink-0 rounded-full ring-1 ring-white/20"
      style={{ backgroundColor: color }}
    />
  )
}

function PhoneRow({
  phone,
  arms,
}: {
  phone: MarionettePhone
  arms: MarionetteState["arms"] | null
}) {
  return (
    <li
      data-testid={`phone-${phone.id}`}
      className={cn(
        "flex flex-col gap-1.5 py-2.5 first:pt-0 last:pb-0",
        !phone.connected && "opacity-60"
      )}
    >
      <div className="flex items-center gap-2">
        <Dot color={phone.color} />
        <span className="text-sm font-medium">{phone.name}</span>
        <span
          className={cn("ml-auto text-xs", phone.connected ? "text-emerald-300" : "text-white/45")}
        >
          {phone.connected ? "Connected" : "Offline"}
        </span>
      </div>
      {phone.connected && (
        <>
          <p className="text-xs text-white/75">{phoneActivity(phone, arms)}</p>
          <div className="flex flex-wrap gap-1">
            <Chip>{phone.arm === "left" ? "Left arm" : "Right arm"}</Chip>
            <Chip tone={phone.frame === "axol" ? "neutral" : "active"}>
              {frameLabel(phone.frame)}
            </Chip>
            <Chip tone={phone.tracking === "normal" ? "good" : "warn"}>
              {TRACKING_LABELS[phone.tracking] ?? phone.tracking}
            </Chip>
            <Chip tone={phone.calibration === "captured" ? "good" : "warn"}>
              {CALIBRATION_LABELS[phone.calibration] ?? phone.calibration}
            </Chip>
            <Chip tone={phone.deadman ? "active" : "neutral"}>
              {phone.deadman ? "Holding" : "Not holding"}
            </Chip>
            {phone.stop && <Chip tone="stop">Stop on</Chip>}
            {phone.source === "synthetic" && <Chip>Simulated</Chip>}
          </div>
        </>
      )}
    </li>
  )
}

function PhonesCard({
  phones,
  arms,
  hint,
  pairing,
}: {
  phones: MarionettePhone[]
  arms: MarionetteState["arms"] | null
  /** Where the Marionette app connects. */
  hint: string
  /** The gateway's QR code and pairing code, once read. */
  pairing: MarionettePairing | null
}) {
  return (
    <Card data-testid="marionette-phones" className={PANEL}>
      <CardHeader>
        <CardTitle>Phones</CardTitle>
        {pairing ? (
          <div data-testid="marionette-pairing" className="flex items-start gap-3 pt-1">
            <img
              src={qrImageSource(pairing.qrSvg)}
              alt="QR code to pair a phone"
              className="size-32 shrink-0 rounded-md bg-white p-1.5"
            />
            <div className="flex min-w-0 flex-col gap-1.5 text-xs text-white/60">
              <p>
                To pair a phone, scan this with its camera, or tap{" "}
                <strong className="text-white/80">Scan QR code</strong> in Marionette.
              </p>
              <p data-testid="marionette-connect">{hint}</p>
              {pairing.code ? (
                <p>
                  Pairing code{" "}
                  <span
                    data-testid="marionette-pairing-code"
                    className="font-mono text-sm whitespace-nowrap text-white/90"
                  >
                    {pairing.code}
                  </span>
                </p>
              ) : (
                <p>Pairing is off, so phones need no code.</p>
              )}
            </div>
          </div>
        ) : (
          <CardDescription data-testid="marionette-connect" className="text-xs">
            {hint}
          </CardDescription>
        )}
      </CardHeader>
      {phones.length === 0 ? (
        <p className="text-sm text-white/45">No phones connected.</p>
      ) : (
        <ul className="flex flex-col divide-y divide-white/10">
          {phones.map((phone) => (
            <PhoneRow key={phone.id} phone={phone} arms={arms} />
          ))}
        </ul>
      )}
    </Card>
  )
}

function ArmRow({
  side,
  arm,
  phones,
  canReset,
}: {
  side: JogArm
  arm: MarionetteArm | null
  phones: MarionettePhone[]
  /** Marionette runs, Stop is off and no arm is resetting. */
  canReset: boolean
}) {
  const owner = arm?.owner ? phones.find((p) => p.id === arm.owner) : undefined
  const closed = arm ? Math.max(0, Math.min(1, arm.grip)) : null
  const [sending, setSending] = useState(false)
  const [refusal, setRefusal] = useState<string | null>(null)
  useEffect(() => {
    if (!refusal) return
    const timer = setTimeout(() => setRefusal(null), 6000)
    return () => clearTimeout(timer)
  }, [refusal])
  const reset = async () => {
    setSending(true)
    setRefusal(null)
    try {
      await resetMarionetteArm(side)
    } catch (reason) {
      setRefusal(
        reason instanceof TypeError
          ? "Can't reach the Host Computer."
          : reason instanceof Error
            ? reason.message
            : String(reason)
      )
    } finally {
      setSending(false)
    }
  }
  return (
    <div data-testid={`arm-${side}`} className="flex min-w-0 flex-col gap-2">
      <div className="flex flex-col items-start gap-1">
        <span className="text-sm font-medium">{side === "left" ? "Left arm" : "Right arm"}</span>
        <Chip tone={arm?.engaged || arm?.resetting ? "active" : "neutral"}>
          {arm?.resetting ? "Returning to rest" : arm?.engaged ? "Following" : "Not following"}
        </Chip>
      </div>
      <div className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-1 text-xs">
        <span className="text-white/50">Phone</span>
        <span className="flex min-w-0 items-center justify-end gap-1.5 truncate text-white/80">
          {owner && <Dot color={owner.color} />}
          {arm?.ownerName ?? (arm?.owner ? arm.owner : "None")}
        </span>
        <span className="text-white/50">Gripper</span>
        <span className="text-right font-mono text-white/80 tabular-nums">
          {closed != null ? `${Math.round(closed * 100)} % closed` : "–"}
        </span>
      </div>
      {closed != null && (
        <div className="h-1 overflow-hidden rounded-full bg-white/10" aria-hidden>
          <div className="h-full bg-[#eff483]/70" style={{ width: `${closed * 100}%` }} />
        </div>
      )}
      {arm?.limited && (
        <p className="text-xs text-amber-300" role="status">
          <Hand className="mr-1 inline size-3.5" />
          At reach limit
        </p>
      )}
      {arm?.reason && <p className="text-xs text-white/60">{arm.reason}</p>}
      <Button
        variant="outline"
        size="sm"
        className="self-start"
        title={`Move the ${side} arm back to its rest pose.`}
        disabled={!canReset || sending}
        onClick={() => void reset()}
      >
        {sending ? <Loader2 className="animate-spin" /> : <Undo2 />}
        Reset arm
      </Button>
      {refusal && (
        <p className="text-xs text-amber-300" role="alert">
          {refusal}
        </p>
      )}
    </div>
  )
}

function KeyLegend({ pressed, active }: { pressed: ReadonlySet<DriveKey>; active: boolean }) {
  return (
    <div className={cn("flex flex-col gap-1.5", !active && "opacity-50")} aria-label="Drive keys">
      {KEY_ROWS.map((row, i) => (
        <div key={i} className="grid grid-cols-4 gap-1.5">
          {row.map(({ key, action }) => {
            const down = active && pressed.has(key)
            return (
              <div
                key={key}
                data-testid={`key-${key}`}
                data-pressed={down}
                className={cn(
                  "flex flex-col items-center gap-0.5 rounded-md border px-1 py-1.5 transition-colors",
                  key === "z" || key === "x" ? "ml-1.5" : "",
                  down
                    ? "border-[#eff483]/70 bg-[#eff483]/20 text-[#eff483]"
                    : "border-white/15 bg-white/[0.03] text-white/80"
                )}
              >
                <kbd className="font-mono text-sm font-semibold">{key.toUpperCase()}</kbd>
                <span className={cn("text-[10px]", down ? "text-[#eff483]/80" : "text-white/45")}>
                  {action}
                </span>
              </div>
            )
          })}
        </div>
      ))}
      <p className="flex items-center gap-2 text-[11px] text-white/45">
        <kbd className="rounded border border-white/15 bg-white/[0.03] px-1.5 py-0.5 font-mono text-[11px] text-white/80">
          Esc
        </kbd>
        Turns Stop on.
      </p>
    </div>
  )
}

export default function Marionette() {
  const [simPick] = useState(storedSimRobot)
  const [mode, setMode] = useState<Mode>("sim")
  const [simModel, setSimModel] = useState<AxolModel>(simPick?.model ?? "mobile")
  const [simGripper] = useState<GripperType>(simPick?.gripper ?? "parallel")
  const [host, setHost] = useState<{ model: AxolModel; gripper: GripperType } | null>(null)
  const [hostError, setHostError] = useState<string | null>(null)
  const [opStatus, setOpStatus] = useState<OpStatus | null>(null)
  // Whether the Host Computer answered the last status poll; null until it does.
  const [reachable, setReachable] = useState<boolean | null>(null)
  const hostSummary = useHostSummary(reachable)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [gateway, setGateway] = useState(storedGateway)
  const [driveSpeed, setDriveSpeed] = useState(0.5)
  const [pressed, setPressed] = useState<ReadonlySet<DriveKey>>(() => new Set())

  // Keys and on-screen controls merge into one held command, sent by the
  // jog link every 50 ms. Arms are the phones' alone: no arm jog goes out.
  const holdRef = useRef<JogHold>({ ...IDLE_HOLD })
  const keys = useRef(new Set<DriveKey>())
  const pad = useRef<Vec3>([0, 0, 0])
  const turn = useRef<-1 | 0 | 1>(0)
  const lift = useRef<-1 | 0 | 1>(0)
  const speed = useRef(driveSpeed)
  // Nothing but an idle hold goes out unless Marionette runs, even from a
  // drag that began before it stopped.
  const armed = useRef(false)
  const pushHold = useCallback(() => {
    holdRef.current = armed.current
      ? desktopHold(
          keys.current,
          { pad: pad.current, turn: turn.current, lift: lift.current },
          speed.current
        )
      : { ...IDLE_HOLD }
  }, [])
  useEffect(() => {
    speed.current = driveSpeed
    pushHold()
  }, [driveSpeed, pushHold])

  const { state, connected, error: linkError } = useJogLink(true, holdRef)

  useEffect(() => {
    setServerBase(localStorage.getItem("axolServerHost") ?? "")
    fetchSettings()
      .then((settings) =>
        setHost({
          model: settings.robotModel ?? "classic",
          gripper: parseGripper(settings.values["axol.gripper"]),
        })
      )
      .catch(() =>
        setHostError("Can't reach the Host Computer. Connect to it on the Devices page.")
      )
  }, [])

  useEffect(() => rememberOperation("marionette"), [])

  useEffect(() => {
    let cancelled = false
    const poll = () =>
      fetchOpStatus()
        .then((status) => {
          if (cancelled) return
          setOpStatus(status)
          setReachable(true)
        })
        .catch(() => !cancelled && setReachable(false))
    poll()
    const timer = setInterval(poll, 1000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const session = opStatus?.running ? opStatus.session : null
  const ownSession = session?.command === "marionette" ? session : null
  const otherOperation = session && !ownSession ? session.command : null
  const running = Boolean(ownSession)
  const live: JogState | null = running && state?.running ? state : null
  const ready = Boolean(live?.ready)
  const sessionMode: Mode | null = live?.mode ?? null
  // The robot drawn is what the session runs, or what Start would run.
  const viewMode: Mode = sessionMode ?? mode
  const sessionModel = parseAxolModel(String(ownSession?.args?.["kinematics.robot_model"] ?? ""))
  const robotModel: AxolModel =
    viewMode === "sim"
      ? running
        ? (sessionModel ?? simModel)
        : simModel
      : (host?.model ?? "classic")
  const gripper: GripperType = viewMode === "sim" ? simGripper : (host?.gripper ?? "parallel")
  const marionette = live?.marionette ?? null
  const stop = marionette?.stop ?? null
  const phones = marionette?.phones ?? []
  const arms = marionette?.arms ?? null
  // The Host Computer as this page reaches it (the page may come from elsewhere).
  const axolHost = new URL(wsBaseUrl()).hostname
  const address = running
    ? marionette?.gateway.address || String(ownSession?.args?.gateway ?? "") || gateway
    : gateway
  const pairing = usePairing(address)
  const canReset =
    ready && !stop?.on && !ARMS.some((side) => arms?.[side]?.resetting) && Boolean(arms)
  // The view draws the robot only: no target, plan or arm markers.
  const viewState = useMemo(() => (live ? { ...live, tips: null } : null), [live])

  const setStop = useCallback(async (on: boolean) => {
    try {
      await setMarionetteStop(on)
    } catch (reason) {
      const detail =
        reason instanceof TypeError
          ? "Can't reach the Host Computer."
          : reason instanceof Error
            ? reason.message
            : String(reason)
      setError(`Couldn't turn Stop ${on ? "on" : "off"}. ${detail}`)
    }
  }, [])

  // The keys drive only while Marionette runs and nobody is typing. Losing
  // focus, hiding the page or stopping lets go of everything, on-screen
  // controls included: a missed key-up must never leave the base running.
  useEffect(() => {
    if (!running) return
    armed.current = true
    const held = keys.current
    const sync = () => {
      setPressed(new Set(held))
      pushHold()
    }
    const releaseAll = () => {
      held.clear()
      pad.current = [0, 0, 0]
      turn.current = 0
      lift.current = 0
      sync()
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextEntry(event.target)) return
      // A shortcut may hide the key-up (Cmd on a Mac), so it lets go instead.
      if (event.ctrlKey || event.metaKey || event.altKey) {
        if (held.size) releaseAll()
        return
      }
      if (event.key === "Escape") {
        event.preventDefault()
        if (!event.repeat) void setStop(true)
        return
      }
      const key = driveKeyOf(event)
      if (!key) return
      event.preventDefault()
      if (event.repeat || held.has(key)) return
      held.add(key)
      sync()
    }
    const onKeyUp = (event: KeyboardEvent) => {
      const key = driveKeyOf(event)
      if (!key || !held.has(key)) return
      held.delete(key)
      sync()
    }
    window.addEventListener("keydown", onKeyDown)
    window.addEventListener("keyup", onKeyUp)
    window.addEventListener("blur", releaseAll)
    document.addEventListener("visibilitychange", releaseAll)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
      window.removeEventListener("blur", releaseAll)
      document.removeEventListener("visibilitychange", releaseAll)
      armed.current = false
      releaseAll()
    }
  }, [running, pushHold, setStop])

  const run = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label)
    setError(null)
    try {
      await action()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(null)
    }
  }

  const start = () =>
    run("start", async () => {
      const args: Record<string, string | boolean> = {
        sim: mode === "sim",
        gateway: gateway.trim() || DEFAULT_GATEWAY,
      }
      if (mode === "sim") {
        args["kinematics.robot_model"] = simModel
        args["kinematics.gripper"] = simGripper
      }
      await startOperation("marionette", args)
      setOpStatus(await fetchOpStatus())
    })
  const stopRun = () =>
    run("stop", async () => {
      await stopOperation()
      setOpStatus(await fetchOpStatus())
    })

  const changeGateway = (value: string) => {
    setGateway(value)
    try {
      localStorage.setItem(GATEWAY_STORAGE_KEY, value)
    } catch {
      // The address lasts until the page closes.
    }
  }

  const liftState = live?.lift ?? null
  const baseState = live?.base ?? null
  const status = !connected
    ? { variant: "neutral" as const, text: "Offline" }
    : running
      ? ready
        ? {
            variant: sessionMode === "hardware" ? ("warning" as const) : ("success" as const),
            text: sessionMode === "hardware" ? "Robot" : "Sim",
          }
        : { variant: "neutral" as const, text: "Starting" }
      : { variant: "neutral" as const, text: "Stopped" }

  const stopHeadline = running && stop?.on ? `Stop is on${stop.by ? ` (${stop.by})` : ""}.` : null
  // A phone that dropped off with Stop on can't turn it off; the page can.
  const offlineStop = Boolean(
    running && stop?.on && !stop.desktop && stop.by?.endsWith("(offline)")
  )
  const stopNote: ReactNode =
    running && stop?.on ? (
      offlineStop ? (
        "That phone is offline. Press Clear Stop once it's safe to carry on."
      ) : stop.desktop ? (
        <>
          To move again, turn every Stop off, then press{" "}
          <strong className="text-white/80">Hold to move</strong> on the phone.
        </>
      ) : (
        "Turn it off on that phone to move again."
      )
    ) : (
      "Holds both arms where they are and stops the base and lift."
    )

  return (
    // On a wide screen the page fills the window: the stage takes what the
    // header and Stop leave, and the side panels scroll on their own.
    <AppShell
      title="Operations"
      back="/control/operations"
      fill
      badges={{ "/control/operations": session ? "running" : undefined }}
      host={hostSummary}
    >
      <section
        className={cn(
          "flex flex-wrap items-center gap-x-6 gap-y-3 rounded-2xl border p-4",
          running
            ? "border-emerald-400/25 bg-gradient-to-br from-emerald-400/[0.06] to-transparent"
            : "border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent"
        )}
      >
        <div className="flex min-w-64 flex-1 items-center gap-3">
          <OperationIcon entry={OPERATION} running={running} />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="font-heading text-lg font-semibold">{OPERATION.label}</h2>
              <Badge variant={status.variant}>{status.text}</Badge>
            </div>
            <p className="mt-0.5 text-sm text-white/55">{OPERATION.summary}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-white/55">
            Gateway
            <Input
              aria-label="Gateway address"
              className="w-44"
              value={running ? address : gateway}
              placeholder={DEFAULT_GATEWAY}
              disabled={running}
              spellCheck={false}
              autoComplete="off"
              onChange={(event) => changeGateway(event.target.value)}
            />
          </label>
          <div
            className="flex rounded-lg border border-white/10 bg-white/[0.03] p-1"
            role="radiogroup"
            aria-label="Mode"
          >
            {(["sim", "hardware"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={viewMode === value}
                disabled={running}
                onClick={() => setMode(value)}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed",
                  viewMode === value
                    ? "bg-white/10 text-white"
                    : "text-white/50 hover:text-white/80"
                )}
              >
                {value === "sim" ? "Sim" : "Robot"}
              </button>
            ))}
          </div>
          {viewMode === "sim" && (
            <Select
              aria-label="Robot model"
              className="w-48"
              value={robotModel}
              disabled={running}
              onChange={(event) => {
                const next = event.target.value as AxolModel
                setSimModel(next)
                storeSimRobot({ model: next, gripper: simGripper })
              }}
            >
              {Object.values(ROSTER).map(({ model, name }) => (
                <SelectOption key={model} value={model} label={name} />
              ))}
            </Select>
          )}
          {running ? (
            <Button variant="outline" onClick={stopRun} disabled={busy === "stop"}>
              {busy === "stop" ? <Loader2 className="animate-spin" /> : <Square />}
              Stop Marionette
            </Button>
          ) : (
            <Button
              className="min-w-32"
              onClick={start}
              disabled={Boolean(busy) || Boolean(otherOperation) || !connected}
            >
              {busy === "start" ? <Loader2 className="animate-spin" /> : <Play />}
              {mode === "sim" ? "Start sim" : "Start on robot"}
            </Button>
          )}
        </div>
      </section>

      {(error || linkError || hostError || otherOperation || live?.solverError) && (
        <div
          role="alert"
          className="rounded-lg border border-amber-400/25 bg-amber-400/[0.06] px-4 py-2.5 text-sm text-amber-200"
        >
          {otherOperation ? (
            <>
              {catalogEntry(otherOperation).label} is running. Stop it to use Marionette.{" "}
              <Link
                to={`/control/operations/${encodeURIComponent(otherOperation)}`}
                className="font-medium underline underline-offset-2"
              >
                Open {catalogEntry(otherOperation).label}
              </Link>
            </>
          ) : (
            (error ?? live?.solverError ?? linkError ?? hostError)
          )}
        </div>
      )}

      <div
        data-testid="marionette-stop"
        className={cn(
          "flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border px-3 py-2",
          stop?.on ? "border-red-400/40 bg-red-500/[0.08]" : "border-white/10 bg-white/[0.03]"
        )}
      >
        <Button
          data-state={stop?.desktop ? "on" : "off"}
          disabled={!running}
          onClick={(event) => {
            // The second click of a double-click must not turn Stop back off.
            if (event.detail > 1) return
            void setStop(offlineStop ? false : !stop?.desktop)
          }}
          className={cn(
            "h-10 min-w-40 font-semibold",
            stop?.desktop
              ? "border-red-400/50 bg-transparent text-red-200 hover:bg-red-500/10"
              : "bg-red-600 text-white hover:bg-red-600/85"
          )}
        >
          <OctagonX />
          {offlineStop ? "Clear Stop" : stop?.desktop ? "Turn Stop off" : "Stop motion"}
        </Button>
        <div className="min-w-0 flex-1 text-sm">
          {stopHeadline && <p className="font-medium text-red-200">{stopHeadline}</p>}
          <p className="text-white/50">
            {!running && mode === "hardware" && !otherOperation ? (
              <>
                On the robot, an arm moves only while{" "}
                <strong className="text-white/80">Hold to move</strong> is held on its phone.
              </>
            ) : (
              stopNote
            )}
          </p>
        </div>
        {running && marionette && (
          <div className="flex flex-col items-end gap-0.5 text-xs">
            <span
              className={cn(
                "flex items-center gap-1.5",
                marionette.gateway.connected ? "text-emerald-300" : "text-amber-300"
              )}
            >
              <span
                aria-hidden
                className={cn(
                  "size-2 rounded-full",
                  marionette.gateway.connected ? "bg-emerald-400" : "bg-amber-400"
                )}
              />
              {marionette.gateway.connected ? "Gateway connected" : "Gateway not connected"}
            </span>
            {marionette.gateway.error && (
              <span className="max-w-80 text-right text-amber-200/80">
                {marionette.gateway.error}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="grid gap-3 lg:min-h-0 lg:flex-1 lg:grid-cols-[minmax(0,1fr)_340px] lg:grid-rows-[minmax(0,1fr)] xl:grid-cols-[minmax(0,1fr)_370px]">
        <MarionetteStage
          running={running}
          ready={ready}
          cameras={marionette?.cameras ?? null}
          robot={
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center gap-2 text-sm text-white/45">
                  <Loader2 className="size-4 animate-spin" /> Loading the 3D view…
                </div>
              }
            >
              <JogViewer
                key={`${robotModel}:${gripper}`}
                robotModel={robotModel}
                gripper={gripper}
                state={viewState}
                plan={null}
                activeArm="left"
                target={null}
                onTargetChange={ignore}
                onSelectArm={ignore}
              />
            </Suspense>
          }
        />

        <aside
          aria-label="Phones, arms and drive"
          className="flex min-h-0 flex-col gap-3 lg:-mr-2 lg:overflow-y-auto lg:pr-2"
        >
          <PhonesCard
            phones={phones}
            arms={arms}
            hint={phoneConnectHint(address, axolHost)}
            pairing={pairing}
          />

          <Card data-testid="marionette-arms" className={PANEL}>
            <CardHeader>
              <CardTitle>Arms</CardTitle>
              <CardDescription className="text-xs">
                <strong className="text-white/80">Reset arm</strong> moves an arm back to its rest
                pose, which straightens out twisted joints. Stop halts it.
              </CardDescription>
            </CardHeader>
            <div className="grid grid-cols-2 gap-x-4 divide-x divide-white/10 [&>*+*]:pl-4">
              {ARMS.map((side) => (
                <ArmRow
                  key={side}
                  side={side}
                  arm={arms?.[side] ?? null}
                  phones={phones}
                  canReset={canReset}
                />
              ))}
            </div>
          </Card>

          <Card data-testid="marionette-drive" className={PANEL}>
            <CardHeader>
              <CardTitle>Drive</CardTitle>
              <CardDescription className="text-xs">
                {robotModel === "mobile"
                  ? "Hold a key or a button. Jelly and the lift stop when you let go."
                  : "Only Axol on Jelly has a powered base and lift."}
              </CardDescription>
            </CardHeader>
            <KeyLegend pressed={pressed} active={running} />
            <label className="flex flex-col gap-1 text-xs text-white/55">
              Speed limit
              <Slider
                aria-label="Speed limit"
                value={driveSpeed}
                min={0.1}
                max={1}
                step={0.05}
                onChange={setDriveSpeed}
                format={(v) => `${Math.round(v * 100)} %`}
              />
            </label>
            <div className="flex items-center gap-3">
              <DrivePad
                className="size-36"
                disabled={!running || !baseState}
                speed={driveSpeed}
                onDrive={(command) => {
                  pad.current = command
                  pushHold()
                }}
              />
              <div className="grid min-w-0 flex-1 gap-1.5">
                <HoldButton
                  label="Turn left"
                  icon={<RotateCcw />}
                  className="h-8"
                  disabled={!running || !baseState}
                  onHold={(held) => {
                    turn.current = held ? 1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Turn right"
                  icon={<RotateCw />}
                  className="h-8"
                  disabled={!running || !baseState}
                  onHold={(held) => {
                    turn.current = held ? -1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Raise lift"
                  icon={<ArrowUp />}
                  className="h-8"
                  disabled={!running || !liftState}
                  onHold={(held) => {
                    lift.current = held ? 1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Lower lift"
                  icon={<ArrowDown />}
                  className="h-8"
                  disabled={!running || !liftState}
                  onHold={(held) => {
                    lift.current = held ? -1 : 0
                    pushHold()
                  }}
                />
              </div>
            </div>
          </Card>

          <RobotStatus state={live} />
        </aside>
      </div>
    </AppShell>
  )
}
