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
import {
  ArrowDown,
  ArrowUp,
  Hand,
  Loader2,
  OctagonX,
  Power,
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
import { SiteNav } from "@/components/site-nav"
import { DrivePad, HoldButton } from "@/components/jog/hold-controls"
import { RobotStatus } from "@/components/jog/robot-status"
import { CameraPanes } from "@/components/marionette/camera-panes"
import { ROSTER } from "@/lib/axol-roster"
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
  isTextEntry,
  phoneActivity,
  phoneConnectHint,
  resetMarionetteArm,
  setMarionetteStop,
  type DriveKey,
  type MarionetteArm,
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
        "flex flex-col gap-1.5 py-3 first:pt-0 last:pb-0",
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
}: {
  phones: MarionettePhone[]
  arms: MarionetteState["arms"] | null
}) {
  return (
    <Card data-testid="marionette-phones">
      <CardHeader>
        <CardTitle>Phones</CardTitle>
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
          ? "Can't reach the Axol host."
          : reason instanceof Error
            ? reason.message
            : String(reason)
      )
    } finally {
      setSending(false)
    }
  }
  return (
    <div data-testid={`arm-${side}`} className="flex flex-col gap-2 py-3 first:pt-0 last:pb-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">{side === "left" ? "Left arm" : "Right arm"}</span>
        <Chip tone={arm?.engaged || arm?.resetting ? "active" : "neutral"}>
          {arm?.resetting ? "Returning to rest" : arm?.engaged ? "Following" : "Not following"}
        </Chip>
      </div>
      <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1 text-xs">
        <span className="text-white/50">Phone</span>
        <span className="flex items-center justify-end gap-1.5 text-white/80">
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
        setHostError("Can't reach the Axol host. Connect to it in the Control Panel first.")
      )
  }, [])

  useEffect(() => {
    let cancelled = false
    const poll = () =>
      fetchOpStatus()
        .then((status) => !cancelled && setOpStatus(status))
        .catch(() => undefined)
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
  // The Axol computer as this page reaches it (the page may come from elsewhere).
  const axolHost = new URL(wsBaseUrl()).hostname
  const address = running
    ? marionette?.gateway.address || String(ownSession?.args?.gateway ?? "") || gateway
    : gateway
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
          ? "Can't reach the Axol host."
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
    <div className="min-h-screen">
      <SiteNav current="marionette" right={<Badge variant={status.variant}>{status.text}</Badge>} />
      <main className="safe-x mx-auto flex max-w-6xl flex-col gap-6 py-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-heading text-2xl font-semibold tracking-tight">Marionette</h1>
            <p data-testid="marionette-connect" className="mt-1 text-sm text-white/50">
              {phoneConnectHint(address, axolHost)}
            </p>
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
                <Square /> Stop Marionette
              </Button>
            ) : (
              <Button
                onClick={start}
                disabled={Boolean(busy) || Boolean(otherOperation) || !connected}
              >
                {busy === "start" ? <Loader2 className="animate-spin" /> : <Power />}
                {mode === "sim" ? "Start sim" : "Start on robot"}
              </Button>
            )}
          </div>
        </div>

        {(error || linkError || hostError || otherOperation || live?.solverError) && (
          <div
            role="alert"
            className="rounded-lg border border-amber-400/25 bg-amber-400/[0.06] px-4 py-3 text-sm text-amber-200"
          >
            {otherOperation
              ? `${otherOperation} is running. Stop it in the Control Panel to use Marionette.`
              : (error ?? live?.solverError ?? linkError ?? hostError)}
          </div>
        )}
        {!running && mode === "hardware" && !otherOperation && (
          <p className="text-sm text-white/50">
            On the robot, an arm moves only while{" "}
            <strong className="text-white/80">Hold to move</strong> is held on its phone.
          </p>
        )}

        <div
          data-testid="marionette-stop"
          className={cn(
            "flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border px-4 py-3",
            stop?.on ? "border-red-400/40 bg-red-500/[0.08]" : "border-white/10 bg-white/[0.03]"
          )}
        >
          <Button
            size="lg"
            data-state={stop?.desktop ? "on" : "off"}
            disabled={!running}
            onClick={(event) => {
              // The second click of a double-click must not turn Stop back off.
              if (event.detail > 1) return
              void setStop(offlineStop ? false : !stop?.desktop)
            }}
            className={cn(
              "min-w-40 font-semibold",
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
            <p className="text-white/50">{stopNote}</p>
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

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
          <div className="flex min-w-0 flex-col gap-6">
            <Card className="gap-0 overflow-hidden p-0">
              <div className="relative h-[clamp(360px,52vh,560px)]">
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
                {running && !ready && (
                  <div className="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
                    <span className="flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                      <Loader2 className="size-3.5 animate-spin" /> Starting Marionette…
                    </span>
                  </div>
                )}
                {!running && (
                  <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center">
                    <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                      Start Marionette to move the robot.
                    </span>
                  </div>
                )}
              </div>
            </Card>
            <CameraPanes running={running} cameras={marionette?.cameras ?? null} />
          </div>

          <div className="flex flex-col gap-6">
            <PhonesCard phones={phones} arms={arms} />

            <Card data-testid="marionette-arms">
              <CardHeader>
                <CardTitle>Arms</CardTitle>
                <CardDescription>
                  <strong className="text-white/80">Reset arm</strong> moves an arm back to its rest
                  pose, which straightens out twisted joints. Stop halts it.
                </CardDescription>
              </CardHeader>
              <div className="flex flex-col divide-y divide-white/10">
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

            <Card data-testid="marionette-drive">
              <CardHeader>
                <CardTitle>Drive</CardTitle>
                <CardDescription>
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
              <div className="flex justify-center">
                <DrivePad
                  disabled={!running || !baseState}
                  speed={driveSpeed}
                  onDrive={(command) => {
                    pad.current = command
                    pushHold()
                  }}
                />
              </div>
              <div className="flex gap-2">
                <HoldButton
                  label="Turn left"
                  icon={<RotateCcw />}
                  disabled={!running || !baseState}
                  onHold={(held) => {
                    turn.current = held ? 1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Turn right"
                  icon={<RotateCw />}
                  disabled={!running || !baseState}
                  onHold={(held) => {
                    turn.current = held ? -1 : 0
                    pushHold()
                  }}
                />
              </div>
              <div className="flex gap-2">
                <HoldButton
                  label="Raise lift"
                  icon={<ArrowUp />}
                  disabled={!running || !liftState}
                  onHold={(held) => {
                    lift.current = held ? 1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Lower lift"
                  icon={<ArrowDown />}
                  disabled={!running || !liftState}
                  onHold={(held) => {
                    lift.current = held ? -1 : 0
                    pushHold()
                  }}
                />
              </div>
            </Card>

            <RobotStatus state={live} />
          </div>
        </div>
      </main>
    </div>
  )
}
