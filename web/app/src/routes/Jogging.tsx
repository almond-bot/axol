import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { Link } from "react-router-dom"
import {
  ArrowDown,
  ArrowUp,
  Hand,
  House,
  Loader2,
  Move3d,
  Play,
  RotateCcw,
  RotateCw,
  Square,
  Target,
  Zap,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectOption } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { AppShell } from "@/components/app-shell"
import { OperationIcon } from "@/components/operation-launcher"
import { useHostSummary } from "@/lib/host-summary"
import { catalogEntry, rememberOperation } from "@/lib/operation-catalog"
import { cn } from "@/lib/utils"
import {
  cancelJogMove,
  combineHold,
  describePlanError,
  DRIVE_KEYS,
  executeJogPlan,
  goToRest,
  gamepadControls,
  IDLE_HOLD,
  keyDrive,
  NO_DRIVE,
  NO_TWIST,
  padCommand,
  planJogMove,
  robotConnected,
  useJogLink,
  withArmJog,
  type JogArm,
  type JogFrame,
  type JogHold,
  type JogPlan,
  type JogPose,
  type JogState,
  type GamepadMode,
  type JogTwist,
  type OtherDrive,
  type Vec3,
} from "@/lib/jog"
import { nudgePose, rpyDegrees } from "@/lib/jog-pose"
import {
  fetchOpStatus,
  fetchRobotStatus,
  fetchSettings,
  parseAxolModel,
  parseGripper,
  setServerBase,
  startOperation,
  stopOperation,
  type AxolModel,
  type GripperType,
  type OpStatus,
} from "@/lib/supervisor"

const JogViewer = lazy(() => import("@/components/jog/jog-viewer"))

const OPERATION = catalogEntry("jog")

const JOINT_LABELS = [
  "Shoulder 1",
  "Shoulder 2",
  "Shoulder 3",
  "Elbow",
  "Wrist 1",
  "Wrist 2",
  "Wrist 3",
]
const MODEL_LABELS: Record<AxolModel, string> = {
  classic: "Axol Legacy",
  standalone: "Axol Standalone",
  ox_cart: "Axol on Ox Cart",
  owl: "Axol on Owl Mount",
  mobile: "Axol on Jelly",
}
const STEPS_MM = [1, 5, 10, 25, 50]
const STEPS_DEG = [1, 5, 15, 45]
const JOG_AXES = [
  ["X", "Y", "Z"],
  ["Roll", "Pitch", "Yaw"],
] as const

type Mode = "sim" | "hardware"
type Edits = { epoch: string; poses: Partial<Record<JogArm, JogPose>> }

function storedSimRobot(): { model: AxolModel; gripper: GripperType } | null {
  try {
    const raw = JSON.parse(localStorage.getItem("axolSimRobot") ?? "null")
    const model = parseAxolModel(raw?.model)
    return model ? { model, gripper: parseGripper(raw?.gripper) } : null
  } catch {
    return null
  }
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-white/20 bg-white/[0.06] px-1 font-mono text-[10px] text-white/70">
      {children}
    </kbd>
  )
}

/** A button that acts only while it is pressed, and always lets go. */
function HoldButton({
  label,
  icon,
  disabled,
  active,
  onHold,
  className,
  "aria-label": ariaLabel,
}: {
  label: string
  icon?: ReactNode
  disabled?: boolean
  /** Shown pressed by another control (a key or a game controller). */
  active?: boolean
  onHold: (held: boolean) => void
  className?: string
  "aria-label"?: string
}) {
  const [held, setHeld] = useState(false)
  const set = (value: boolean) => {
    setHeld(value)
    onHold(value)
  }
  return (
    <Button
      type="button"
      variant="outline"
      disabled={disabled}
      aria-pressed={held}
      aria-label={ariaLabel}
      className={cn(
        "h-11 flex-1 touch-none select-none",
        (held || active) && "border-[#eff483]/60 bg-[#eff483]/15",
        className
      )}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        set(true)
      }}
      onPointerUp={() => set(false)}
      onPointerCancel={() => set(false)}
      onLostPointerCapture={() => held && set(false)}
      onKeyDown={(event) => {
        if ((event.key === " " || event.key === "Enter") && !event.repeat) set(true)
      }}
      onKeyUp={(event) => {
        if (event.key === " " || event.key === "Enter") set(false)
      }}
      onBlur={() => held && set(false)}
    >
      {icon}
      {label}
    </Button>
  )
}

function DrivePad({
  disabled,
  speed,
  shown,
  onDrive,
}: {
  disabled: boolean
  speed: number
  /** Drive from another control (forward, left; -1 to 1), shown on the knob
   * while the pad itself is not held. */
  shown: Vec3
  onDrive: (command: Vec3) => void
}) {
  const pad = useRef<HTMLDivElement>(null)
  const [knob, setKnob] = useState<[number, number] | null>(null)
  const update = (clientX: number, clientY: number) => {
    const rect = pad.current?.getBoundingClientRect()
    if (!rect) return
    const radius = rect.width / 2
    const dx = clientX - (rect.left + radius)
    const dy = clientY - (rect.top + rect.height / 2)
    const length = Math.hypot(dx, dy)
    const clip = length > radius ? radius / length : 1
    setKnob([dx * clip, dy * clip])
    onDrive(padCommand(dx, dy, radius, speed))
  }
  const release = () => {
    setKnob(null)
    onDrive([0, 0, 0])
  }
  return (
    <div
      ref={pad}
      role="application"
      aria-label="Drive pad"
      aria-disabled={disabled}
      className={cn(
        "relative size-52 shrink-0 touch-none rounded-full border border-white/15 bg-white/[0.03] select-none",
        disabled ? "opacity-40" : "cursor-grab active:cursor-grabbing"
      )}
      onPointerDown={(event) => {
        if (disabled) return
        event.currentTarget.setPointerCapture(event.pointerId)
        update(event.clientX, event.clientY)
      }}
      onPointerMove={(event) => {
        if (knob) update(event.clientX, event.clientY)
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={() => knob && release()}
    >
      <span className="absolute inset-x-6 top-1/2 h-px bg-white/10" />
      <span className="absolute inset-y-6 left-1/2 w-px bg-white/10" />
      <span className="absolute top-2 left-1/2 -translate-x-1/2 text-[10px] text-white/40">
        Forward
      </span>
      <span className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-white/40">
        Back
      </span>
      <span className="absolute top-1/2 left-2 -translate-y-1/2 text-[10px] text-white/40">
        Left
      </span>
      <span className="absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-white/40">
        Right
      </span>
      <span
        className={cn(
          "absolute top-1/2 left-1/2 size-12 rounded-full border transition-colors",
          knob || shown[0] || shown[1]
            ? "border-[#eff483]/70 bg-[#eff483]/25"
            : "border-white/20 bg-white/[0.06]"
        )}
        style={{
          // The pad is 13rem across: an outside control's full deflection
          // puts the knob 5rem from the centre.
          transform: knob
            ? `translate(calc(-50% + ${knob[0]}px), calc(-50% + ${knob[1]}px))`
            : `translate(calc(-50% + ${-shown[1] * 5}rem), calc(-50% + ${-shown[0] * 5}rem))`,
        }}
      />
    </div>
  )
}

/** Hold buttons for a gripper jog: move along and turn about x, y and z. */
function JogButtons({
  disabled,
  onJog,
}: {
  disabled: boolean
  onJog: (axis: number, sign: 1 | -1, held: boolean) => void
}) {
  return JOG_AXES.map((names, group) => (
    <div key={group} className="grid grid-cols-3 gap-1.5">
      {names.map((name, i) =>
        ([1, -1] as const).map((sign) => (
          <HoldButton
            key={`${name}${sign}`}
            label={`${name} ${sign > 0 ? "+" : "−"}`}
            aria-label={`Jog ${name} ${sign > 0 ? "plus" : "minus"}`}
            className="h-9 flex-none text-sm"
            disabled={disabled}
            onHold={(held) => onJog(group * 3 + i, sign, held)}
          />
        ))
      )}
    </div>
  ))
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2">
      <p className="text-[11px] text-white/45">{label}</p>
      <p className="mt-0.5 font-mono text-sm tabular-nums">{value}</p>
    </div>
  )
}

const deg = (rad: number, digits = 1) => `${((rad * 180) / Math.PI).toFixed(digits)}°`

/** One condensed table of everything moving: both arms, the lift and the base. */
function RobotStatus({
  state,
  activeArm,
  plan,
}: {
  state: JogState | null
  activeArm: JogArm
  plan: JogPlan | null
}) {
  const arms = state?.arms
  const lift = state?.lift
  const base = state?.base
  const row = "contents"
  const name = "text-white/50"
  const value = "text-right font-mono text-white/80 tabular-nums"
  const armHeader = (side: JogArm) =>
    cn("text-right font-medium", side === activeArm ? "text-[#eff483]" : "text-white/50")
  return (
    <Card data-testid="jog-status">
      <CardHeader>
        <CardTitle>Robot state</CardTitle>
      </CardHeader>
      <div
        className={cn(
          "grid gap-x-3 gap-y-1 text-xs",
          plan ? "grid-cols-[1fr_auto_auto_auto]" : "grid-cols-[1fr_auto_auto]"
        )}
      >
        <span className={name}>Joint</span>
        <span className={armHeader("left")}>Left</span>
        <span className={armHeader("right")}>Right</span>
        {plan && (
          <span className="text-right font-medium text-[#ff9a3c]">
            Plan {plan.arm === "left" ? "L" : "R"}
          </span>
        )}
        {JOINT_LABELS.map((label, i) => (
          <div key={label} className={row}>
            <span className={name}>{label}</span>
            <span data-testid={`joint-left-${i}`} className={value}>
              {arms ? deg(arms.left[i]) : "–"}
            </span>
            <span data-testid={`joint-right-${i}`} className={value}>
              {arms ? deg(arms.right[i]) : "–"}
            </span>
            {plan && (
              <span className="text-right font-mono text-[#ff9a3c] tabular-nums">
                {deg(plan.joints[i])}
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 border-t border-white/10 pt-3 text-xs">
        <span className={name}>Lift</span>
        <span data-testid="status-lift" className={value}>
          {lift?.raised != null
            ? `${(lift.raised * 100).toFixed(1)} % · ${(lift.velocity * 100).toFixed(2)} %/s`
            : lift
              ? "Not homed"
              : "–"}
        </span>
        <span className={name}>Base position{base?.estimated ? " (est.)" : ""}</span>
        <span data-testid="status-odometry" className={value}>
          {base ? `${base.x.toFixed(2)}, ${base.y.toFixed(2)} m · ${deg(base.yaw, 0)}` : "–"}
        </span>
        <span className={name}>Base speed</span>
        <span className={value}>
          {base
            ? `${Math.hypot(base.velocity[0], base.velocity[1]).toFixed(2)} m/s · ${deg(base.velocity[2], 0)}/s`
            : "–"}
        </span>
        <span className={name}>Wheel command</span>
        <span className={value}>
          {base ? base.command.map((c) => `${Math.round(c * 100)}`).join(" / ") + " %" : "–"}
        </span>
        {base?.fault && (
          <span className="col-span-2 text-amber-300">
            <Zap className="mr-1 inline size-3.5" />A wheel is faulted.
          </span>
        )}
      </div>
    </Card>
  )
}

export default function Jogging() {
  const [simPick] = useState(storedSimRobot)
  const [mode, setMode] = useState<Mode>("sim")
  // Whether the operator picked Sim or Robot: the robot's status then no
  // longer moves the default.
  const modeChosen = useRef(false)
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

  const [arm, setArm] = useState<JogArm>("left")
  const [edits, setEdits] = useState<Edits>({ epoch: "", poses: {} })
  const [planned, setPlanned] = useState<{ epoch: string; plan: JogPlan } | null>(null)
  const [stepMm, setStepMm] = useState(10)
  const [stepDeg, setStepDeg] = useState(15)
  const [driveSpeed, setDriveSpeed] = useState(0.5)
  const [jogFrame, setJogFrame] = useState<JogFrame>("torso")
  const [jogSpeed, setJogSpeed] = useState(0.5)
  // The arm and epoch the shown preview was asked for.
  const [previewFor, setPreviewFor] = useState<{ epoch: string; arm: JogArm } | null>(null)

  const holdRef = useRef<JogHold>({ ...IDLE_HOLD })
  const pad = useRef<Vec3>([0, 0, 0])
  const turn = useRef<-1 | 0 | 1>(0)
  const lift = useRef<-1 | 0 | 1>(0)
  const armJog = useRef<JogTwist>([...NO_TWIST])
  // The drive keys held down, and what a game controller asks for (unscaled).
  const keys = useRef(new Set<string>())
  const stick = useRef<OtherDrive>(NO_DRIVE)
  const stickTwist = useRef<JogTwist>([...NO_TWIST])
  // What the keys and the controller ask the base and lift for, unscaled, for
  // the pad's knob and the turn and lift buttons.
  const [otherDrive, setOtherDrive] = useState<OtherDrive>(NO_DRIVE)
  const pushHold = useCallback(() => {
    // The page's own controls win over the controller, and the controller
    // over the keys.
    const keyed = keyDrive(keys.current, 1)
    const stickDrives = stick.current.pad[0] !== 0 || stick.current.pad[1] !== 0
    const other: OtherDrive = {
      pad: stickDrives ? stick.current.pad : keyed.pad,
      turn: stick.current.turn || keyed.turn,
      lift: stick.current.lift || keyed.lift,
    }
    const scale = Math.max(0, Math.min(1, driveSpeed))
    const padHeld = pad.current[0] !== 0 || pad.current[1] !== 0
    const drive: Vec3 = padHeld ? pad.current : [other.pad[0] * scale, other.pad[1] * scale, 0]
    const base = combineHold(
      drive,
      turn.current || other.turn,
      lift.current || other.lift,
      driveSpeed
    )
    const twist = armJog.current.some((v) => v !== 0) ? armJog.current : stickTwist.current
    holdRef.current = withArmJog(base, arm, twist, jogFrame, jogSpeed)
    setOtherDrive((shown) =>
      shown.pad[0] === other.pad[0] &&
      shown.pad[1] === other.pad[1] &&
      shown.turn === other.turn &&
      shown.lift === other.lift
        ? shown
        : other
    )
  }, [driveSpeed, arm, jogFrame, jogSpeed])
  useEffect(() => pushHold(), [pushHold])

  const { state, connected, error: linkError, preview, requestPreview } = useJogLink(true, holdRef)

  // Let go of everything when the window loses focus: a missed pointer-up
  // must never leave the base, lift or an arm running.
  useEffect(() => {
    const releaseAll = () => {
      pad.current = [0, 0, 0]
      turn.current = 0
      lift.current = 0
      armJog.current = [...NO_TWIST]
      keys.current.clear()
      stick.current = NO_DRIVE
      stickTwist.current = [...NO_TWIST]
      holdRef.current = { ...IDLE_HOLD }
      setOtherDrive(NO_DRIVE)
    }
    window.addEventListener("blur", releaseAll)
    document.addEventListener("visibilitychange", releaseAll)
    return () => {
      window.removeEventListener("blur", releaseAll)
      document.removeEventListener("visibilitychange", releaseAll)
    }
  }, [])

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
    // Start on the robot when one is connected, like the other operations.
    let active = true
    fetchRobotStatus()
      .then((status) => {
        if (active && !modeChosen.current && robotConnected(status)) setMode("hardware")
      })
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [])

  useEffect(() => rememberOperation("jog"), [])

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
  const jogSession = session?.command === "jog" ? session : null
  const otherOperation = session && session.command !== "jog" ? session.command : null
  const running = Boolean(state?.running && jogSession)
  const sessionMode: Mode | null = running ? (state?.mode ?? null) : null
  // The robot drawn is what the session runs, or what Start would run.
  const viewMode: Mode = sessionMode ?? mode
  const sessionModel = parseAxolModel(String(jogSession?.args?.["kinematics.robot_model"] ?? ""))
  const robotModel: AxolModel =
    viewMode === "sim"
      ? sessionMode
        ? (sessionModel ?? simModel)
        : simModel
      : (host?.model ?? "classic")
  const gripper: GripperType = viewMode === "sim" ? simGripper : (host?.gripper ?? "parallel")
  const ready = running && Boolean(state?.ready)
  const executing = state?.execution ?? null
  // Edits and plans belong to one session and one arm pose: starting a new
  // session or finishing a move puts the targets back on the grippers.
  const epoch = `${jogSession?.id ?? ""}:${state?.lastExecution?.id ?? ""}`
  const tips = running ? state?.tips : null
  const edited = edits.epoch === epoch ? edits.poses[arm] : undefined
  const target = edited ?? tips?.[arm] ?? null
  const plan = running && planned?.epoch === epoch ? planned.plan : null
  const setPlan = (next: JogPlan | null) => setPlanned(next ? { epoch, plan: next } : null)
  // The live preview belongs to an edited target of this arm and session.
  const shownPreview =
    running &&
    edited &&
    preview?.arm === arm &&
    previewFor?.arm === arm &&
    previewFor.epoch === epoch
      ? preview
      : null

  const askPreview = (pose: JogPose) => {
    if (!running || !state?.ready) return
    requestPreview(arm, pose)
    setPreviewFor((current) =>
      current?.epoch === epoch && current.arm === arm ? current : { epoch, arm }
    )
  }
  const changeTarget = (pose: JogPose) => {
    setEdits((current) => ({
      epoch,
      poses: { ...(current.epoch === epoch ? current.poses : {}), [arm]: pose },
    }))
    setPlan(null)
    askPreview(pose)
  }
  const releaseArmJog = () => {
    armJog.current = [...NO_TWIST]
    pushHold()
  }
  const selectArm = (next: JogArm) => {
    releaseArmJog()
    setArm(next)
    setPlan(null)
  }
  const jogArm = (axis: number, sign: 1 | -1, held: boolean) => {
    const twist = [...armJog.current] as JogTwist
    twist[axis] = held ? sign : twist[axis] === sign ? 0 : twist[axis]
    armJog.current = twist
    pushHold()
    // Moving the arm makes a plan stale; a preview is solved again from
    // where the arm stops.
    if (held) {
      setPlan(null)
      setPreviewFor(null)
    } else if (edited && !armJog.current.some((v) => v !== 0)) {
      askPreview(edited)
    }
  }
  // Back on the gripper: the target follows it again, jogs included.
  const resetTarget = () => {
    setEdits((current) => {
      const poses = { ...(current.epoch === epoch ? current.poses : {}) }
      delete poses[arm]
      return { epoch, poses }
    })
    setPlan(null)
  }

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
      const args: Record<string, string | boolean> = { sim: mode === "sim" }
      if (mode === "sim") {
        args["kinematics.robot_model"] = simModel
        args["kinematics.gripper"] = simGripper
      }
      await startOperation("jog", args)
      setOpStatus(await fetchOpStatus())
    })
  const stop = () =>
    run("stop", async () => {
      await stopOperation()
      setOpStatus(await fetchOpStatus())
    })
  const planMove = () =>
    run("plan", async () => {
      if (!target) return
      setPlan(await planJogMove(arm, target))
    })
  const execute = () =>
    run("execute", async () => {
      if (plan) await executeJogPlan(plan.id)
    })
  const cancel = () => run("cancel", () => cancelJogMove())
  const rest = () =>
    run("rest", async () => {
      await goToRest()
    })

  const nudge = (kind: "move" | "turn", axis: 0 | 1 | 2, sign: 1 | -1) => {
    if (!target) return
    const amount = kind === "move" ? (sign * stepMm) / 1000 : (sign * stepDeg * Math.PI) / 180
    changeTarget(nudgePose(target, kind, axis, amount))
  }

  const planError = plan ? describePlanError(plan) : null
  const previewError = shownPreview ? describePlanError(shownPreview) : null
  const liftState = state?.lift ?? null
  const baseState = state?.base ?? null
  const canMoveArm = ready && Boolean(state?.hasArms) && !executing
  const armLimited = Boolean(state?.armJog?.[arm]?.limited)
  const canDrive = running && Boolean(baseState)
  const canLift = running && Boolean(liftState)
  // A connected controller: its id, "" for one without the standard layout,
  // or null; and what its sticks control.
  const [gamepad, setGamepad] = useState<string | null>(null)
  const [gamepadMode, setGamepadMode] = useState<GamepadMode>("drive")

  // Drive the base and the lift from the keyboard: W A S D move, the left
  // and right arrows turn, the up and down arrows raise and lower the lift.
  // Typing in a field, a focused slider or menu, and shortcuts with a
  // modifier key are left alone.
  useEffect(() => {
    if (!canDrive && !canLift) {
      if (keys.current.size) {
        keys.current.clear()
        pushHold()
      }
      return
    }
    const ownsKeys = (target: EventTarget | null) =>
      target instanceof HTMLElement &&
      (target.isContentEditable ||
        target.closest(
          "input, textarea, select, [role=slider], [role=listbox], [role=menu], [role=combobox]"
        ) !== null)
    const down = (event: KeyboardEvent) => {
      if (!DRIVE_KEYS.has(event.code)) return
      if (event.ctrlKey || event.metaKey || event.altKey || ownsKeys(event.target)) return
      event.preventDefault()
      if (keys.current.has(event.code)) return
      keys.current.add(event.code)
      pushHold()
    }
    const up = (event: KeyboardEvent) => {
      if (keys.current.delete(event.code)) pushHold()
    }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    return () => {
      window.removeEventListener("keydown", down)
      window.removeEventListener("keyup", up)
    }
  }, [canDrive, canLift, pushHold])

  // Drive Jelly, the lift and the selected gripper from a game controller in
  // the standard layout (see gamepadControls). It acts only while this page
  // has focus, and lets go when the controller is unplugged.
  useEffect(() => {
    if (typeof navigator === "undefined" || !navigator.getGamepads) return
    let frame = 0
    let picked: JogArm | null = null
    const poll = () => {
      frame = requestAnimationFrame(poll)
      const pads = Array.from(navigator.getGamepads()).filter(
        (gp): gp is Gamepad => gp !== null && gp.connected
      )
      const gp = pads.find((p) => p.mapping === "standard") ?? null
      setGamepad(gp ? gp.id : pads.length ? "" : null)
      const focused = document.hasFocus() && document.visibilityState === "visible"
      const read = gp && focused ? gamepadControls(gp.axes, gp.buttons) : null
      setGamepadMode(read?.mode ?? "drive")
      // Pick an arm on the press, not while it is held.
      if (read?.pick && read.pick !== picked && canMoveArm) setArm(read.pick)
      picked = read?.pick ?? null
      const next: OtherDrive = read
        ? {
            pad: canDrive ? read.drive.pad : [0, 0, 0],
            turn: canDrive ? read.drive.turn : 0,
            lift: canLift ? read.drive.lift : 0,
          }
        : NO_DRIVE
      const twist: JogTwist = read && canMoveArm ? read.twist : [...NO_TWIST]
      const last = stick.current
      const moved = (a: number, b: number) => Math.abs(a - b) > 0.01 || (a === 0) !== (b === 0)
      if (
        moved(next.pad[0], last.pad[0]) ||
        moved(next.pad[1], last.pad[1]) ||
        moved(next.turn, last.turn) ||
        next.lift !== last.lift ||
        twist.some((v, i) => moved(v, stickTwist.current[i]))
      ) {
        stick.current = next
        stickTwist.current = twist
        pushHold()
      }
    }
    frame = requestAnimationFrame(poll)
    return () => {
      cancelAnimationFrame(frame)
      stick.current = NO_DRIVE
      stickTwist.current = [...NO_TWIST]
      pushHold()
    }
  }, [canDrive, canLift, canMoveArm, pushHold])
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

  return (
    <AppShell
      title="Operations"
      back="/control/operations"
      wide
      badges={{ "/control/operations": session ? "running" : undefined }}
      host={hostSummary}
    >
      <section
        className={cn(
          "flex flex-col gap-5 rounded-2xl border p-5 lg:flex-row lg:items-center",
          running
            ? "border-emerald-400/25 bg-gradient-to-br from-emerald-400/[0.06] to-transparent"
            : "border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent"
        )}
      >
        <div className="flex min-w-0 flex-1 items-start gap-4">
          <OperationIcon entry={OPERATION} size="lg" running={running} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="font-heading text-xl font-semibold">{OPERATION.label}</h2>
              <Badge variant={status.variant}>{status.text}</Badge>
            </div>
            <p className="mt-1.5 max-w-prose text-sm leading-relaxed text-white/55">
              Drag the target, plan the move, check the orange preview, then execute it.
            </p>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
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
                onClick={() => {
                  modeChosen.current = true
                  setMode(value)
                }}
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
                localStorage.setItem(
                  "axolSimRobot",
                  JSON.stringify({ model: next, gripper: simGripper })
                )
              }}
            >
              {Object.entries(MODEL_LABELS).map(([value, label]) => (
                <SelectOption key={value} value={value} label={label} />
              ))}
            </Select>
          )}
          {running ? (
            <Button variant="destructive" size="lg" onClick={stop} disabled={busy === "stop"}>
              {busy === "stop" ? <Loader2 className="animate-spin" /> : <Square />}
              Stop jogging
            </Button>
          ) : (
            <Button
              size="lg"
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

      {(error || linkError || hostError || otherOperation || state?.solverError) && (
        <div
          role="alert"
          className="rounded-lg border border-amber-400/25 bg-amber-400/[0.06] px-4 py-3 text-sm text-amber-200"
        >
          {otherOperation ? (
            <>
              {catalogEntry(otherOperation).label} is running. Stop it to jog.{" "}
              <Link
                to={`/control/operations/${encodeURIComponent(otherOperation)}`}
                className="font-medium underline underline-offset-2"
              >
                Open {catalogEntry(otherOperation).label}
              </Link>
            </>
          ) : (
            (error ?? state?.solverError ?? linkError ?? hostError)
          )}
        </div>
      )}
      {!running && mode === "hardware" && !otherOperation && (
        <p className="text-sm text-white/50">
          On the robot, the arms hold their pose when jogging starts and move only when you press{" "}
          <strong className="text-white/80">Execute</strong>. The lift and base move only while you
          hold their controls.
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-start">
        <Card className="gap-0 overflow-hidden p-0">
          <div className="relative h-[clamp(420px,64vh,680px)]">
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
                state={running ? (state ?? null) : null}
                plan={plan}
                preview={shownPreview}
                activeArm={arm}
                target={canMoveArm || plan ? target : null}
                onTargetChange={changeTarget}
                onSelectArm={selectArm}
              />
            </Suspense>
            {running && !ready && (
              <div className="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
                <span className="flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                  <Loader2 className="size-3.5 animate-spin" /> Starting the IK solver…
                </span>
              </div>
            )}
            {!running && (
              <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center">
                <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                  Start jogging to move the robot.
                </span>
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 border-t border-white/10 px-4 py-3 text-xs text-white/50">
            <Move3d className="size-4" />
            Drag the gizmo arrows to move the target and its rings to turn it. Drag empty space to
            orbit, scroll to zoom.
            {target && (
              <>
                <span data-testid="jog-target" className="font-mono text-white/70 tabular-nums">
                  Target {target.position.map((v) => (v * 1000).toFixed(0)).join(", ")} mm
                </span>
                <span
                  data-testid="jog-target-rpy"
                  className="font-mono text-white/70 tabular-nums"
                  title="Roll, pitch and yaw in the robot's axes"
                >
                  {rpyDegrees(target.quaternion)
                    .map((v) => v.toFixed(0))
                    .join(", ")}
                  °
                </span>
              </>
            )}
            {plan && planError && (
              <span
                className={cn("ml-auto", plan.reachable ? "text-emerald-300" : "text-amber-300")}
              >
                {plan.reachable
                  ? `Reaches the target in ${plan.duration.toFixed(1)} s`
                  : `Closest reachable pose: ${planError.mm.toFixed(0)} mm and ${planError.degrees.toFixed(0)}° from the target`}
              </span>
            )}
            {!plan && shownPreview && previewError && (
              <span
                data-testid="jog-preview"
                className={cn(
                  "ml-auto",
                  shownPreview.reachable ? "text-emerald-300" : "text-amber-300"
                )}
              >
                {shownPreview.reachable
                  ? "Preview: the gripper reaches the target"
                  : `Preview: closest pose is ${previewError.mm.toFixed(0)} mm and ${previewError.degrees.toFixed(0)}° away`}
              </span>
            )}
          </div>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Arm move</CardTitle>
              <CardDescription>
                Plan a move of one gripper to the target, then execute it.
              </CardDescription>
            </CardHeader>
            <label className="flex flex-col gap-1 text-xs text-white/55">
              Arm
              <Select
                aria-label="Arm"
                value={arm}
                onChange={(event) => selectArm(event.target.value as JogArm)}
              >
                <SelectOption value="left" label="Left" />
                <SelectOption value="right" label="Right" />
              </Select>
            </label>
            <div className="flex gap-2">
              <Button
                className="flex-1"
                onClick={planMove}
                disabled={!canMoveArm || !target || Boolean(busy)}
              >
                {busy === "plan" ? <Loader2 className="animate-spin" /> : <Target />} Plan move
              </Button>
              {executing ? (
                <Button variant="destructive" className="flex-1" onClick={cancel}>
                  <Square /> Stop move
                </Button>
              ) : (
                <Button
                  className="flex-1"
                  onClick={execute}
                  disabled={!plan || !canMoveArm || Boolean(busy)}
                >
                  {busy === "execute" ? <Loader2 className="animate-spin" /> : <Play />} Execute
                </Button>
              )}
            </div>
            {executing && (
              <div
                className="h-1.5 overflow-hidden rounded-full bg-white/10"
                aria-label="Move progress"
              >
                <div
                  className="h-full bg-[#eff483]"
                  style={{ width: `${Math.round(executing.progress * 100)}%` }}
                />
              </div>
            )}
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex-1"
                onClick={resetTarget}
                disabled={!canMoveArm}
              >
                <RotateCcw /> Reset target
              </Button>
              <Button
                variant="outline"
                className="flex-1"
                onClick={rest}
                disabled={!canMoveArm || Boolean(busy)}
                title="Move both arms to the rest pose. Stop move halts it."
              >
                {busy === "rest" ? <Loader2 className="animate-spin" /> : <House />} Go to rest
              </Button>
            </div>
            <div className="flex flex-col gap-2 border-t border-white/10 pt-4">
              <div className="flex items-center justify-between text-xs text-white/55">
                <span>Move target</span>
                <Select
                  aria-label="Move step"
                  className="h-8 w-24"
                  value={stepMm}
                  onChange={(event) => setStepMm(Number(event.target.value))}
                >
                  {STEPS_MM.map((mm) => (
                    <SelectOption key={mm} value={String(mm)} label={`${mm} mm`} />
                  ))}
                </Select>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {(["X", "Y", "Z"] as const).map((name, axis) =>
                  ([1, -1] as const).map((sign) => (
                    <Button
                      key={`${name}${sign}`}
                      variant="outline"
                      size="sm"
                      disabled={!canMoveArm || !target}
                      onClick={() => nudge("move", axis as 0 | 1 | 2, sign)}
                    >
                      {name} {sign > 0 ? "+" : "−"}
                    </Button>
                  ))
                )}
              </div>
              <div className="mt-1 flex items-center justify-between text-xs text-white/55">
                <span>Turn target</span>
                <Select
                  aria-label="Turn step"
                  className="h-8 w-24"
                  value={stepDeg}
                  onChange={(event) => setStepDeg(Number(event.target.value))}
                >
                  {STEPS_DEG.map((deg) => (
                    <SelectOption key={deg} value={String(deg)} label={`${deg}°`} />
                  ))}
                </Select>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {(["Roll", "Pitch", "Yaw"] as const).map((name, axis) =>
                  ([1, -1] as const).map((sign) => (
                    <Button
                      key={`${name}${sign}`}
                      variant="outline"
                      size="sm"
                      disabled={!canMoveArm || !target}
                      onClick={() => nudge("turn", axis as 0 | 1 | 2, sign)}
                    >
                      {name} {sign > 0 ? "+" : "−"}
                    </Button>
                  ))
                )}
              </div>
              <p className="text-[11px] text-white/40">
                X is forward, Y is left and Z is up. Turns are about the gripper.
              </p>
            </div>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Jog gripper</CardTitle>
              <CardDescription>
                Hold a button to move the {arm} gripper. It stops when you let go.
              </CardDescription>
            </CardHeader>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-1 text-xs text-white/55">
                Axes
                <Select
                  aria-label="Jog axes"
                  value={jogFrame}
                  onChange={(event) => setJogFrame(event.target.value as JogFrame)}
                >
                  <SelectOption value="torso" label="Robot" />
                  <SelectOption value="gripper" label="Gripper" />
                </Select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-white/55">
                Speed
                <Slider
                  aria-label="Jog speed"
                  className="h-9"
                  value={jogSpeed}
                  min={0.1}
                  max={1}
                  step={0.05}
                  onChange={setJogSpeed}
                  format={(v) => `${Math.round(v * 100)} %`}
                />
              </label>
            </div>
            <JogButtons disabled={!canMoveArm} onJog={jogArm} />
            {armLimited && (
              <p className="text-xs text-amber-300" role="status">
                <Hand className="mr-1 inline size-3.5" />
                The {arm} gripper can't move further this way.
              </p>
            )}
            <p className="text-[11px] text-white/40">
              {jogFrame === "torso"
                ? "X is forward, Y is left and Z is up. Turns are about the gripper tip."
                : "The axes turn with the gripper. The fingers point along Z −."}
            </p>
          </Card>

          <RobotStatus state={running ? (state ?? null) : null} activeArm={arm} plan={plan} />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Lift</CardTitle>
            <CardDescription>
              {robotModel === "mobile"
                ? "Hold a button to raise or lower the lift. It stops when you let go."
                : "Only Axol on Jelly has a powered lift."}
            </CardDescription>
          </CardHeader>
          <div className="flex gap-2">
            <HoldButton
              label="Raise"
              icon={<ArrowUp />}
              disabled={!canLift}
              active={canLift && otherDrive.lift > 0}
              onHold={(held) => {
                lift.current = held ? 1 : 0
                pushHold()
              }}
            />
            <HoldButton
              label="Lower"
              icon={<ArrowDown />}
              disabled={!canLift}
              active={canLift && otherDrive.lift < 0}
              onHold={(held) => {
                lift.current = held ? -1 : 0
                pushHold()
              }}
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Readout
              label="Height"
              value={
                liftState?.raised != null
                  ? `${(liftState.raised * 100).toFixed(1)} %`
                  : running && liftState
                    ? "Not homed"
                    : "–"
              }
            />
            <Readout
              label="Speed"
              value={liftState ? `${(liftState.velocity * 100).toFixed(2)} %/s` : "–"}
            />
          </div>
          {liftState?.raised != null && (
            <div className="h-2 overflow-hidden rounded-full bg-white/10" aria-label="Lift height">
              <div
                className="h-full bg-[#eff483]/80"
                style={{ width: `${liftState.raised * 100}%` }}
              />
            </div>
          )}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Drive base</CardTitle>
            <CardDescription>
              {robotModel === "mobile"
                ? "Hold the pad to drive. Jelly ramps up and down like it does in teleop."
                : "Only Axol on Jelly has a powered base."}
            </CardDescription>
          </CardHeader>
          <div className="flex flex-wrap items-start justify-center gap-4">
            <DrivePad
              disabled={!canDrive}
              speed={driveSpeed}
              shown={otherDrive.pad}
              onDrive={(command) => {
                pad.current = command
                pushHold()
              }}
            />
            <div className="flex min-w-64 flex-1 flex-col gap-3">
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
              <div className="flex gap-2">
                <HoldButton
                  label="Turn left"
                  icon={<RotateCcw />}
                  disabled={!canDrive}
                  active={canDrive && otherDrive.turn > 0}
                  onHold={(held) => {
                    turn.current = held ? 1 : 0
                    pushHold()
                  }}
                />
                <HoldButton
                  label="Turn right"
                  icon={<RotateCw />}
                  disabled={!canDrive}
                  active={canDrive && otherDrive.turn < 0}
                  onHold={(held) => {
                    turn.current = held ? -1 : 0
                    pushHold()
                  }}
                />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Readout label="X" value={baseState ? `${baseState.x.toFixed(2)} m` : "–"} />
                <Readout label="Y" value={baseState ? `${baseState.y.toFixed(2)} m` : "–"} />
                <Readout
                  label="Heading"
                  value={baseState ? `${((baseState.yaw * 180) / Math.PI).toFixed(0)}°` : "–"}
                />
              </div>
              <Readout
                label="Speed"
                value={
                  baseState
                    ? `${Math.hypot(baseState.velocity[0], baseState.velocity[1]).toFixed(2)} m/s, ${((baseState.velocity[2] * 180) / Math.PI).toFixed(0)}°/s`
                    : "–"
                }
              />
              {baseState?.estimated && (
                <p className="text-[11px] text-white/40">
                  Position is estimated from the wheel commands.
                </p>
              )}
              {baseState?.fault && (
                <p className="text-xs text-amber-300">
                  <Zap className="mr-1 inline size-3.5" />A wheel is faulted. Check Jelly on the
                  Devices page.
                </p>
              )}
            </div>
          </div>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Keyboard and controller</CardTitle>
            <CardDescription role="status">
              {gamepad
                ? gamepadMode === "move"
                  ? `Controller connected. Moving the ${arm} gripper.`
                  : gamepadMode === "turn"
                    ? `Controller connected. Turning the ${arm} gripper.`
                    : "Controller connected."
                : gamepad === ""
                  ? "The browser can't read this controller's buttons. On a Logitech controller, move the switch on the back to the other letter (D on a Mac), then press a button."
                  : "To use a game controller, plug it in and press any button on it."}
            </CardDescription>
          </CardHeader>
          <div className="grid gap-4 text-xs text-white/60 sm:grid-cols-2">
            <ControlList
              title="Keyboard"
              rows={[
                [<KeyCombo key="k" keys={["W", "A", "S", "D"]} />, "Drive Jelly"],
                [<KeyCombo key="k" keys={["Left", "Right"]} />, "Turn Jelly in place"],
                [<KeyCombo key="k" keys={["Up", "Down"]} />, "Raise or lower the lift"],
              ]}
            />
            <ControlList
              title="Controller"
              rows={[
                ["Left stick", "Drive Jelly"],
                ["Right stick", "Turn Jelly in place"],
                ["D-pad up and down", "Raise or lower the lift"],
                ["D-pad left and right", "Choose the left or right arm"],
                ["Hold LB", "Move the gripper: left stick across, right stick up and down"],
                ["Hold RB", "Turn the gripper: left stick tilts, right stick turns"],
              ]}
            />
          </div>
          <p className="text-[11px] text-white/40">
            The gripper moves in the frame and at the speed set under{" "}
            <strong className="text-white/70">Jog gripper</strong>. Everything stops when you let go
            or leave this page.
          </p>
        </Card>
      </div>
    </AppShell>
  )
}

function KeyCombo({ keys }: { keys: string[] }) {
  return (
    <span className="flex flex-wrap gap-1">
      {keys.map((k) => (
        <Kbd key={k}>{k}</Kbd>
      ))}
    </span>
  )
}

function ControlList({ title, rows }: { title: string; rows: [ReactNode, string][] }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-[11px] font-medium tracking-wide text-white/40 uppercase">{title}</span>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5">
        {rows.map(([control, action], i) => (
          <div key={i} className="contents">
            <dt className="text-white/80">{control}</dt>
            <dd>{action}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
