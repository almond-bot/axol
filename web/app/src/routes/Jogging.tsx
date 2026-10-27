import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import {
  ArrowDown,
  ArrowUp,
  Hand,
  Loader2,
  Move3d,
  Play,
  Power,
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
import { SiteNav } from "@/components/site-nav"
import type { GizmoMode, GizmoSpace } from "@/components/jog/jog-viewer"
import { cn } from "@/lib/utils"
import {
  cancelJogMove,
  combineHold,
  describePlanError,
  executeJogPlan,
  IDLE_HOLD,
  NO_TWIST,
  padCommand,
  planJogMove,
  useJogLink,
  withArmJog,
  type JogArm,
  type JogFrame,
  type JogHold,
  type JogPlan,
  type JogPose,
  type JogTwist,
  type Vec3,
} from "@/lib/jog"
import { nudgePose } from "@/lib/jog-pose"
import {
  fetchOpStatus,
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

/** A button that acts only while it is pressed, and always lets go. */
function HoldButton({
  label,
  icon,
  disabled,
  onHold,
  className,
  "aria-label": ariaLabel,
}: {
  label: string
  icon?: ReactNode
  disabled?: boolean
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
        held && "border-[#eff483]/60 bg-[#eff483]/15",
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
  onDrive,
}: {
  disabled: boolean
  speed: number
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
          knob ? "border-[#eff483]/70 bg-[#eff483]/25" : "border-white/20 bg-white/[0.06]"
        )}
        style={{
          transform: `translate(calc(-50% + ${knob?.[0] ?? 0}px), calc(-50% + ${knob?.[1] ?? 0}px))`,
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

export default function Jogging() {
  const [simPick] = useState(storedSimRobot)
  const [mode, setMode] = useState<Mode>("sim")
  const [simModel, setSimModel] = useState<AxolModel>(simPick?.model ?? "mobile")
  const [simGripper] = useState<GripperType>(simPick?.gripper ?? "parallel")
  const [host, setHost] = useState<{ model: AxolModel; gripper: GripperType } | null>(null)
  const [hostError, setHostError] = useState<string | null>(null)
  const [opStatus, setOpStatus] = useState<OpStatus | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [arm, setArm] = useState<JogArm>("left")
  const [edits, setEdits] = useState<Edits>({ epoch: "", poses: {} })
  const [planned, setPlanned] = useState<{ epoch: string; plan: JogPlan } | null>(null)
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate")
  const [gizmoSpace, setGizmoSpace] = useState<GizmoSpace>("world")
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
  const pushHold = useCallback(() => {
    const base = combineHold(pad.current, turn.current, lift.current, driveSpeed)
    holdRef.current = withArmJog(base, arm, armJog.current, jogFrame, jogSpeed)
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
      holdRef.current = { ...IDLE_HOLD }
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

  const nudge = (kind: "move" | "turn", axis: 0 | 1 | 2, sign: 1 | -1) => {
    if (!target) return
    const amount = kind === "move" ? (sign * stepMm) / 1000 : (sign * stepDeg * Math.PI) / 180
    changeTarget(nudgePose(target, kind, axis, amount))
  }

  const planError = plan ? describePlanError(plan) : null
  const previewError = shownPreview ? describePlanError(shownPreview) : null
  const liftState = state?.lift ?? null
  const baseState = state?.base ?? null
  const armJoints = state?.arms?.[arm] ?? null
  const canMoveArm = ready && Boolean(state?.hasArms) && !executing
  const armLimited = Boolean(state?.armJog?.[arm]?.limited)
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
    <div className="min-h-screen">
      <SiteNav current="jog" right={<Badge variant={status.variant}>{status.text}</Badge>} />
      <main className="safe-x mx-auto flex max-w-6xl flex-col gap-6 py-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-heading text-2xl font-semibold tracking-tight">Jog Axol</h1>
            <p className="mt-1 text-sm text-white/50">
              Drag the target, plan the move, check the orange preview, then execute it.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
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
              <Button variant="outline" onClick={stop} disabled={busy === "stop"}>
                <Square /> Stop jogging
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

        {(error || linkError || hostError || otherOperation || state?.solverError) && (
          <div
            role="alert"
            className="rounded-lg border border-amber-400/25 bg-amber-400/[0.06] px-4 py-3 text-sm text-amber-200"
          >
            {otherOperation
              ? `${otherOperation} is running. Stop it in the Control Panel to jog.`
              : (error ?? state?.solverError ?? linkError ?? hostError)}
          </div>
        )}
        {!running && mode === "hardware" && !otherOperation && (
          <p className="text-sm text-white/50">
            On the robot, the arms hold their pose when jogging starts and move only when you press{" "}
            <strong className="text-white/80">Execute</strong>. The lift and base move only while
            you hold their controls.
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
                  gizmoMode={gizmoMode}
                  gizmoSpace={gizmoSpace}
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
              Drag the target gizmo. Drag empty space to orbit, scroll to zoom.
              {target && (
                <span data-testid="jog-target" className="font-mono text-white/70 tabular-nums">
                  Target {target.position.map((v) => (v * 1000).toFixed(0)).join(", ")} mm
                </span>
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
              <div className="grid grid-cols-2 gap-2">
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
                <label className="flex flex-col gap-1 text-xs text-white/55">
                  Gizmo
                  <Select
                    aria-label="Gizmo"
                    value={`${gizmoMode}:${gizmoSpace}`}
                    onChange={(event) => {
                      const [m, s] = event.target.value.split(":")
                      setGizmoMode(m as GizmoMode)
                      setGizmoSpace(s as GizmoSpace)
                    }}
                  >
                    <SelectOption value="translate:world" label="Move" />
                    <SelectOption value="rotate:local" label="Rotate" />
                    <SelectOption value="translate:local" label="Move along gripper" />
                  </Select>
                </label>
              </div>
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
              <Button variant="outline" onClick={resetTarget} disabled={!canMoveArm}>
                <RotateCcw /> Reset target
              </Button>
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

            <Card>
              <CardHeader>
                <CardTitle>{arm === "left" ? "Left" : "Right"} arm joints</CardTitle>
              </CardHeader>
              <div className="grid grid-cols-[1fr_auto_auto] gap-x-3 gap-y-1 text-xs">
                {JOINT_LABELS.map((label, i) => (
                  <div key={label} className="contents">
                    <span className="text-white/55">{label}</span>
                    <span className="text-right font-mono text-white/80 tabular-nums">
                      {armJoints ? `${((armJoints[i] * 180) / Math.PI).toFixed(1)}°` : "–"}
                    </span>
                    <span className="w-14 text-right font-mono text-[#ff9a3c] tabular-nums">
                      {plan && plan.arm === arm
                        ? `${((plan.joints[i] * 180) / Math.PI).toFixed(1)}°`
                        : ""}
                    </span>
                  </div>
                ))}
              </div>
              {plan && plan.arm === arm && (
                <p className="text-[11px] text-white/40">Orange values are the planned pose.</p>
              )}
            </Card>
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
                disabled={!running || !liftState}
                onHold={(held) => {
                  lift.current = held ? 1 : 0
                  pushHold()
                }}
              />
              <HoldButton
                label="Lower"
                icon={<ArrowDown />}
                disabled={!running || !liftState}
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
              <div
                className="h-2 overflow-hidden rounded-full bg-white/10"
                aria-label="Lift height"
              >
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
            <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-start">
              <DrivePad
                disabled={!running || !baseState}
                speed={driveSpeed}
                onDrive={(command) => {
                  pad.current = command
                  pushHold()
                }}
              />
              <div className="flex w-full flex-col gap-3">
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
                    <Zap className="mr-1 inline size-3.5" />A wheel is faulted. Check Jelly in the
                    Control Panel.
                  </p>
                )}
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  )
}
