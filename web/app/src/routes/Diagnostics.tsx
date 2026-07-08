import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Activity, Cable, Crosshair, Loader2, Radio, Tag, Wrench } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { SiteNav } from "@/components/site-nav"
import { useToast } from "@/components/ui/toast"
import { MotorGrid } from "@/components/diagnostics/motor-grid"
import { RunHistory } from "@/components/diagnostics/run-history"
import { JointFilter } from "@/components/diagnostics/joint-filter"
import {
  ActionDialog,
  ActiveRunPanel,
  DiagnosticActions,
  type ActionMode,
} from "@/components/diagnostics/diagnostic-actions"
import {
  TelemetryChart,
  type ChartSeries,
  type ChartView,
} from "@/components/diagnostics/telemetry-chart"
import { cn } from "@/lib/utils"
import {
  fetchCommands,
  fetchRobotStatus,
  robotConnect,
  sendSessionInput,
  setServerBase,
  stopSession,
  useSessionLogs,
  type CommandSpec,
  type FormValue,
  type RobotState,
  type RobotStatus,
  type SessionInfo,
} from "@/lib/supervisor"
import {
  JOINTS,
  JOINT_COLORS,
  clearDiagnosticsRuns,
  fetchDiagnosticsRuns,
  jointLabel,
  motorKey,
  startDiagnosticsRun,
  useTelemetryStream,
  type ArmSide,
  type DiagnosticsRunMeta,
  type JointName,
} from "@/lib/telemetry"

const WINDOWS: { label: string; seconds: number }[] = [
  { label: "30s", seconds: 30 },
  { label: "1m", seconds: 60 },
  { label: "2m", seconds: 120 },
  { label: "5m", seconds: 300 },
  { label: "10m", seconds: 600 },
]

const STATE_BADGE: Record<
  RobotState,
  { variant: "success" | "warning" | "destructive" | "neutral"; text: string }
> = {
  connected: { variant: "success", text: "streaming" },
  busy: { variant: "warning", text: "test owns the bus" },
  connecting: { variant: "neutral", text: "connecting" },
  disconnected: { variant: "neutral", text: "robot disconnected" },
  error: { variant: "destructive", text: "error" },
}

/**
 * Motor diagnostics dashboard: live per-motor status (health, temperature,
 * voltage), always-running position / velocity / torque charts with joint
 * filtering and zoom/pan, one-click diagnostics with parameter dialogs, and
 * the recorded history of past runs.
 *
 * Telemetry streams whenever the idle robot link owns the CAN bus. While a
 * diagnostic or operation owns it the stream pauses (single owner) — charts
 * keep their history and show why.
 */
export default function Diagnostics() {
  const toast = useToast()
  const [serverOk, setServerOk] = useState(false)
  const [commands, setCommands] = useState<CommandSpec[]>([])
  const [robot, setRobot] = useState<RobotStatus | null>(null)
  const [robotBusy, setRobotBusy] = useState(false)

  const [arm, setArm] = useState<ArmSide>(
    () => (localStorage.getItem("axolDiagArm") as ArmSide) || "left"
  )
  const [windowSec, setWindowSec] = useState(120)
  const [hiddenJoints, setHiddenJoints] = useState<Set<JointName>>(new Set())
  // Zoom/pan pins the charts to a fixed range; null follows the live edge.
  const [pinnedView, setPinnedView] = useState<ChartView | null>(null)

  const [runs, setRuns] = useState<DiagnosticsRunMeta[]>([])
  const [runsLoading, setRunsLoading] = useState(false)

  // One diagnostics launch at a time (the CAN bus has a single owner). Both
  // the action cards and the ad-hoc CAN buttons go through this.
  const [activeRun, setActiveRun] = useState<{
    command: string
    session: SessionInfo
  } | null>(null)
  const [launchBusy, setLaunchBusy] = useState(false)
  const { lines: activeLines, status: activeStatus } = useSessionLogs(
    activeRun?.session.id ?? null
  )
  // Hands-on steps (the ROM tests' gripper prompts) print a "[prompt] …"
  // marker and then block on stdin; the run's Continue button answers them.
  // Prompts are strictly ordered, so the pending one is the (answered+1)th.
  const prompts = useMemo(
    () =>
      activeLines
        .filter((l) => l.startsWith("[prompt] "))
        .map((l) => l.slice("[prompt] ".length).trim()),
    [activeLines]
  )
  const [answered, setAnswered] = useState(0)
  const pendingPrompt = answered < prompts.length ? prompts[answered] : null
  // Latest meaningful output line, surfaced on the running action card as
  // secondary progress context.
  const activeLine =
    [...activeLines]
      .reverse()
      .find((l) => l.trim() && !l.startsWith("[serve]") && !l.startsWith("[prompt] ")) ?? null

  const stream = useTelemetryStream(serverOk)

  // Connect to the serve host saved by the control panel (same-origin when
  // the panel is served by the robot itself).
  useEffect(() => {
    setServerBase(localStorage.getItem("axolServerHost") ?? "")
    let active = true
    fetchCommands()
      .then((cmds) => {
        if (!active) return
        setCommands(cmds)
        setServerOk(true)
      })
      .catch((e) => {
        if (active) toast.error(`Can't reach axol serve: ${e}`)
      })
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Robot status poll (reachability counts, connect gating).
  useEffect(() => {
    if (!serverOk) return
    let active = true
    const poll = () => {
      fetchRobotStatus()
        .then((r) => {
          if (active) setRobot(r)
        })
        .catch(() => {})
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [serverOk])

  const refreshRuns = useCallback(() => {
    setRunsLoading(true)
    fetchDiagnosticsRuns()
      .then(({ runs }) => setRuns(runs))
      .catch(() => {})
      .finally(() => setRunsLoading(false))
  }, [])

  const clearRuns = useCallback(() => {
    setRunsLoading(true)
    clearDiagnosticsRuns()
      .then(() => setRuns([]))
      .catch((e) => toast.error(String(e)))
      .finally(() => setRunsLoading(false))
  }, [toast])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on connect
    if (serverOk) refreshRuns()
  }, [serverOk, refreshRuns])

  // Completion feedback for the active run: toast + refresh the history once
  // its session reaches a terminal state.
  const notifiedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!activeRun || !activeStatus) return
    if (activeStatus.status !== "exited" && activeStatus.status !== "error") return
    if (notifiedRef.current === activeRun.session.id) return
    notifiedRef.current = activeRun.session.id
    const label =
      commands.find((c) => c.id === activeRun.command)?.label ?? activeRun.command
    if (activeStatus.status === "exited" && (activeStatus.exitCode ?? 0) === 0) {
      toast.success(`${label} finished.`)
    } else {
      toast.error(`${label} failed — see its run in the history for the log.`)
    }
    const t = setTimeout(() => {
      refreshRuns()
      setActiveRun(null)
    }, 800)
    return () => clearTimeout(t)
  }, [activeRun, activeStatus, commands, refreshRuns, toast])

  const launch = useCallback(
    async (command: string, args: Record<string, FormValue>) => {
      setLaunchBusy(true)
      try {
        const { run, session } = await startDiagnosticsRun(command, args)
        setAnswered(0)
        setActiveRun({ command, session })
        if (run) setRuns((prev) => [run, ...prev])
      } catch (e) {
        toast.error(String(e))
      } finally {
        setLaunchBusy(false)
      }
    },
    [toast]
  )

  const continuePrompt = useCallback(async () => {
    if (!activeRun) return
    // Optimistically advance so the button hides until the next prompt marker
    // arrives — this also guards against a double-click sending two newlines
    // (which would skip the following prompt).
    setAnswered((n) => n + 1)
    try {
      await sendSessionInput(activeRun.session.id)
    } catch (e) {
      setAnswered((n) => Math.max(0, n - 1))
      toast.error(String(e))
    }
  }, [activeRun, toast])

  const stopActive = useCallback(async () => {
    if (!activeRun) return
    setLaunchBusy(true)
    try {
      await stopSession(activeRun.session.id)
    } catch (e) {
      toast.error(String(e))
    } finally {
      setLaunchBusy(false)
    }
  }, [activeRun, toast])

  const connectRobot = useCallback(async () => {
    setRobotBusy(true)
    try {
      setRobot(await robotConnect())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setRobotBusy(false)
    }
  }, [toast])

  // Auto-connect the robot link once after the host comes online if it's
  // sitting idle — same one-shot latch as the control panel, so a manual
  // disconnect elsewhere isn't immediately undone.
  const autoRobotRef = useRef(false)
  useEffect(() => {
    if (!serverOk) {
      autoRobotRef.current = false
      return
    }
    if (autoRobotRef.current || !robot) return
    autoRobotRef.current = true
    if (robot.state === "disconnected" && !robotBusy) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot auto-connect on host online
      connectRobot()
    }
  }, [serverOk, robot, robotBusy, connectRobot])

  function selectArm(a: ArmSide) {
    setArm(a)
    localStorage.setItem("axolDiagArm", a)
  }

  const series: ChartSeries[] = useMemo(
    () =>
      JOINTS.filter((j) => !hiddenJoints.has(j)).map((joint) => ({
        key: motorKey(arm, joint),
        label: jointLabel(joint),
        color: JOINT_COLORS[joint],
      })),
    [arm, hiddenJoints]
  )

  const linkState = robot?.state ?? stream.state
  const stateBadge = STATE_BADGE[linkState] ?? STATE_BADGE.disconnected
  const quietReason =
    linkState === "busy"
      ? "paused — a test or operation owns the bus"
      : linkState !== "connected"
        ? "robot link down"
        : null
  // The CAN bus is owned by something we didn't launch (an in-process
  // operation like teleop) — the server would reject a diagnostic launch, so
  // gray out the launchers rather than let a click bounce off a 409.
  const busyElsewhere = linkState === "busy" && activeRun == null

  const diagCommands = useMemo(
    () => commands.filter((c) => c.category === "Diagnostics"),
    [commands]
  )
  const canCommand = (id: string) => commands.find((c) => c.id === id) ?? null

  // Motor calibration tools surfaced as buttons in the Motors header. Zeroing
  // is one button whose dialog tabs between "specific motor" and the guided
  // walk of every joint (both back motor.set-zero-pos via presets).
  const MOTOR_TOOLS: {
    key: string
    command: string
    label: string
    icon: typeof Tag
    description?: string
    modes?: ActionMode[]
  }[] = [
    {
      key: "set-can-id",
      command: "motor.set-can-id",
      label: "Set CAN ID",
      icon: Tag,
    },
    {
      key: "zero",
      command: "motor.set-zero-pos",
      label: "Set zero position",
      icon: Crosshair,
      modes: [
        {
          key: "single",
          label: "Specific motor",
          description:
            "Set one motor's zero to its current mechanical position (persisted to " +
            "flash). Damiao motors need a power cycle afterwards.",
          hideKeys: ["guided"],
        },
        {
          key: "guided",
          label: "Guided",
          description:
            "Walk every joint of one arm and zero each against its closer end stop — " +
            "each step pauses with instructions and a Continue button.",
          presetArgs: { guided: true },
          hideKeys: ["id", "type"],
        },
      ],
    },
  ]
  const [motorTool, setMotorTool] = useState<string | null>(null)
  const openTool = MOTOR_TOOLS.find((t) => t.key === motorTool) ?? null
  const openToolSpec = openTool ? canCommand(openTool.command) : null

  const activeLabel = activeRun
    ? (commands.find((c) => c.id === activeRun.command)?.label ?? activeRun.command)
    : null

  // Follow mode anchors the window to the newest sample; the page re-renders
  // on every stream tick, so the live edge advances with the data (and holds
  // still while the stream is paused). Zoom/pan pins a fixed range.
  const lastT =
    stream.frames.length > 0 ? stream.frames[stream.frames.length - 1].t : windowSec
  const view: ChartView = pinnedView ?? { t0: lastT - windowSec, t1: lastT }

  return (
    <div className="min-h-screen">
      <SiteNav
        current="diagnostics"
        right={
          <span className="mr-1 hidden items-center gap-2 sm:flex">
            <Activity className="size-4 text-white/30" />
            <Badge variant={stateBadge.variant}>{stateBadge.text}</Badge>
          </span>
        }
      />
      <main className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-8">
        {/* Robot link gate */}
        {robot && robot.state === "disconnected" && (
          <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
            <p className="text-sm text-white/60">
              The robot link is disconnected — connect to start streaming motor telemetry.
            </p>
            <Button size="sm" className="ml-auto" onClick={connectRobot} disabled={robotBusy}>
              {robotBusy ? <Loader2 className="animate-spin" /> : null} Connect robot
            </Button>
          </div>
        )}
        {robot?.state === "error" && robot.error && (
          <p className="rounded-lg border border-red-400/25 bg-red-400/[0.05] p-3 text-xs text-red-200/80">
            {robot.error}
          </p>
        )}

        {/* Motor status */}
        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-heading text-base font-semibold">Motors</h2>
            {robot && (
              <span className="text-xs text-white/40">
                {robot.reachableCount}/{robot.motorCount} reachable
              </span>
            )}
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {(["can.setup", "can.enable"] as const).map((id) => {
                const cmd = canCommand(id)
                if (!cmd) return null
                const running = activeRun?.command === id
                return (
                  <Button
                    key={id}
                    variant="outline"
                    size="sm"
                    title={cmd.description}
                    disabled={
                      !serverOk ||
                      launchBusy ||
                      busyElsewhere ||
                      (activeRun != null && !running)
                    }
                    onClick={() => (running ? stopActive() : launch(id, {}))}
                  >
                    {running ? (
                      <Loader2 className="animate-spin" />
                    ) : id === "can.setup" ? (
                      <Wrench />
                    ) : (
                      <Cable />
                    )}
                    {running ? `Stop ${cmd.label}` : cmd.label}
                  </Button>
                )
              })}
              {MOTOR_TOOLS.map((tool) => {
                const cmd = canCommand(tool.command)
                if (!cmd) return null
                return (
                  <Button
                    key={tool.key}
                    variant="outline"
                    size="sm"
                    title={tool.description ?? cmd.description}
                    disabled={!serverOk || busyElsewhere}
                    onClick={() => setMotorTool(tool.key)}
                  >
                    <tool.icon /> {tool.label}
                  </Button>
                )
              })}
            </div>
          </div>
          {(["left", "right"] as ArmSide[]).map((side) => (
            <div key={side} className="flex flex-col gap-2">
              <span className="text-xs font-medium tracking-wide text-white/45 uppercase">
                {side} arm
              </span>
              <MotorGrid
                arm={side}
                slow={stream.slow}
                frames={stream.frames}
                version={stream.version}
                canInspect={linkState === "connected"}
              />
            </div>
          ))}
        </section>

        {/* Live charts */}
        <section className="flex flex-col gap-3">
          {/* One filter row scoping every chart below it. */}
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="mr-2 font-heading text-base font-semibold">Live telemetry</h2>
            <div className="flex overflow-hidden rounded-md border border-white/10">
              {WINDOWS.map((w) => (
                <button
                  key={w.seconds}
                  type="button"
                  onClick={() => {
                    setWindowSec(w.seconds)
                    setPinnedView(null)
                  }}
                  className={cn(
                    "px-2.5 py-1 text-xs transition-colors",
                    pinnedView == null && windowSec === w.seconds
                      ? "bg-[#eff483]/15 text-[#eff483]"
                      : "text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {w.label}
                </button>
              ))}
            </div>
            <div className="flex overflow-hidden rounded-md border border-white/10">
              {(["left", "right"] as ArmSide[]).map((a) => (
                <button
                  key={a}
                  type="button"
                  onClick={() => selectArm(a)}
                  className={cn(
                    "px-2.5 py-1 text-xs capitalize transition-colors",
                    arm === a
                      ? "bg-[#eff483]/15 text-[#eff483]"
                      : "text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {a} arm
                </button>
              ))}
            </div>
            {pinnedView != null && (
              <Button
                variant="ghost"
                size="sm"
                className="text-[#eff483]/90"
                onClick={() => setPinnedView(null)}
              >
                <Radio /> Go live
              </Button>
            )}
          </div>
          <JointFilter hidden={hiddenJoints} onChange={setHiddenJoints} />
          <p className="text-xs text-white/30">
            Scroll to zoom, drag to pan — zooming pauses the live follow until you go live again.
          </p>
          {/* Stacked full-width so each chart gets real reading space; the
              header button on each takes it truly full screen. */}
          <div className="grid grid-cols-1 gap-4">
            <TelemetryChart
              title="Position"
              unit="rad"
              series={series}
              frames={stream.frames}
              version={stream.version}
              metric={0}
              view={view}
              onViewChange={setPinnedView}
              quietReason={quietReason}
              height={300}
            />
            <TelemetryChart
              title="Velocity"
              unit="rad/s"
              series={series}
              frames={stream.frames}
              version={stream.version}
              metric={1}
              view={view}
              onViewChange={setPinnedView}
              quietReason={quietReason}
              height={300}
            />
            <TelemetryChart
              title="Torque"
              unit="Nm"
              series={series}
              frames={stream.frames}
              version={stream.version}
              metric={2}
              view={view}
              onViewChange={setPinnedView}
              quietReason={quietReason}
              height={300}
            />
          </div>
        </section>

        {/* Diagnostics actions */}
        <section className="flex flex-col gap-3">
          <h2 className="font-heading text-base font-semibold">Diagnostics</h2>
          <DiagnosticActions
            commands={diagCommands}
            activeCommand={activeRun?.command ?? null}
            activeSince={activeRun?.session.startedAt ?? null}
            busy={launchBusy}
            disabled={!serverOk || busyElsewhere}
            onLaunch={launch}
            onStop={stopActive}
          />
        </section>

        {/* Run history */}
        <RunHistory
          runs={runs}
          loading={runsLoading}
          onRefresh={refreshRuns}
          onClear={clearRuns}
        />
      </main>

      {/* Motor calibration tool dialog (Set CAN ID / Set zero position) */}
      {openTool && openToolSpec && (
        <ActionDialog
          spec={openToolSpec}
          title={openTool.label}
          description={openTool.description}
          modes={openTool.modes}
          running={activeRun?.command === openTool.command}
          blocked={activeRun != null && activeRun.command !== openTool.command}
          busy={launchBusy}
          disabled={!serverOk}
          onLaunch={(args) => {
            launch(openTool.command, args)
            setMotorTool(null)
          }}
          onStop={stopActive}
          onClose={() => setMotorTool(null)}
        />
      )}

      {/* Floating status for the run in flight: live line, hands-on prompts
          (Continue), and Stop — wherever the run was launched from. */}
      {activeRun && activeLabel && (
        <ActiveRunPanel
          label={activeLabel}
          since={activeRun.session.startedAt}
          line={activeLine}
          prompt={pendingPrompt}
          busy={launchBusy}
          onContinue={continuePrompt}
          onStop={stopActive}
        />
      )}
    </div>
  )
}
