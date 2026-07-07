import { useCallback, useEffect, useMemo, useState } from "react"
import { Activity, Loader2, Pause, Play } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { SiteNav } from "@/components/site-nav"
import { useToast } from "@/components/ui/toast"
import { MotorGrid } from "@/components/diagnostics/motor-grid"
import { RunHistory } from "@/components/diagnostics/run-history"
import { ScriptRunner } from "@/components/diagnostics/script-runner"
import { TelemetryChart, type ChartSeries } from "@/components/diagnostics/telemetry-chart"
import { cn } from "@/lib/utils"
import {
  fetchCommands,
  fetchRobotStatus,
  robotConnect,
  setServerBase,
  type CommandSpec,
  type RobotState,
  type RobotStatus,
} from "@/lib/supervisor"
import {
  JOINTS,
  JOINT_COLORS,
  fetchDiagnosticsRuns,
  jointLabel,
  motorKey,
  useTelemetryStream,
  type ArmSide,
  type DiagnosticsRunMeta,
} from "@/lib/telemetry"

const WINDOWS: { label: string; seconds: number }[] = [
  { label: "30s", seconds: 30 },
  { label: "2m", seconds: 120 },
  { label: "10m", seconds: 600 },
]

const STATE_BADGE: Record<RobotState, { variant: "success" | "warning" | "destructive" | "neutral"; text: string }> = {
  connected: { variant: "success", text: "streaming" },
  busy: { variant: "warning", text: "operation owns the bus" },
  connecting: { variant: "neutral", text: "connecting" },
  disconnected: { variant: "neutral", text: "robot disconnected" },
  error: { variant: "destructive", text: "error" },
}

/**
 * Motor diagnostics dashboard: live per-motor status (health, temperature,
 * voltage), always-running position / velocity / torque charts, a launcher for
 * the diagnostics scripts, and the recorded history of past runs.
 *
 * Telemetry streams whenever the idle robot link owns the CAN bus. While an
 * operation or script owns it the stream pauses (single owner) — charts keep
 * their history and show why.
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
  const [paused, setPaused] = useState(false)

  const [runs, setRuns] = useState<DiagnosticsRunMeta[]>([])
  const [runsLoading, setRunsLoading] = useState(false)

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

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on connect
    if (serverOk) refreshRuns()
  }, [serverOk, refreshRuns])

  async function connectRobot() {
    setRobotBusy(true)
    try {
      setRobot(await robotConnect())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  function selectArm(a: ArmSide) {
    setArm(a)
    localStorage.setItem("axolDiagArm", a)
  }

  const series: ChartSeries[] = useMemo(
    () =>
      JOINTS.map((joint) => ({
        key: motorKey(arm, joint),
        label: jointLabel(joint),
        color: JOINT_COLORS[joint],
      })),
    [arm]
  )

  const linkState = robot?.state ?? stream.state
  const stateBadge = STATE_BADGE[linkState] ?? STATE_BADGE.disconnected
  const quietReason =
    linkState === "busy"
      ? "paused — an operation owns the CAN bus"
      : linkState !== "connected"
        ? "robot link down"
        : null

  // The Diagnostics catalog category, plus the CAN bring-up commands (Setup
  // category in the control panel, but part of the motor troubleshooting
  // toolkit: "motors unreachable" usually starts with the CAN interfaces).
  const diagCommands = useMemo(
    () =>
      commands.filter(
        (c) => c.category === "Diagnostics" || c.id === "can.setup" || c.id === "can.enable"
      ),
    [commands]
  )

  // Freeze the live charts by pinning version + anchoring the window to the
  // last frame; the buffer keeps filling so resume snaps back to now.
  const chartVersion = paused ? -1 : stream.version

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
      <main className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-8">
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
        <section className="flex flex-col gap-3">
          <div className="flex items-baseline gap-3">
            <h2 className="font-heading text-base font-semibold">Motors</h2>
            {robot && (
              <span className="text-xs text-white/40">
                {robot.reachableCount}/{robot.motorCount} reachable
              </span>
            )}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
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
          </div>
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
                  onClick={() => setWindowSec(w.seconds)}
                  className={cn(
                    "px-2.5 py-1 text-xs transition-colors",
                    windowSec === w.seconds
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
            <Button
              variant="ghost"
              size="sm"
              className="text-white/50"
              onClick={() => setPaused((v) => !v)}
            >
              {paused ? <Play /> : <Pause />}
              {paused ? "Resume" : "Pause"}
            </Button>
          </div>
          <div className="grid gap-4 xl:grid-cols-3">
            <TelemetryChart
              title="Position"
              unit="rad"
              series={series}
              frames={stream.frames}
              version={chartVersion}
              metric={0}
              windowSec={windowSec}
              live={!paused}
              quietReason={quietReason}
            />
            <TelemetryChart
              title="Velocity"
              unit="rad/s"
              series={series}
              frames={stream.frames}
              version={chartVersion}
              metric={1}
              windowSec={windowSec}
              live={!paused}
              quietReason={quietReason}
            />
            <TelemetryChart
              title="Torque"
              unit="Nm"
              series={series}
              frames={stream.frames}
              version={chartVersion}
              metric={2}
              windowSec={windowSec}
              live={!paused}
              quietReason={quietReason}
            />
          </div>
        </section>

        {/* Script runner */}
        <section className="flex flex-col gap-3">
          <h2 className="font-heading text-base font-semibold">Diagnostics scripts</h2>
          <ScriptRunner
            commands={diagCommands}
            disabled={linkState === "disconnected" || linkState === "error"}
            onRunStarted={(run) => setRuns((prev) => [run, ...prev])}
            onRunFinished={refreshRuns}
          />
        </section>

        {/* Run history */}
        <RunHistory runs={runs} loading={runsLoading} onRefresh={refreshRuns} />
      </main>
    </div>
  )
}
