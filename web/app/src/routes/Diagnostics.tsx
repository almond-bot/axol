import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  Cable,
  Crosshair,
  ClipboardList,
  Loader2,
  Radio,
  SlidersHorizontal,
  Tag,
  Upload,
  Wrench,
} from "lucide-react"
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
import { CanAdapterDialog } from "@/components/diagnostics/can-adapter-dialog"
import {
  TelemetryChart,
  type ChartSeries,
  type ChartView,
} from "@/components/diagnostics/telemetry-chart"
import { cn } from "@/lib/utils"
import {
  autoConnectPollStateKnown,
  autoConnectRetryDelay,
  autoConnectSignature,
  canDiscoveryAttemptSignature,
  canDiscoveryBlocksAutoConnect,
  chooseDiagnosticsAutoConnectProfile,
  nextAutoConnectAttempt,
  shouldStartCanDiscovery,
} from "@/lib/can-auto-connect"
import {
  canDiscoveryRequestCanRetry,
  discoverCanHardware,
  fetchCanInterfaces,
  fetchCommands,
  fetchRobotStatus,
  fetchSessions,
  flattenFields,
  robotConnect,
  sendSessionInput,
  setServerBase,
  stopSession,
  useSessionLogs,
  type CanDiscoveryState,
  type CanInterfaceInventory,
  type CanProfileInventory,
  type CommandSpec,
  type FormValue,
  type HardwareProfile,
  type RobotChannels,
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
  jointsFor,
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

// Commands this dashboard launches as manager sessions besides the fetched
// Diagnostics-category catalog: the CAN quick buttons and the motor calibration
// tools. Used to recognise an already-running session and adopt it (see the
// adoption effect) so its Stop button shows on any browser, not just the tab
// that started it.
const PAGE_COMMAND_IDS = [
  "tracker.pair",
  "tracker.identify",
  "can.setup",
  "can.enable",
  "motor.set-can-id",
  "motor.set-zero-pos",
  "motor.dump-config",
  "motor.set-config",
  "motor.flash",
]

// The Axol hub adapter's persistent interface names (created by can.setup).
// A configured channel equal to its hub default is omitted from launches so
// the commands run their own defaults.
const HARDWARE_DEFAULTS: Record<HardwareProfile, Record<"left" | "right", string>> = {
  axol: { left: "can_alm_axol_l", right: "can_alm_axol_r" },
  mantis: { left: "can_mantis_l", right: "can_mantis_r" },
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
  const robotStatusPollFailedRef = useRef(false)
  const robotStatusRecoveryEpochRef = useRef(0)
  const robotStatusKnownRef = useRef(false)
  const canInventoryKnownRef = useRef(false)
  const sessionInventoryKnownRef = useRef(false)
  const autoRobotPollStateKnown = useCallback(
    () =>
      autoConnectPollStateKnown(
        robotStatusKnownRef.current,
        canInventoryKnownRef.current,
        sessionInventoryKnownRef.current
      ),
    []
  )
  const [canProfiles, setCanProfiles] = useState<CanProfileInventory | null>(null)
  const [canDiscovery, setCanDiscovery] = useState<CanDiscoveryState | null>(null)
  const automaticCanDiscoveryAttemptsRef = useRef(new Set<string>())
  const canDiscoveryNoticesRef = useRef(new Set<string>())
  // A successful response without profile summaries (or a 404) identifies an
  // older host. It cannot distinguish attached roles, so only retain the
  // historical profile already reported by that host's robot status.
  const [legacyCanInventory, setLegacyCanInventory] = useState(false)
  const installCanInventory = useCallback((inventory: CanInterfaceInventory) => {
    setCanDiscovery(inventory.discovery ?? null)
    if (inventory.profiles) {
      setCanProfiles(inventory.profiles)
      setLegacyCanInventory(false)
    } else {
      setCanProfiles(null)
      setLegacyCanInventory(true)
    }
  }, [])

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
  const [sessionInventoryReady, setSessionInventoryReady] = useState(false)
  const [hardwareSessionBusy, setHardwareSessionBusy] = useState(false)
  const [launchBusy, setLaunchBusy] = useState(false)
  const { lines: activeLines, status: activeStatus } = useSessionLogs(activeRun?.session.id ?? null)
  // Hands-on steps (the ROM tests' gripper prompts) print a "[prompt] …" marker
  // and then block on stdin; the run's Continue button answers them. The run is
  // waiting on one iff its most recent output is that marker — after the operator
  // answers, the script always prints more output, so a non-prompt tail means
  // nothing is pending. Deriving the pending prompt from the log tail (instead of
  // counting how many were answered) keeps it correct on any browser, including
  // one that adopted a run already in flight from another computer.
  const promptTail = useMemo(() => {
    for (let i = activeLines.length - 1; i >= 0; i--) {
      const l = activeLines[i]
      if (!l.trim() || l.startsWith("[serve]")) continue
      return l.startsWith("[prompt] ") ? l.slice("[prompt] ".length).trim() : null
    }
    return null
  }, [activeLines])
  // Hide the Continue button the instant it's clicked (until the next line
  // arrives) so a double-click can't send two newlines and skip the following
  // prompt. Tagged with the session id so the suppression never bleeds from one
  // run into the next.
  const [dismissed, setDismissed] = useState<{ id: string; len: number } | null>(null)
  const pendingPrompt =
    promptTail &&
    !(dismissed && dismissed.id === activeRun?.session.id && dismissed.len === activeLines.length)
      ? promptTail
      : null
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

  // Observe the server-owned link and configured CAN presence. Diagnostics may
  // bootstrap a disconnected host only when exactly one profile is detected;
  // it never chooses between two profiles or switches an open link.
  useEffect(() => {
    if (!serverOk) {
      robotStatusKnownRef.current = false
      canInventoryKnownRef.current = false
      return
    }
    let active = true
    const poll = () => {
      fetchRobotStatus()
        .then((r) => {
          if (!active) return
          if (robotStatusPollFailedRef.current) {
            robotStatusPollFailedRef.current = false
            robotStatusRecoveryEpochRef.current += 1
          }
          robotStatusKnownRef.current = true
          setRobot(r)
        })
        .catch(() => {
          if (active) {
            robotStatusPollFailedRef.current = true
            robotStatusKnownRef.current = false
            setRobot(null)
          }
        })
      fetchCanInterfaces()
        .then((inventory) => {
          if (!active) return
          canInventoryKnownRef.current = true
          installCanInventory(inventory)
        })
        .catch((error) => {
          if (!active) return
          if (String(error).includes("HTTP 404")) {
            canInventoryKnownRef.current = true
            setCanProfiles(null)
            setCanDiscovery(null)
            setLegacyCanInventory(true)
            return
          }
          canInventoryKnownRef.current = false
          setCanProfiles(null)
          setCanDiscovery(null)
          setLegacyCanInventory(false)
        })
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [installCanInventory, serverOk])

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
    const label = commands.find((c) => c.id === activeRun.command)?.label ?? activeRun.command
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

  // Fold the configured left/right adapter mapping (set in the CAN enable
  // dialog, persisted on the host) into every launch, keyed by which channel
  // arguments the command accepts: the two-arm tests get per-arm interface
  // overrides plus a --no-left/--no-right skip for an arm with no adapter;
  // the single-arm tools get --channel for whichever arm the dialog picked.
  // Interface names never need to be typed into a run dialog.
  const channelArgs = useCallback(
    (command: string, args: Record<string, FormValue>): Record<string, FormValue> => {
      const spec = commands.find((c) => c.id === command)
      const channels = robot?.channels
      if (!spec || !channels) return {}
      const profile = robot.profile ?? "axol"
      const defaults = HARDWARE_DEFAULTS[profile]
      const keys = new Set(flattenFields(spec.schema).map((f) => f.key))
      const out: Record<string, FormValue> = {}
      if (keys.has("target")) out.target = profile
      if (profile === "mantis" && keys.has("joints")) out.joints = "gripper"
      if (keys.has("left_channel")) {
        if (channels.left && channels.left !== defaults.left) out.left_channel = channels.left
        if (!channels.left && keys.has("no_left")) out.no_left = true
      }
      if (keys.has("right_channel")) {
        if (channels.right && channels.right !== defaults.right) out.right_channel = channels.right
        if (!channels.right && keys.has("no_right")) out.no_right = true
      }
      if (keys.has("channel") && keys.has("arm")) {
        const side: "left" | "right" = args.arm === "right" ? "right" : "left"
        const chan = channels[side]
        if (chan && chan !== defaults[side]) out.channel = chan
      }
      return out
    },
    [commands, robot]
  )

  const launch = useCallback(
    async (command: string, args: Record<string, FormValue>) => {
      setLaunchBusy(true)
      try {
        const { run, session } = await startDiagnosticsRun(command, {
          ...args,
          ...channelArgs(command, args),
        })
        setActiveRun({ command, session })
        if (run) setRuns((prev) => [run, ...prev])
      } catch (e) {
        toast.error(String(e))
      } finally {
        setLaunchBusy(false)
      }
    },
    [toast, channelArgs]
  )

  const continuePrompt = useCallback(async () => {
    if (!activeRun) return
    // Hide the button until the next line arrives (double-click guard); restore
    // it if the input fails to send.
    setDismissed({ id: activeRun.session.id, len: activeLines.length })
    try {
      await sendSessionInput(activeRun.session.id)
    } catch (e) {
      setDismissed(null)
      toast.error(String(e))
    }
  }, [activeRun, activeLines.length, toast])

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

  const activeProfile = robot?.profile ?? "axol"
  const manualRobotOverrideRef = useRef(false)
  const autoRobotRef = useRef<string | null>(null)
  const autoRobotAttemptsRef = useRef(new Map<string, number>())
  const autoRobotRetryTimerRef = useRef<number | null>(null)
  const autoRobotMountedRef = useRef(true)
  const [autoRobotRetryRevision, setAutoRobotRetryRevision] = useState(0)
  const resetAutoRobotRetry = useCallback(() => {
    autoRobotRef.current = null
    autoRobotAttemptsRef.current.clear()
    if (autoRobotRetryTimerRef.current !== null) {
      window.clearTimeout(autoRobotRetryTimerRef.current)
      autoRobotRetryTimerRef.current = null
    }
  }, [])
  useEffect(() => {
    autoRobotMountedRef.current = true
    return () => {
      autoRobotMountedRef.current = false
      resetAutoRobotRetry()
    }
  }, [resetAutoRobotRetry])
  const connectRobot = useCallback(
    async (profile = activeProfile, automatic = false): Promise<boolean> => {
      if (!automatic) {
        resetAutoRobotRetry()
      }
      setRobotBusy(true)
      try {
        const status = await robotConnect(undefined, profile, automatic)
        if (!autoRobotMountedRef.current) return false
        setRobot(status)
        if (!status.connected) {
          throw new Error(status.error ?? `Could not connect the ${profile} CAN link`)
        }
        if (!automatic) manualRobotOverrideRef.current = true
        return true
      } catch (e) {
        if (!autoRobotMountedRef.current) return false
        if (!automatic) toast.error(String(e))
        return false
      } finally {
        if (autoRobotMountedRef.current) setRobotBusy(false)
      }
    },
    [toast, activeProfile, resetAutoRobotRetry]
  )

  // Manual CAN interface selection — the fallback when the Axol hub adapter
  // (and its auto-named interfaces) can't be found. The server persists the
  // choice, so later connects and operations reuse it.
  const [adapterOpen, setAdapterOpen] = useState(false)
  const connectWithChannels = useCallback(
    async (profile: HardwareProfile, channels: RobotChannels) => {
      resetAutoRobotRetry()
      setRobotBusy(true)
      try {
        const status = await robotConnect(channels, profile)
        setRobot(status)
        if (!status.connected) {
          throw new Error(status.error ?? `Could not connect the ${profile} CAN link`)
        }
        manualRobotOverrideRef.current = true
        setAdapterOpen(false)
      } catch (e) {
        toast.error(String(e))
      } finally {
        setRobotBusy(false)
      }
    },
    [resetAutoRobotRetry, toast]
  )

  // Identify a fresh anonymous dual-channel hub before Diagnostics decides
  // whether exactly one hardware profile is safe to connect. The backend
  // coalesces requests across tabs; this latch avoids duplicates from this
  // route's inventory poll.
  useEffect(() => {
    if (!serverOk) {
      automaticCanDiscoveryAttemptsRef.current.clear()
      canDiscoveryNoticesRef.current.clear()
      return
    }
    if (!canDiscovery) return
    if (canDiscovery.status === "ready" && canDiscovery.candidateCount === 0) {
      automaticCanDiscoveryAttemptsRef.current.clear()
      return
    }
    const signature = canDiscoveryAttemptSignature(canDiscovery)
    const hardwareIdle = !activeRun && !hardwareSessionBusy && !launchBusy && !robotBusy
    if (
      signature === null ||
      !shouldStartCanDiscovery(canDiscovery, robot?.state, autoRobotPollStateKnown(), hardwareIdle)
    )
      return

    const hostEpoch = robotStatusRecoveryEpochRef.current
    const attemptKey = `${hostEpoch}:${signature}`
    if (automaticCanDiscoveryAttemptsRef.current.has(attemptKey)) return
    automaticCanDiscoveryAttemptsRef.current.add(attemptKey)

    void discoverCanHardware()
      .then((inventory) => {
        if (!autoRobotMountedRef.current || hostEpoch !== robotStatusRecoveryEpochRef.current)
          return
        installCanInventory(inventory)
      })
      .catch((error) => {
        if (!autoRobotMountedRef.current || hostEpoch !== robotStatusRecoveryEpochRef.current)
          return
        if (canDiscoveryRequestCanRetry(error)) {
          automaticCanDiscoveryAttemptsRef.current.delete(attemptKey)
        }
        const noticeKey = `request:${attemptKey}`
        if (canDiscoveryNoticesRef.current.has(noticeKey)) return
        canDiscoveryNoticesRef.current.add(noticeKey)
        toast.error(`Automatic CAN discovery failed: ${String(error)}`)
      })
  }, [
    activeRun,
    autoRobotPollStateKnown,
    canDiscovery,
    hardwareSessionBusy,
    installCanInventory,
    launchBusy,
    robot?.state,
    robotBusy,
    serverOk,
    toast,
  ])

  useEffect(() => {
    if (
      !serverOk ||
      !canDiscovery ||
      (canDiscovery.status !== "partial" &&
        canDiscovery.status !== "unidentified" &&
        canDiscovery.status !== "error")
    )
      return
    const noticeKey = `${robotStatusRecoveryEpochRef.current}:${canDiscovery.generation}:${canDiscovery.status}`
    if (canDiscoveryNoticesRef.current.has(noticeKey)) return
    canDiscoveryNoticesRef.current.add(noticeKey)
    const fallback =
      canDiscovery.status === "unidentified"
        ? "Power the attached hardware, then unplug and reconnect its USB hub to retry, or run axol can.setup in a terminal."
        : "Run axol can.setup in a terminal if the problem continues."
    if (canDiscovery.status === "partial") toast.warning(canDiscovery.message ?? fallback)
    else toast.error(canDiscovery.message ?? fallback)
  }, [canDiscovery, serverOk, toast])

  // Direct navigation to Diagnostics still brings up an unambiguous host, but
  // this route is never a second profile-policy authority. With both profiles
  // attached it waits for an explicit choice, and an already-open link is only
  // observed (even if another control panel selected a different operation).
  useEffect(() => {
    if (!serverOk) {
      resetAutoRobotRetry()
      manualRobotOverrideRef.current = false
      return
    }
    if (!robot || (!canProfiles && !legacyCanInventory) || !sessionInventoryReady) return
    if (
      canDiscoveryBlocksAutoConnect(canDiscovery) ||
      activeRun ||
      hardwareSessionBusy ||
      launchBusy ||
      robotBusy ||
      (robot.state !== "disconnected" && robot.state !== "error") ||
      manualRobotOverrideRef.current ||
      !autoRobotPollStateKnown()
    )
      return

    const target = chooseDiagnosticsAutoConnectProfile(
      canProfiles,
      legacyCanInventory,
      activeProfile
    )
    if (target === null) {
      resetAutoRobotRetry()
      return
    }
    const profileSignature = canProfiles
      ? autoConnectSignature(target, canProfiles[target])
      : `legacy:${target}`
    const signature = `${profileSignature}:host-${robotStatusRecoveryEpochRef.current}`
    if (autoRobotRef.current === signature) return
    const attempts = nextAutoConnectAttempt(
      autoRobotAttemptsRef.current.get(signature) ?? 0,
      autoRobotPollStateKnown()
    )
    if (attempts === null) {
      autoRobotRef.current = signature
      return
    }
    autoRobotRef.current = signature
    void connectRobot(target, true).then((connected) => {
      if (connected || !autoRobotMountedRef.current || manualRobotOverrideRef.current) return
      if (!autoRobotPollStateKnown()) {
        autoRobotRef.current = null
        return
      }
      autoRobotAttemptsRef.current.set(signature, attempts)
      const delay = autoConnectRetryDelay(attempts)
      if (delay === null) return
      if (autoRobotRetryTimerRef.current !== null) {
        window.clearTimeout(autoRobotRetryTimerRef.current)
      }
      autoRobotRetryTimerRef.current = window.setTimeout(() => {
        autoRobotRetryTimerRef.current = null
        if (
          autoRobotMountedRef.current &&
          !manualRobotOverrideRef.current &&
          autoRobotRef.current === signature
        ) {
          autoRobotRef.current = null
          // A later successful poll causes the rerender and retains the same
          // attempt count; unknown authority never spends a retry.
          if (!autoRobotPollStateKnown()) return
          setAutoRobotRetryRevision((revision) => revision + 1)
        }
      }, delay)
    })
  }, [
    activeProfile,
    activeRun,
    autoRobotPollStateKnown,
    autoRobotRetryRevision,
    canDiscovery,
    canProfiles,
    connectRobot,
    hardwareSessionBusy,
    launchBusy,
    legacyCanInventory,
    robot,
    robotBusy,
    resetAutoRobotRetry,
    serverOk,
    sessionInventoryReady,
  ])

  function selectArm(a: ArmSide) {
    setArm(a)
    localStorage.setItem("axolDiagArm", a)
  }

  // The joints this robot actually has — the gripperless SKU drops GRIPPER
  // from the motor tiles, chart series, and `--joints` pickers.
  const joints = useMemo(
    () => (robot?.profile === "mantis" ? (["GRIPPER"] as const) : jointsFor(robot?.hasGripper)),
    [robot]
  )

  const series: ChartSeries[] = useMemo(
    () =>
      joints
        .filter((j) => !hiddenJoints.has(j))
        .map((joint) => ({
          key: motorKey(arm, joint),
          label: jointLabel(joint),
          color: JOINT_COLORS[joint],
        })),
    [arm, hiddenJoints, joints]
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
  const busyElsewhere =
    activeRun == null && (linkState === "busy" || hardwareSessionBusy || launchBusy)

  const diagCommands = useMemo(
    () =>
      commands.filter(
        (c) =>
          c.category === "Diagnostics" &&
          (!c.hardwareProfiles || c.hardwareProfiles.includes(robot?.profile ?? "axol"))
      ),
    [commands, robot?.profile]
  )
  const canCommand = (id: string) => commands.find((c) => c.id === id) ?? null

  // The configured arm→adapter mapping drives the whole page: the only arm
  // with an adapter (single-arm setups) is pre-picked in the calibration
  // tools, and channel fields never appear in run dialogs — they're injected
  // by `channelArgs`. `no_left`/`no_right` stay visible only for arms that
  // have an adapter (skipping a configured arm is still a per-run choice).
  const onlySide: ArmSide | null =
    robot?.channels && robot.channels.left && !robot.channels.right
      ? "left"
      : robot?.channels && robot.channels.right && !robot.channels.left
        ? "right"
        : null
  const configHiddenKeys = useMemo(() => {
    const keys = ["left_channel", "right_channel", "channel", "target"]
    if (robot?.profile === "mantis") keys.push("joints")
    if (robot?.channels && !robot.channels.left) keys.push("no_left")
    if (robot?.channels && !robot.channels.right) keys.push("no_right")
    return keys
  }, [robot])

  // Adopt a diagnostics/CAN run that's already in flight so its Stop button,
  // live output and Continue prompt appear on *any* browser — not just the tab
  // that launched it. Without this, opening the dashboard on a second computer
  // while a run is going gives no way to cancel it. Poll only while nothing is
  // tracked locally; once a run is adopted (or launched) activeRun is set and
  // this stops. The completion effect clears activeRun when the run ends, which
  // re-arms the poll (the finished session is then no longer "live").
  useEffect(() => {
    if (!serverOk) {
      sessionInventoryKnownRef.current = false
      return
    }
    if (activeRun != null) return
    const ours = (command: string) =>
      PAGE_COMMAND_IDS.includes(command) || diagCommands.some((c) => c.id === command)
    let active = true
    const poll = () => {
      fetchSessions()
        .then((sessions) => {
          if (!active) return
          const liveSessions = sessions.filter(
            (session) =>
              session.status === "starting" ||
              session.status === "running" ||
              session.status === "stopping"
          )
          sessionInventoryKnownRef.current = true
          setSessionInventoryReady(true)
          setHardwareSessionBusy(liveSessions.length > 0)
          const live = liveSessions
            .filter((session) => ours(session.command))
            .sort((a, b) => b.startedAt - a.startedAt)[0]
          if (live) setActiveRun({ command: live.command, session: live })
        })
        .catch(() => {
          if (!active) return
          sessionInventoryKnownRef.current = false
          setSessionInventoryReady(false)
        })
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [serverOk, activeRun, diagCommands])

  // Per-motor service tools surfaced as buttons in the Motors header: CAN ID,
  // zeroing, configuration parameters, and firmware. A tool with modes tabs
  // between presets of one command — zeroing between "specific motor" and the
  // guided walk of every joint, config between reading and writing.
  const MOTOR_TOOLS: {
    key: string
    command: string
    label: string
    icon: typeof Tag
    description?: string
    presetArgs?: Record<string, FormValue>
    hideKeys?: string[]
    modes?: ActionMode[]
    /** Restrict the `--joints` picker's choices for this tool's dialog. */
    pickerJoints?: readonly JointName[]
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
      // The gripper is absent from the guided picker: it self-calibrates
      // against its hard stops at enable time and has no zero to set.
      pickerJoints: JOINTS.filter((j) => j !== "GRIPPER"),
      modes: [
        {
          key: "single",
          label: "Specific motor",
          description:
            "Set one motor's zero to its current mechanical position (persisted to " +
            "flash). Damiao motors need a power cycle afterwards.",
          hideKeys: ["guided", "joints"],
        },
        {
          key: "guided",
          label: "Guided",
          description:
            "Walk the selected joints of one arm and zero each against an end " +
            "stop (wrist_2/wrist_3 accept either side) — each step pauses with " +
            "instructions and a Continue button.",
          presetArgs: { guided: true },
          hideKeys: ["id", "type"],
        },
      ],
    },
    {
      key: "dump-config",
      command: "motor.dump-config",
      label: "Dump config",
      icon: ClipboardList,
      description:
        "Read every configuration parameter from one motor, or all of them when the " +
        "CAN ID is left blank. Works for MyActuator and Damiao alike. Read-only — the " +
        "run log is kept in the history below, so this is what to capture before " +
        "changing anything.",
    },
    {
      key: "set-config",
      command: "motor.set-config",
      label: "Set config",
      icon: SlidersHorizontal,
      modes: [
        {
          key: "read",
          label: "Read",
          description:
            "Read one configuration parameter without writing anything. Parameter " +
            "names cover both motor families, including Damiao's CAN timeout.",
          hideKeys: ["value", "force_protected", "yes"],
        },
        {
          key: "write",
          label: "Write",
          description:
            "Write one configuration parameter and persist it. Damiao's CAN timeout " +
            "is in milliseconds. Protected parameters — factory calibration, CAN IDs, " +
            "baud rate — need the override, since changing those can leave the motor " +
            "unable to commutate or unreachable on the bus.",
          // Running the dialog is the confirmation; the CLI prompt would other-
          // wise block the session waiting on stdin.
          presetArgs: { yes: true },
        },
      ],
    },
    {
      key: "flash",
      command: "motor.flash",
      label: "Flash firmware",
      icon: Upload,
      description:
        "Overwrite one motor's firmware from a .bin file on the robot host. Leave " +
        "the arm powered and idle — nothing else may use the bus. An interrupted " +
        "flash leaves the motor in its bootloader until you run this again.",
      // Running the dialog is the confirmation; the CLI prompt would otherwise
      // block the session waiting on stdin.
      presetArgs: { yes: true },
    },
  ]
  const visibleMotorTools =
    robot?.profile === "mantis"
      ? MOTOR_TOOLS.filter((tool) => !["zero", "flash"].includes(tool.key))
      : MOTOR_TOOLS
  const [motorTool, setMotorTool] = useState<string | null>(null)
  const openTool = MOTOR_TOOLS.find((t) => t.key === motorTool) ?? null
  const openToolSpec = openTool ? canCommand(openTool.command) : null

  const activeLabel = activeRun
    ? (commands.find((c) => c.id === activeRun.command)?.label ?? activeRun.command)
    : null

  // Follow mode anchors the window to the newest sample; the page re-renders
  // on every stream tick, so the live edge advances with the data (and holds
  // still while the stream is paused). Zoom/pan pins a fixed range.
  const lastT = stream.frames.length > 0 ? stream.frames[stream.frames.length - 1].t : windowSec
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
      <main className="safe-x mx-auto flex max-w-6xl flex-col gap-8 py-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:py-8">
        {/* Robot link gate */}
        {robot && robot.state === "disconnected" && (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
            <p className="text-sm text-white/60">
              The {robot.profile === "mantis" ? "Mantis" : "robot"} link is disconnected — connect
              to start streaming motor telemetry.
            </p>
            <div className="ml-auto flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAdapterOpen(true)}
                disabled={robotBusy || hardwareSessionBusy || launchBusy}
                title="Choose Axol or Mantis and the CAN interface(s) to inspect."
              >
                <Cable /> CAN adapter…
              </Button>
              <Button
                size="sm"
                onClick={() => void connectRobot()}
                disabled={robotBusy || hardwareSessionBusy || launchBusy}
              >
                {robotBusy ? <Loader2 className="animate-spin" /> : null} Connect{" "}
                {robot.profile === "mantis" ? "Mantis" : "robot"}
              </Button>
            </div>
          </div>
        )}
        {robot?.state === "error" && robot.error && (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-red-400/25 bg-red-400/[0.05] p-3">
            <p className="text-xs text-red-200/80">{robot.error}</p>
            <div className="ml-auto flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAdapterOpen(true)}
                disabled={robotBusy || hardwareSessionBusy || launchBusy}
                title="Choose Axol or Mantis and the CAN interfaces to inspect."
              >
                <Cable /> Choose CAN adapter
              </Button>
              <Button
                size="sm"
                onClick={() => void connectRobot()}
                disabled={robotBusy || hardwareSessionBusy || launchBusy}
              >
                {robotBusy ? <Loader2 className="animate-spin" /> : null} Retry
              </Button>
            </div>
          </div>
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
                // CAN enable is the page's home for the arm→adapter mapping:
                // it opens the adapter dialog, whose Save & connect persists
                // the choice and brings the interfaces up. Reopen it any time
                // to remap. can.setup stays a direct launch (hub provisioning
                // with visible output).
                const onClick = running
                  ? stopActive
                  : id === "can.enable"
                    ? () => setAdapterOpen(true)
                    : () => launch(id, {})
                return (
                  <Button
                    key={id}
                    variant="outline"
                    size="sm"
                    title={
                      id === "can.enable"
                        ? "Choose which CAN adapter drives each arm (or a single arm) and bring the interfaces up. The mapping is saved and applied to every diagnostic and calibration run."
                        : cmd.description
                    }
                    disabled={
                      !serverOk || launchBusy || busyElsewhere || (activeRun != null && !running)
                    }
                    onClick={onClick}
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
              {visibleMotorTools.map((tool) => {
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
                {side} {robot?.profile === "mantis" ? "gripper" : "arm"}
              </span>
              <MotorGrid
                arm={side}
                slow={stream.slow}
                frames={stream.frames}
                version={stream.version}
                canInspect={linkState === "connected"}
                joints={joints}
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
                  {a} {robot?.profile === "mantis" ? "gripper" : "arm"}
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
          <JointFilter hidden={hiddenJoints} onChange={setHiddenJoints} joints={joints} />
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
            hiddenKeys={configHiddenKeys}
            pickerJoints={joints}
            onLaunch={launch}
            onStop={stopActive}
          />
        </section>

        {/* Run history */}
        <RunHistory runs={runs} loading={runsLoading} onRefresh={refreshRuns} onClear={clearRuns} />
      </main>

      {/* Per-motor service tool dialog (CAN ID / zero / config / firmware) */}
      {openTool && openToolSpec && (
        <ActionDialog
          spec={openToolSpec}
          title={openTool.label}
          description={openTool.description}
          modes={openTool.modes}
          pickerJoints={openTool.pickerJoints}
          // The adapter mapping decides the interface (injected at launch);
          // a single-arm config also decides the arm. The tool's own presets
          // (e.g. skipping a CLI confirmation the dialog already covers) and
          // hidden fields stack on top.
          hideKeys={[...configHiddenKeys, ...(openTool.hideKeys ?? [])]}
          presetArgs={{
            ...(onlySide ? { arm: onlySide } : {}),
            ...(openTool.presetArgs ?? {}),
          }}
          running={activeRun?.command === openTool.command}
          blocked={activeRun != null && activeRun.command !== openTool.command}
          busy={launchBusy}
          disabled={!serverOk || busyElsewhere}
          onLaunch={(args) => {
            launch(openTool.command, args)
            setMotorTool(null)
          }}
          onStop={stopActive}
          onClose={() => setMotorTool(null)}
        />
      )}

      {/* Manual CAN interface selection (non-Axol-hub adapters) */}
      {adapterOpen && (
        <CanAdapterDialog
          profile={robot?.profile ?? "axol"}
          channels={robot?.channels}
          busy={robotBusy || hardwareSessionBusy || launchBusy}
          onConnect={connectWithChannels}
          onClose={() => setAdapterOpen(false)}
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
