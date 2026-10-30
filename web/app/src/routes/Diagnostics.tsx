import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { discoveryNeedsRecheck, discoveryNotice, useSettledDiscovery } from "@/lib/hardware-status"
import { Navigate, useLocation, useNavigate } from "react-router-dom"
import {
  Cable,
  Crosshair,
  ClipboardList,
  Loader2,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  Tag,
  Upload,
  Wrench,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AppShell, PageSection } from "@/components/app-shell"
import { SafetyControl } from "@/components/safety-control"
import { motorFaultLabels } from "@/lib/safety-view"
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
import { ControlHealth } from "@/components/diagnostics/control-health"
import { HardwareOverview } from "@/components/diagnostics/hardware-overview"
import { TuningWorkbench } from "@/components/diagnostics/tuning-workbench"
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
  canServerEpoch,
  chooseDiagnosticsAutoConnectProfile,
  claimCanInventoryPollResponse,
  issueCanInventoryPoll,
  newCanInventoryPollSequence,
  nextAutoConnectAttempt,
  shouldStartCanDiscovery,
  voidCanInventoryPolls,
} from "@/lib/can-auto-connect"
import {
  canDiscoveryRequestCanRetry,
  discoverCanHardware,
  fetchCanInterfaces,
  fetchCommands,
  fetchJellyStatus,
  fetchRobotStatus,
  fetchSafetyStatus,
  fetchSessions,
  flattenFields,
  JELLY_DEVICE_LABELS,
  JELLY_DEVICES,
  jellyConnect,
  jellyDisconnect,
  robotConnect,
  robotDisconnect,
  sendSessionInput,
  setServerBase,
  stopSession,
  useSessionLogs,
  type CanDeviceInventory,
  type CanDiscoveryState,
  type CanInterfaceInventory,
  type CanProfileInventory,
  type CommandSpec,
  type FormValue,
  type HardwareProfile,
  type JellyDevice,
  type JellyStatus,
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

/** The live chart's selectable metrics. `slow` charts the 1 Hz sweep buffer
 * (temperature) instead of the 10 Hz fast frames. */
const RAD2DEG = 180 / Math.PI

const METRICS: {
  key: string
  label: string
  title: string
  unit: string
  metric: number
  /** Display multiplier — the stream carries radians, humans read degrees. */
  scale?: number
  slow?: boolean
}[] = [
  { key: "pos", label: "Position", title: "Position", unit: "°", metric: 0, scale: RAD2DEG },
  { key: "vel", label: "Velocity", title: "Velocity", unit: "°/s", metric: 1, scale: RAD2DEG },
  { key: "torque", label: "Torque", title: "Torque", unit: "Nm", metric: 2 },
  {
    key: "temp",
    label: "Temp",
    title: "Temperature",
    unit: "°C",
    metric: 0,
    slow: true,
  },
]

const STATE_BADGE: Record<
  RobotState,
  { variant: "success" | "warning" | "destructive" | "neutral"; text: string }
> = {
  connected: { variant: "success", text: "Streaming" },
  busy: { variant: "warning", text: "Bus in use" },
  connecting: { variant: "neutral", text: "Connecting" },
  disconnected: { variant: "neutral", text: "Robot disconnected" },
  error: { variant: "destructive", text: "Error" },
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
 * Telemetry streams whenever the robot link is up. Idle, the link polls the
 * motors itself; while a diagnostic or operation owns command of the bus,
 * the server decodes the task's own CAN traffic through a passive observer
 * socket, so the charts stay live throughout.
 */
/** The diagnostics pages, listed under Diagnostics in the sidebar. */
const DIAGNOSTICS_TABS = [
  { key: "", label: "Live" },
  { key: "tests", label: "Tests and helpers" },
  { key: "tuning", label: "Tuning" },
  { key: "history", label: "Run history" },
] as const

type DiagnosticsTab = (typeof DIAGNOSTICS_TABS)[number]["key"]

export default function Diagnostics() {
  const toast = useToast()
  const location = useLocation()
  const navigate = useNavigate()
  const tabKey = location.pathname.replace(/^\/diagnostics\/?/, "").split("/")[0] ?? ""
  const tab = DIAGNOSTICS_TABS.find((t) => t.key === tabKey)?.key as DiagnosticsTab | undefined
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
  const [canServerInstanceId, setCanServerInstanceId] = useState<string | null>(null)
  const canServerInstanceIdRef = useRef<string | null>(null)
  const previousCanServerInstanceIdRef = useRef<string | null>(null)
  const canInventoryPollRef = useRef(newCanInventoryPollSequence())
  const [canDiscoveryRetryBusy, setCanDiscoveryRetryBusy] = useState(false)
  const currentCanServerEpoch = useCallback(
    () => canServerEpoch(canServerInstanceIdRef.current, robotStatusRecoveryEpochRef.current),
    []
  )
  const automaticCanDiscoveryAttemptsRef = useRef(new Set<string>())
  // A successful response without profile summaries (or a 404) identifies an
  // older host. It cannot distinguish attached roles, so only retain the
  // historical profile already reported by that host's robot status.
  const [legacyCanInventory, setLegacyCanInventory] = useState(false)
  // Jelly's wheels and lift: two idle links beside the Axol / Mantis one,
  // with their own CAN presence. null until fetched; jellySupported drops to
  // false on a host that predates the Jelly links (its status endpoint 404s).
  const [canDevices, setCanDevices] = useState<CanDeviceInventory | null>(null)
  const [jelly, setJelly] = useState<JellyStatus | null>(null)
  const [jellySupported, setJellySupported] = useState(true)
  const [jellyBusy, setJellyBusy] = useState<Partial<Record<JellyDevice, boolean>>>({})
  const autoJellyAttemptsRef = useRef(new Set<string>())
  const manualJellyOverrideRef = useRef(new Set<JellyDevice>())
  const installCanInventory = useCallback((inventory: CanInterfaceInventory) => {
    const instanceId = inventory.serverInstanceId ?? null
    canServerInstanceIdRef.current = instanceId
    setCanServerInstanceId(instanceId)
    setCanDiscovery(inventory.discovery ?? null)
    setCanDevices(inventory.devices ?? null)
    if (inventory.profiles) {
      setCanProfiles(inventory.profiles)
      setLegacyCanInventory(false)
    } else {
      setCanProfiles(null)
      setLegacyCanInventory(true)
    }
  }, [])

  const [metricKey, setMetricKey] = useState(() => localStorage.getItem("axolDiagMetric") || "pos")
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
      if (!l.trim() || l.startsWith("[serve]") || l.startsWith("@@live")) continue
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
      .find(
        (l) =>
          l.trim() &&
          !l.startsWith("[serve]") &&
          !l.startsWith("[prompt] ") &&
          !l.startsWith("@@live")
      ) ?? null

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
      voidCanInventoryPolls(canInventoryPollRef.current)
      autoJellyAttemptsRef.current.clear()
      manualJellyOverrideRef.current.clear()
      return
    }
    let active = true
    const poll = () => {
      fetchJellyStatus()
        .then((status) => {
          if (!active) return
          setJelly(status)
          setJellySupported(true)
        })
        .catch((error) => {
          if (!active) return
          if (String(error).includes("HTTP 404")) setJellySupported(false)
          setJelly(null)
        })
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
      // Drop only responses older than the last one applied (not every response
      // superseded by a newer *request*): the endpoint can take longer than the
      // 2 s cadence during CAN discovery, and inventory must still land.
      const canRequest = issueCanInventoryPoll(canInventoryPollRef.current)
      fetchCanInterfaces()
        .then((inventory) => {
          if (!active || !claimCanInventoryPollResponse(canInventoryPollRef.current, canRequest))
            return
          canInventoryKnownRef.current = true
          installCanInventory(inventory)
        })
        .catch((error) => {
          if (!active || !claimCanInventoryPollResponse(canInventoryPollRef.current, canRequest))
            return
          if (String(error).includes("HTTP 404")) {
            canInventoryKnownRef.current = true
            setCanProfiles(null)
            setCanDiscovery(null)
            setCanDevices(null)
            canServerInstanceIdRef.current = null
            setCanServerInstanceId(null)
            setLegacyCanInventory(true)
            return
          }
          canInventoryKnownRef.current = false
          setCanProfiles(null)
          setCanDiscovery(null)
          setCanDevices(null)
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
      toast.error(`${label} failed. Open its run in Run history to see the log.`)
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
  useEffect(() => {
    if (canServerInstanceId === null) return
    const previous = previousCanServerInstanceIdRef.current
    previousCanServerInstanceIdRef.current = canServerInstanceId
    if (previous === null || previous === canServerInstanceId) return
    resetAutoRobotRetry()
    manualRobotOverrideRef.current = false
    automaticCanDiscoveryAttemptsRef.current.clear()
    autoJellyAttemptsRef.current.clear()
    manualJellyOverrideRef.current.clear()
  }, [canServerInstanceId, resetAutoRobotRetry])
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
          throw new Error(
            status.error ??
              `Could not connect the ${profile === "mantis" ? "Mantis" : "Axol"} CAN link.`
          )
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
          throw new Error(
            status.error ??
              `Could not connect the ${profile === "mantis" ? "Mantis" : "Axol"} CAN link.`
          )
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

  // Releasing the Axol / Mantis link by hand also stops this page bringing it
  // back up on its own until the host changes.
  const disconnectRobot = useCallback(async () => {
    resetAutoRobotRetry()
    manualRobotOverrideRef.current = true
    setRobotBusy(true)
    try {
      const status = await robotDisconnect()
      if (!autoRobotMountedRef.current) return
      setRobot(status)
      if (status.state !== "disconnected") {
        throw new Error(status.error ?? "Could not disconnect the CAN link.")
      }
    } catch (e) {
      if (autoRobotMountedRef.current) toast.error(String(e))
    } finally {
      if (autoRobotMountedRef.current) setRobotBusy(false)
    }
  }, [resetAutoRobotRetry, toast])

  // -- Jelly wheels / lift links --
  const connectJelly = useCallback(
    async (device: JellyDevice, automatic = false): Promise<boolean> => {
      if (!automatic) manualJellyOverrideRef.current.add(device)
      setJellyBusy((prev) => ({ ...prev, [device]: true }))
      try {
        const status = await jellyConnect(device, automatic)
        if (!autoRobotMountedRef.current) return false
        setJelly(status)
        if (!status[device].connected) {
          throw new Error(
            status[device].error ?? `Could not connect the ${JELLY_DEVICE_LABELS[device]} link.`
          )
        }
        return true
      } catch (e) {
        if (!autoRobotMountedRef.current) return false
        if (!automatic) toast.error(String(e))
        return false
      } finally {
        if (autoRobotMountedRef.current) setJellyBusy((prev) => ({ ...prev, [device]: false }))
      }
    },
    [toast]
  )
  const disconnectJelly = useCallback(
    async (device: JellyDevice) => {
      manualJellyOverrideRef.current.add(device)
      setJellyBusy((prev) => ({ ...prev, [device]: true }))
      try {
        const status = await jellyDisconnect(device)
        if (!autoRobotMountedRef.current) return
        setJelly(status)
        if (status[device].state !== "disconnected") {
          throw new Error(
            status[device].error ?? `Could not disconnect the ${JELLY_DEVICE_LABELS[device]} link.`
          )
        }
      } catch (e) {
        if (autoRobotMountedRef.current) toast.error(String(e))
      } finally {
        if (autoRobotMountedRef.current) setJellyBusy((prev) => ({ ...prev, [device]: false }))
      }
    },
    [toast]
  )

  const settledCanDiscovery = useSettledDiscovery(canDiscovery)
  const canDiscoveryNeedsRetry = discoveryNeedsRecheck(settledCanDiscovery)
  const hardwareNotice = discoveryNotice(settledCanDiscovery)
  const canDiscoveryChecking = canDiscoveryRetryBusy || canDiscovery?.status === "running"
  const retryCanIdentification = useCallback(async () => {
    if (canDiscoveryRetryBusy) return
    setCanDiscoveryRetryBusy(true)
    try {
      const inventory = await discoverCanHardware(true)
      if (!autoRobotMountedRef.current) return
      canInventoryKnownRef.current = true
      installCanInventory(inventory)
      if (inventory.discovery?.status === "configured" || inventory.discovery?.status === "ready") {
        toast.success("Hardware found")
      }
    } catch {
      // Busy or rate-limited: the notice stays until the hardware answers.
    } finally {
      if (autoRobotMountedRef.current) setCanDiscoveryRetryBusy(false)
    }
  }, [canDiscoveryRetryBusy, installCanInventory, toast])

  // Identify a fresh anonymous dual-channel hub before Diagnostics decides
  // whether exactly one hardware profile is safe to connect. The backend
  // coalesces requests across tabs; this latch avoids duplicates from this
  // route's inventory poll.
  useEffect(() => {
    if (!serverOk) {
      automaticCanDiscoveryAttemptsRef.current.clear()
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

    const serverEpoch = currentCanServerEpoch()
    const attemptKey = `${serverEpoch}:${signature}`
    if (automaticCanDiscoveryAttemptsRef.current.has(attemptKey)) return
    automaticCanDiscoveryAttemptsRef.current.add(attemptKey)

    void discoverCanHardware()
      .then((inventory) => {
        if (!autoRobotMountedRef.current || serverEpoch !== currentCanServerEpoch()) return
        installCanInventory(inventory)
      })
      .catch((error) => {
        if (!autoRobotMountedRef.current || serverEpoch !== currentCanServerEpoch()) return
        if (canDiscoveryRequestCanRetry(error)) {
          automaticCanDiscoveryAttemptsRef.current.delete(attemptKey)
        }
      })
  }, [
    activeRun,
    autoRobotPollStateKnown,
    canDiscovery,
    canServerInstanceId,
    currentCanServerEpoch,
    hardwareSessionBusy,
    installCanInventory,
    launchBusy,
    robot?.state,
    robotBusy,
    serverOk,
    toast,
  ])

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
    const serverEpoch = currentCanServerEpoch()
    const signature = `${profileSignature}:host-${serverEpoch}`
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
      if (
        connected ||
        !autoRobotMountedRef.current ||
        serverEpoch !== currentCanServerEpoch() ||
        manualRobotOverrideRef.current
      )
        return
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
          serverEpoch === currentCanServerEpoch() &&
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
    canServerInstanceId,
    canProfiles,
    connectRobot,
    currentCanServerEpoch,
    hardwareSessionBusy,
    launchBusy,
    legacyCanInventory,
    robot,
    robotBusy,
    resetAutoRobotRetry,
    serverOk,
    sessionInventoryReady,
  ])

  // Bring up each detected Jelly device once per (device, interface, server
  // epoch), like the control panel does: only from an idle page, only after
  // CAN discovery has settled, and never for a device the operator
  // disconnected by hand (the server refuses those, and this tab stops
  // asking). Absence is not latched — a replug gets a fresh attempt.
  useEffect(() => {
    if (!serverOk || !jelly || !jellySupported || !canDevices || !sessionInventoryReady) return
    if (
      canDiscoveryBlocksAutoConnect(canDiscovery) ||
      activeRun ||
      hardwareSessionBusy ||
      launchBusy ||
      !autoRobotPollStateKnown()
    )
      return
    const serverEpoch = currentCanServerEpoch()
    for (const device of JELLY_DEVICES) {
      const presence = canDevices[device]
      const status = jelly[device]
      if (!presence?.present || presence.automaticConnectSuppressed) continue
      if (status.state !== "disconnected" && status.state !== "error") continue
      if (jellyBusy[device] || manualJellyOverrideRef.current.has(device)) continue
      const signature = `${device}:${presence.channel}:host-${serverEpoch}`
      if (autoJellyAttemptsRef.current.has(signature)) continue
      autoJellyAttemptsRef.current.add(signature)
      void connectJelly(device, true)
    }
  }, [
    activeRun,
    autoRobotPollStateKnown,
    canDevices,
    canDiscovery,
    canServerInstanceId,
    connectJelly,
    currentCanServerEpoch,
    hardwareSessionBusy,
    jelly,
    jellyBusy,
    jellySupported,
    launchBusy,
    serverOk,
    sessionInventoryReady,
  ])

  // The joints this robot actually has — the gripperless SKU drops GRIPPER
  // from the motor tiles, chart series, and `--joints` pickers.
  const joints = useMemo(
    () => (robot?.profile === "mantis" ? (["GRIPPER"] as const) : jointsFor(robot?.hasGripper)),
    [robot]
  )

  // Both arms chart at once (a left/right pair per metric), so build one
  // series list per side from the shared joint filter.
  const seriesBySide: Record<ArmSide, ChartSeries[]> = useMemo(() => {
    const forSide = (side: ArmSide) =>
      joints
        .filter((j) => !hiddenJoints.has(j))
        .map((joint) => ({
          key: motorKey(side, joint),
          label: jointLabel(joint),
          color: JOINT_COLORS[joint],
        }))
    return { left: forSide("left"), right: forSide("right") }
  }, [hiddenJoints, joints])

  const linkState = robot?.state ?? stream.state
  const stateBadge = STATE_BADGE[linkState] ?? STATE_BADGE.disconnected
  // "busy" no longer silences the chart: the server keeps decoding the
  // running task's own CAN traffic through a passive bus observer, so frames
  // keep arriving whenever the task is actually commanding the motors. The
  // note explains both the passive source and any lull between commands.
  const quietReason =
    linkState === "busy"
      ? "Bus in use, showing observed CAN traffic"
      : linkState !== "connected"
        ? "Robot disconnected"
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
  // The Diagnostics category splits into three dashboard sections: Tests and
  // Helpers here, and the tuning workbench's tabs for "tuning". A host that
  // predates the section field sends none — everything then lands under
  // Tests, which matches the old single-section layout.
  const testCommands = useMemo(
    () => diagCommands.filter((c) => (c.section ?? "test") === "test"),
    [diagCommands]
  )
  // The safety board helper only makes sense on a Jelly that has one.
  const [safetyDetected, setSafetyDetected] = useState(false)
  useEffect(() => {
    if (!serverOk) return
    let active = true
    const check = () =>
      fetchSafetyStatus()
        .then((s) => {
          if (active) setSafetyDetected(s.detected)
        })
        .catch(() => {})
    void check()
    const timer = setInterval(check, 5000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [serverOk])
  const helperCommands = useMemo(
    () =>
      diagCommands.filter(
        (c) => c.section === "helper" && (c.id !== "safety.update" || safetyDetected)
      ),
    [diagCommands, safetyDetected]
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
            "Set one motor's zero to its current position and save it to flash. " +
            "Damiao motors need a power cycle afterwards.",
          hideKeys: ["guided", "joints"],
        },
        {
          key: "guided",
          label: "Guided",
          description:
            "Zero each selected joint of one arm against an end stop. Wrist 2 and " +
            "wrist 3 can use either end stop. Each step shows instructions and waits " +
            "for Continue. On Jelly with a safety board, after the last joint it asks you " +
            "to press Power-cycle Axol so the new wrist zeros take effect.",
          presetArgs: { guided: true },
          hideKeys: ["id", "type"],
        },
      ],
    },
    {
      key: "dump-config",
      command: "motor.dump-config",
      label: "Read all parameters",
      icon: ClipboardList,
      description:
        "Read every configuration parameter from one motor, or from all motors " +
        "if CAN ID is blank. Works for MyActuator and Damiao motors and changes " +
        "nothing. The log is saved in Run history, so run this before you change " +
        "any parameter.",
    },
    {
      key: "set-config",
      command: "motor.set-config",
      label: "Edit parameter",
      icon: SlidersHorizontal,
      modes: [
        {
          key: "read",
          label: "Read",
          description:
            "Read one configuration parameter without changing it. Works for " +
            "MyActuator and Damiao motors.",
          hideKeys: ["value", "force_protected", "yes"],
        },
        {
          key: "write",
          label: "Write",
          description:
            "Write one configuration parameter and save it to the motor's flash. " +
            "Damiao's CAN timeout is in milliseconds. Protected parameters (factory " +
            "calibration, CAN IDs, baud rate) need Force protected turned on. A wrong " +
            "value can leave the motor unable to commutate or unreachable on the bus.",
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
        "Replace one motor's firmware with a .bin file on the robot host. Keep the " +
        "arm powered and idle, with nothing else using the bus. If a flash is " +
        "interrupted, the motor stays in its bootloader until you run this again.",
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

  const metric = METRICS.find((m) => m.key === metricKey) ?? METRICS[0]
  const chartFrames = metric.slow ? stream.slowFrames : stream.frames

  // Follow mode anchors the window to the newest sample; the page re-renders
  // on every stream tick, so the live edge advances with the data (and holds
  // still while the stream is paused). Zoom/pan pins a fixed range.
  const motorLastT = chartFrames.length > 0 ? chartFrames[chartFrames.length - 1].t : 0
  const timingLastT =
    stream.timingFrames.length > 0 ? stream.timingFrames[stream.timingFrames.length - 1].t : 0
  const lastT = Math.max(motorLastT, timingLastT) || windowSec
  const view: ChartView = pinnedView ?? { t0: lastT - windowSec, t1: lastT }

  if (tab === undefined) return <Navigate to="/diagnostics" replace />

  const storedHost = localStorage.getItem("axolServerHost") ?? ""
  const tabLabel = DIAGNOSTICS_TABS.find((t) => t.key === tab)?.label ?? ""

  return (
    <>
      <AppShell
        title={tab === "" ? "Diagnostics" : `Diagnostics: ${tabLabel}`}
        wide
        actions={<Badge variant={stateBadge.variant}>{stateBadge.text}</Badge>}
        safety={<SafetyControl enabled={serverOk} motorFaults={motorFaultLabels(robot, jelly)} />}
        host={{
          dot: serverOk ? "ok" : "err",
          label: serverOk ? storedHost || "Connected" : "Can't reach the robot",
          onClick: () => navigate("/control/devices"),
        }}
        subNav={{
          under: "/diagnostics",
          items: DIAGNOSTICS_TABS.map((t) => ({
            to: t.key ? `/diagnostics/${t.key}` : "/diagnostics",
            label: t.label,
          })),
        }}
        notices={
          <>
            {serverOk && canDiscoveryNeedsRetry && hardwareNotice && (
              <div className="flex flex-wrap items-center gap-3 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3">
                <p
                  className="min-w-0 flex-1 text-sm text-amber-100/80"
                  title={settledCanDiscovery?.message ?? undefined}
                >
                  {hardwareNotice.text}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void retryCanIdentification()}
                  disabled={
                    canDiscoveryChecking ||
                    hardwareSessionBusy ||
                    launchBusy ||
                    robotBusy ||
                    activeRun !== null
                  }
                >
                  {canDiscoveryChecking ? <Loader2 className="animate-spin" /> : <RefreshCw />}
                  Check again
                </Button>
              </div>
            )}

            {/* Robot link gate */}
            {robot && robot.state === "disconnected" && (
              <div className="flex flex-wrap items-center gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                <p className="text-sm text-white/60">
                  {robot.profile === "mantis" ? "Mantis" : "Axol"} is disconnected. Connect it to
                  see motor telemetry.
                </p>
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
                    {robotBusy ? <Loader2 className="animate-spin" /> : null} Connect{" "}
                    {robot.profile === "mantis" ? "Mantis" : "Axol"}
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
          </>
        }
      >
        {tab === "" && (
          <>
            {/* Every CAN-attached device: Axol / Mantis (the telemetry link's two
            profiles) plus Jelly's wheels and lift on their own adapters. */}
            <PageSection
              title="Hardware"
              description="Only one of Axol and Mantis can be connected at a time. Jelly's wheels and lift connect separately."
            >
              <HardwareOverview
                online={serverOk}
                robot={robot}
                robotBusy={robotBusy}
                canProfiles={canProfiles}
                hardwareBusy={activeRun != null || hardwareSessionBusy || launchBusy}
                onRobotConnect={(profile) => void connectRobot(profile)}
                onRobotDisconnect={() => void disconnectRobot()}
                jelly={jelly}
                jellySupported={jellySupported}
                jellyBusy={jellyBusy}
                canDevices={canDevices}
                onJellyConnect={(device) => void connectJelly(device)}
                onJellyDisconnect={(device) => void disconnectJelly(device)}
              />
            </PageSection>

            {/* Motor status */}
            <section className="flex flex-col gap-4">
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="font-heading text-base font-semibold">
                  {robot?.profile === "mantis" ? "Mantis motors" : "Axol motors"}
                </h2>
                {robot && (
                  <span className="text-xs text-white/40">
                    {robot.reachableCount}/{robot.motorCount} reachable
                  </span>
                )}
                <div className="flex basis-full flex-wrap items-center gap-2">
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
                            ? "Choose the CAN adapter for each arm and bring the interfaces up. Every run uses this mapping."
                            : cmd.description
                        }
                        disabled={
                          !serverOk ||
                          launchBusy ||
                          busyElsewhere ||
                          (activeRun != null && !running)
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

            <ControlHealth
              frames={stream.timingFrames}
              version={stream.version}
              nowT={lastT}
              view={view}
              onViewChange={setPinnedView}
            />

            {/* Live charts */}
            <section className="flex flex-col gap-3">
              {/* One filter row scoping every chart below it. */}
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="mr-2 font-heading text-base font-semibold">Live telemetry</h2>
                <div className="flex overflow-hidden rounded-md border border-white/10">
                  {METRICS.map((m) => (
                    <button
                      key={m.key}
                      type="button"
                      onClick={() => {
                        setMetricKey(m.key)
                        localStorage.setItem("axolDiagMetric", m.key)
                      }}
                      className={cn(
                        "px-2.5 py-1 text-xs transition-colors",
                        metricKey === m.key
                          ? "bg-[#eff483]/15 text-[#eff483]"
                          : "text-white/50 hover:bg-white/[0.05]"
                      )}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
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
                Scroll to zoom and drag to pan. Zooming pauses the live view until you press Go
                live.
              </p>
              {/* One metric at a time — the toggle row above picks it (temperature
              reads the 1 Hz sweep buffer) — charted as a left/right pair
              sharing the same time view so both arms read at once; the header
              button on each chart takes it truly full screen. */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {(["left", "right"] as ArmSide[]).map((side) => (
                  <TelemetryChart
                    key={side}
                    title={`${metric.title}, ${side} ${robot?.profile === "mantis" ? "gripper" : "arm"}`}
                    unit={metric.unit}
                    series={seriesBySide[side]}
                    frames={chartFrames}
                    version={stream.version}
                    metric={metric.metric}
                    scale={metric.scale}
                    view={view}
                    onViewChange={setPinnedView}
                    quietReason={quietReason}
                    height={340}
                    gapBreakS={metric.slow ? 5 : undefined}
                  />
                ))}
              </div>
            </section>
          </>
        )}

        {/* Each Diagnostics command has exactly one home: Tests or Helpers
            below, or a tab of the tuning workbench for the "tuning" section
            (a tuning command without a tab, like the retired offline filter
            test, stays CLI-only). */}

        {/* Tests: pass/fail checks (ROM soak, lift cycle, camera cable). */}
        {tab === "tests" && testCommands.length === 0 && helperCommands.length === 0 && (
          <div className="flex min-h-48 items-center justify-center rounded-xl border border-dashed border-white/10 p-6 text-center text-sm text-white/45">
            {serverOk
              ? "This Host Computer has no tests or helpers. Update Axol on it to get them."
              : "Connect to the Host Computer to see its tests and helpers."}
          </div>
        )}

        {tab === "tests" && testCommands.length > 0 && (
          <section className="flex flex-col gap-3">
            <h2 className="font-heading text-base font-semibold">Tests</h2>
            <DiagnosticActions
              commands={testCommands}
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
        )}

        {/* Helpers: utility moves (ROM cleanup, lift homing / height). */}
        {tab === "tests" && helperCommands.length > 0 && (
          <section className="flex flex-col gap-3">
            <h2 className="font-heading text-base font-semibold">Helpers</h2>
            <DiagnosticActions
              commands={helperCommands}
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
        )}

        {/* Tuning workbench: inline sine/step/motion/analysis launches with
            the resulting graphs, scorecards, and A/B compare in place. */}
        {/* Kept mounted on every tab so a run's charts and A/B picks survive a
            look at another page. */}
        <div className={cn(tab !== "tuning" && "hidden")}>
          <TuningWorkbench
            enabled={serverOk}
            commands={commands}
            activeCommand={activeRun?.command ?? null}
            busy={launchBusy}
            disabled={busyElsewhere}
            liveLines={activeLines}
            onLaunch={launch}
            onStop={stopActive}
          />
        </div>

        {/* Run history */}
        {tab === "history" && (
          <RunHistory
            runs={runs}
            loading={runsLoading}
            onRefresh={refreshRuns}
            onClear={clearRuns}
          />
        )}
      </AppShell>

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
    </>
  )
}
