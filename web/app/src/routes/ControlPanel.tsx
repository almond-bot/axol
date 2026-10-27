import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom"
import { ArrowRight, Camera, CircleAlert, Loader2, Plug, RefreshCw, ScrollText } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  HARDWARE_PROFILE_ARG,
  HARDWARE_PROFILE_SETTING,
  parseAxolModel,
  parseGripper,
  JELLY_DEVICES,
  JELLY_DEVICE_LABELS,
  OPERATIONS,
  fetchDatasets,
  cameraCount,
  cameraSerials,
  canDiscoveryRequestCanRetry,
  clearOperationLockout,
  detectCameras,
  discoverCanHardware,
  fetchCanInterfaces,
  fetchCommands,
  fetchInfo,
  fetchJellyStatus,
  fetchOpStatus,
  fetchRobotStatus,
  fetchSessions,
  fetchSettings,
  fetchUpdateStatus,
  fetchUsbStatus,
  isSimRun,
  jellyConnect,
  jellyDisconnect,
  loadLocalHardwareProfile,
  loadOpSettings,
  missingCameraSerials,
  operationsFromCommands,
  parseHardwareProfile,
  perRunFields,
  robotConnect,
  robotDisconnect,
  saveLocalHardwareProfile,
  saveOpSettings,
  saveSettings,
  sendEpisodeCommand,
  setServerBase,
  startOperation,
  startUpdate,
  stopOperation,
  usbConnect,
  useSessionLogs,
  type CameraDevice,
  type CameraSpec,
  type CanDeviceInventory,
  type CanDiscoveryState,
  type AxolMount,
  type CanInterfaceInventory,
  type CanProfileInventory,
  type CommandSpec,
  type DatasetInfo,
  type FormValue,
  type HardwareProfile,
  type JellyDevice,
  type JellyStatus,
  type OperationId,
  type PolicyState,
  type RobotStatus,
  type ServerInfo,
  type SessionInfo,
  type SettingsPatch,
  type SettingsSnapshot,
  type UpdatePhase,
  type UpdateStatus,
  type UsbStatus,
} from "@/lib/supervisor"
import {
  autoConnectPollStateKnown,
  autoConnectRetryDelay,
  autoConnectSignature,
  canDiscoveryAttemptSignature,
  canDiscoveryBlocksAutoConnect,
  canServerEpoch,
  chooseAutoConnectTarget,
  chooseDetectedHardwareProfile,
  claimCanInventoryPollResponse,
  hardwarePresenceSignature,
  issueCanInventoryPoll,
  newCanInventoryPollSequence,
  nextAutoConnectAttempt,
  shouldStartCanDiscovery,
  voidCanInventoryPolls,
} from "@/lib/can-auto-connect"
import { UpdateBanner } from "@/components/update-banner"
import { VersionMismatchBanner } from "@/components/version-mismatch-banner"
import { versionMismatch } from "@/lib/version"
import { ConnectionsBar } from "@/components/connections-bar"
import { OperationPanel, type BlockerFix } from "@/components/operation-panel"
import { OperationLauncher, type LauncherItem } from "@/components/operation-launcher"
import {
  AttentionList,
  ConnectHero,
  DevicesRow,
  HardwareNoticeCard,
  type DeviceSummary,
  QuickStart,
  RecentDatasets,
  RunningCard,
  type AttentionItem,
} from "@/components/control-home"
import {
  catalogEntry,
  catalogMeta,
  launcherEntries,
  operationStatus,
  quickStartOperations,
  recentOperations,
  rememberOperation,
} from "@/lib/operation-catalog"
import { LogConsole } from "@/components/log-console"
import { DatasetPreview } from "@/components/dataset-preview"
import { DataCollectionOffer } from "@/components/data-collection-offer"
import { RequestDataPage } from "@/components/request-data"
import { requestSource } from "@/lib/data-collection"
import {
  deviceErrorText,
  discoveryNeedsRecheck,
  discoveryNotice,
  nextRecheckDelay,
  useSettledDiscovery,
} from "@/lib/hardware-status"
import { showsDatasetPreview } from "@/lib/dataset-operations"
import { useStartupView } from "@/lib/operation-progress"
import { SetupDialog, type ConnState } from "@/components/setup-dialog"
import { SettingsSection } from "@/components/settings/settings-section"
import { MountChooser, MountRequired } from "@/components/settings/mount-required"
import {
  SETTINGS_SCOPES,
  camerasSettingsTab,
  defaultSettingsTab,
  settingsScopeForTab,
  type SettingsScope,
  type SettingsTab,
} from "@/lib/settings-scope"
import { loadHostHistory, recordHost } from "@/lib/host-history"
import { AppShell, type HostSummary, type NavBadges } from "@/components/app-shell"
import { jellyDeviceView, type StatusDot } from "@/lib/jelly-view"
import { useToast } from "@/components/ui/toast"
import { Button } from "@/components/ui/button"
import type { SimRobot } from "@/components/sim-robot-picker"

type OpSettings = Record<OperationId, Record<string, FormValue>>

/** The pages of the control panel, one per sidebar item. */
type ControlView =
  | "overview"
  | "operations"
  | "datasets"
  | "request-data"
  | "devices"
  | "settings"
  | "logs"

const CONTROL_VIEWS: Record<string, ControlView> = {
  "": "overview",
  operations: "operations",
  datasets: "datasets",
  "request-data": "request-data",
  devices: "devices",
  settings: "settings",
  logs: "logs",
}

/** `/control/settings/cameras` → `{ view: "settings", rest: "cameras" }`. */
function parseControlPath(pathname: string): { view: ControlView | null; rest: string | null } {
  const [section = "", rest] = pathname.replace(/^\/control\/?/, "").split("/")
  return { view: CONTROL_VIEWS[section] ?? null, rest: rest ? decodeURIComponent(rest) : null }
}

const DEFAULT_CAMERAS: CameraSpec = {
  serials: { overhead: "", left_arm: "", right_arm: "" },
  mantis_serials: { overhead: "", left_arm: "", right_arm: "" },
  stream_resolution: "SVGA",
  record_resolution: "SVGA",
  stream: {},
  record: {},
}

function loadCameras(): CameraSpec {
  try {
    const raw = localStorage.getItem("axolCameraSpec")
    if (raw) {
      const parsed = JSON.parse(raw)
      const serials = { ...DEFAULT_CAMERAS.serials, ...(parsed.serials ?? {}) }
      return {
        ...DEFAULT_CAMERAS,
        ...parsed,
        serials,
        mantis_serials: cameraSerials({ serials, mantis_serials: parsed.mantis_serials }, true),
        // Migrate the legacy single `resolution` to the streaming resolution.
        stream_resolution:
          parsed.stream_resolution ?? parsed.resolution ?? DEFAULT_CAMERAS.stream_resolution,
        record_resolution: parsed.record_resolution ?? DEFAULT_CAMERAS.record_resolution,
        // Migrate the earlier per-eye maps to the per-branch participation maps.
        stream: { ...(parsed.stream ?? parsed.stream_eyes ?? {}) },
        record: { ...(parsed.record ?? parsed.record_eyes ?? {}) },
      }
    }
  } catch {
    // ignore malformed storage
  }
  return DEFAULT_CAMERAS
}

function persistLocalCameras(spec: CameraSpec) {
  try {
    localStorage.setItem("axolCameraSpec", JSON.stringify(spec))
  } catch {
    // ignore storage failures
  }
}

const SIM_ROBOT_KEY = "axolSimRobot"

/** The sim robot this browser last picked, if it is still a valid one. */
function loadSimRobot(): SimRobot | null {
  try {
    const raw = JSON.parse(localStorage.getItem(SIM_ROBOT_KEY) ?? "null")
    const model = parseAxolModel(raw?.model)
    return model ? { model, gripper: parseGripper(raw?.gripper) } : null
  } catch {
    return null
  }
}

export default function ControlPanel() {
  const toast = useToast()
  const location = useLocation()
  const navigate = useNavigate()
  const { view, rest: pathRest } = parseControlPath(location.pathname)
  const [commands, setCommands] = useState<CommandSpec[]>([])

  // Back from Stripe's checkout page for a Teleop Data Collection request: confirm
  // the outcome once, then drop the marker from the address.
  const handledCheckoutRef = useRef<string | null>(null)
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const outcome = params.get("data_collection")
    if (!outcome || handledCheckoutRef.current === location.search) return
    handledCheckoutRef.current = location.search
    if (outcome === "paid") {
      toast.success(
        "Payment received. Almond will contact you by email to plan your Teleop Data Collection.",
        12000
      )
    } else if (outcome === "cancelled") {
      toast.info("Teleop Data Collection request cancelled. You weren't charged.")
    }
    params.delete("data_collection")
    const search = params.toString()
    navigate({ pathname: location.pathname, search: search ? `?${search}` : "" }, { replace: true })
  }, [location.pathname, location.search, navigate, toast])
  const [conn, setConn] = useState<{ state: ConnState; message?: string }>({ state: "loading" })
  const [serverHost, setServerHost] = useState<string>(
    () => localStorage.getItem("axolServerHost") ?? ""
  )
  const [hostHistory, setHostHistory] = useState<string[]>(loadHostHistory)
  const [hostInfo, setHostInfo] = useState<ServerInfo | null>(null)
  const [viewerPort, setViewerPort] = useState(8002)
  const [update, setUpdate] = useState<UpdateStatus | null>(null)
  // Bridges the gap between clicking Update and the server's status first
  // reporting the in-flight update, so the banner switches to the spinner
  // immediately; the watcher clears it once the real server state is known.
  const [startingUpdate, setStartingUpdate] = useState(false)
  // Set when the watcher gives up (deadline) so the banner drops the spinner and
  // offers a retry even if the server is still/again reporting "updating".
  // Cleared when a new update is kicked off (and on disconnect).
  const [updateAbandoned, setUpdateAbandoned] = useState(false)
  // Whether the server is applying an update. Derived from its authoritative
  // status (not a local click) so EVERY connected computer shows the in-flight
  // update — spinner + phase — rather than a stale, clickable Update button.
  const updating = !updateAbandoned && (startingUpdate || update?.state === "updating")
  // Current step shown in the banner while updating (so it isn't an opaque
  // spinner). Sourced from the server's reported phase, except "restarting"
  // which we infer locally once the server stops responding (it exited).
  const [updatePhase, setUpdatePhase] = useState<UpdatePhase | null>(null)

  const [robot, setRobot] = useState<RobotStatus | null>(null)
  const [simRobotChoice, setSimRobotChoice] = useState<SimRobot | null>(loadSimRobot)
  const [robotBusy, setRobotBusy] = useState(false)
  // A backend restart can leave the tab's old auto-connect latch intact. A
  // failed status poll followed by recovery starts a new connection epoch,
  // without treating a remote browser's ordinary disconnect as a restart.
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
  const [canDevices, setCanDevices] = useState<CanDeviceInventory | null>(null)
  // Jelly's wheels and lift: two independent idle links next to the Axol /
  // Mantis one. null until fetched; jellySupported drops to false on a host
  // too old to expose them (404), so their tiles say so instead of failing.
  const [jelly, setJelly] = useState<JellyStatus | null>(null)
  const [jellySupported, setJellySupported] = useState(true)
  const [jellyBusy, setJellyBusy] = useState<Partial<Record<JellyDevice, boolean>>>({})
  // One automatic connect per (device, interface, server epoch); a manual
  // Connect/Disconnect on a device takes it out of automatic hands for this tab.
  const autoJellyAttemptsRef = useRef(new Set<string>())
  const manualJellyOverrideRef = useRef(new Set<JellyDevice>())
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
  // The backend owns the real single-flight. This per-tab latch prevents the
  // 2s inventory poll from sending duplicate requests for one hardware epoch.
  const automaticCanDiscoveryAttemptsRef = useRef(new Set<string>())
  // A successful response without profile summaries identifies an older host;
  // retain its historical selected-profile auto-connect behavior.
  const [legacyCanInventory, setLegacyCanInventory] = useState(false)
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
  const [usb, setUsb] = useState<UsbStatus | null>(null)
  const [usbBusy, setUsbBusy] = useState(false)
  const [cameras, setCameras] = useState<CameraSpec>(() => loadCameras())
  // Shared settings stored on the serve host (~/.almond/settings.json); null
  // until fetched. settingsError marks a host too old for the settings API —
  // cameras then fall back to the legacy localStorage flow.
  const [settingsSnap, setSettingsSnap] = useState<SettingsSnapshot | null>(null)
  const [settingsError, setSettingsError] = useState<string | null>(null)
  // The system-wide device selection (Axol or Mantis) that every operation
  // runs on. The host's shared settings are authoritative so all operator
  // devices agree; this browser's copy covers hosts too old to store it.
  const [localHardwareProfile, setLocalHardwareProfile] = useState<HardwareProfile>(() =>
    loadLocalHardwareProfile()
  )
  const [hardwareProfileSaving, setHardwareProfileSaving] = useState(false)
  // Last ZED detection from the serve host (null until first detected), used to
  // verify the assigned serials are actually connected before a task starts.
  const [cameraDevices, setCameraDevices] = useState<CameraDevice[] | null>(null)
  const [cameraDetectError, setCameraDetectError] = useState<string | null>(null)
  const [cameraDetecting, setCameraDetecting] = useState(false)

  const [selectedOp, setSelectedOp] = useState<OperationId>(
    () => localStorage.getItem("axolOp") || OPERATIONS[0].id
  )
  // Only the ops edited this session; the rest are read from localStorage on
  // demand, since which ops exist isn't known until the host answers.
  const [settingsByOp, setSettingsByOp] = useState<OpSettings>({})

  const [session, setSession] = useState<SessionInfo | null>(null)
  const [sessionInventoryReady, setSessionInventoryReady] = useState(false)
  const [hardwareSessionBusy, setHardwareSessionBusy] = useState(false)
  // Generic setup/diagnostic owner (Pair, Identify, installers, etc.). The
  // server serializes it against operations; mirror that owner in the UI so
  // opposite actions disable instead of optimistically ending in HTTP 409.
  const [activeCommandSession, setActiveCommandSession] = useState<SessionInfo | null>(null)
  // run-policy episode phase/count, from the server so the episode controls are
  // correct on any computer (not just the tab that started the run).
  const [policy, setPolicy] = useState<PolicyState | null>(null)
  // The server could not confirm an operation disabled the motors and keeps
  // the robot reserved until the motors are proven torque-free (or it restarts).
  const [lockout, setLockout] = useState(false)
  const [lockoutBusy, setLockoutBusy] = useState(false)
  // The server's own view of whether the runner is busy (`op.running`). It
  // outlives the session's live status: the worker is still tearing down after
  // the session reads error/exited, and a lockout keeps it true after that.
  const [opBusy, setOpBusy] = useState(false)
  const [busy, setBusy] = useState(false)
  // Short label shown on the Start button while a start is being prepared (e.g.
  // "Checking cameras…"), so the wait isn't an opaque spinner — mirrors the
  // update banner's phase display.
  const [startPhase, setStartPhase] = useState<string | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)
  // The settings tab lives in the URL (/control/settings/<tab>) so it can be
  // linked and survives a reload; the last one seen reopens from the sidebar.
  const urlSettingsTab = view === "settings" ? pathRest : null
  const [lastSettingsTab, setLastSettingsTab] = useState<SettingsTab>("cameras")
  const settingsTab = urlSettingsTab ?? lastSettingsTab
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (urlSettingsTab) setLastSettingsTab(urlSettingsTab)
  }, [urlSettingsTab])
  // Staged, unsaved settings edits: flagged on the sidebar so they aren't lost
  // track of on another page (the draft itself survives page changes).
  const [settingsDirty, setSettingsDirty] = useState(false)
  // Every API helper targets the module-level server base. A slow response
  // from the previous host must therefore never update this host's UI (or
  // trigger the legacy camera-settings migration against the new host).
  const connectionGenerationRef = useRef(0)
  const [connectionGeneration, setConnectionGeneration] = useState(0)

  const { lines, status } = useSessionLogs(session?.id ?? null)

  // Enumerate the ZED cameras on the serve host so the Cameras badge can verify
  // the assigned serials are actually connected (best-effort: failures leave the
  // last known state and surface as a "can't detect" warning).
  const refreshCameras = useCallback(async (generation = connectionGenerationRef.current) => {
    if (generation !== connectionGenerationRef.current) return
    setCameraDetecting(true)
    try {
      const result = await detectCameras()
      if (generation !== connectionGenerationRef.current) return
      setCameraDevices(result.devices)
      setCameraDetectError(result.error)
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      setCameraDevices(null)
      setCameraDetectError(String(e).replace(/^Error:\s*/, ""))
    } finally {
      if (generation === connectionGenerationRef.current) setCameraDetecting(false)
    }
  }, [])

  // Pull the shared settings from the serve host. A host whose stored camera
  // spec is empty gets this browser's legacy localStorage spec migrated up
  // once, so nobody has to re-enter serials after updating.
  const loadSettings = useCallback(async (generation = connectionGenerationRef.current) => {
    if (generation !== connectionGenerationRef.current) return
    try {
      const snap = await fetchSettings()
      if (generation !== connectionGenerationRef.current) return
      setSettingsError(null)
      if (snap.cameras) {
        setCameras(snap.cameras)
        persistLocalCameras(snap.cameras)
        setSettingsSnap(snap)
      } else {
        const local = loadCameras()
        if (cameraCount(local) > 0) {
          try {
            if (generation !== connectionGenerationRef.current) return
            await saveSettings({ cameras: local, camerasSet: true })
            if (generation !== connectionGenerationRef.current) return
            setSettingsSnap({ ...snap, cameras: local })
          } catch {
            if (generation !== connectionGenerationRef.current) return
            setSettingsSnap(snap)
          }
        } else {
          setSettingsSnap(snap)
        }
      }
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      // Old serve host without /api/settings: keep the localStorage camera
      // flow; the settings dialog explains the needed update.
      setSettingsSnap(null)
      setSettingsError(String(e).replace(/^Error:\s*/, ""))
    }
  }, [])

  const loadServer = useCallback(
    async (host: string) => {
      const generation = ++connectionGenerationRef.current
      setConnectionGeneration(generation)
      setServerBase(host)
      setConn({ state: "loading" })
      // Hide all state owned by the previous host immediately. The generation
      // checks below also stop its in-flight responses from repopulating it.
      setCommands([])
      setHostInfo(null)
      setUpdate(null)
      setStartingUpdate(false)
      setUpdateAbandoned(false)
      setUpdatePhase(null)
      setRobot(null)
      setRobotBusy(false)
      setCanProfiles(null)
      setCanDevices(null)
      setJelly(null)
      setJellySupported(true)
      setJellyBusy({})
      autoJellyAttemptsRef.current.clear()
      manualJellyOverrideRef.current.clear()
      setCanDiscovery(null)
      canServerInstanceIdRef.current = null
      previousCanServerInstanceIdRef.current = null
      setCanServerInstanceId(null)
      setCanDiscoveryRetryBusy(false)
      voidCanInventoryPolls(canInventoryPollRef.current)
      setLegacyCanInventory(false)
      automaticCanDiscoveryAttemptsRef.current.clear()
      setUsb(null)
      setUsbBusy(false)
      setSession(null)
      setSessionInventoryReady(false)
      setHardwareSessionBusy(false)
      setActiveCommandSession(null)
      setPolicy(null)
      setBusy(false)
      setStartPhase(null)
      setCameraDevices(null)
      setCameraDetectError(null)
      setCameraDetecting(false)
      setSettingsSnap(null)
      setSettingsError(null)
      try {
        const cmds = await fetchCommands()
        if (generation !== connectionGenerationRef.current) return
        setCommands(cmds)
        setConn({ state: "ok" })
        setSetupOpen(false)
        setHostHistory(recordHost(host))
      } catch (e) {
        if (generation !== connectionGenerationRef.current) return
        setCommands([])
        setConn({ state: "err", message: String(e) })
        return
      }
      void refreshCameras(generation)
      void loadSettings(generation)
      fetchInfo()
        .then((info) => {
          if (generation !== connectionGenerationRef.current) return
          setViewerPort(info.viewerPort)
          setHostInfo(info)
        })
        .catch(() => {})
      // Force a synchronous remote check on connect/page load so the banner
      // reflects reality immediately; the steady-state poll below stays cheap.
      fetchUpdateStatus(true)
        .then((value) => {
          if (generation === connectionGenerationRef.current) setUpdate(value)
        })
        .catch(() => {})
      fetchRobotStatus()
        .then((value) => {
          if (generation === connectionGenerationRef.current) {
            robotStatusKnownRef.current = true
            setRobot(value)
          }
        })
        .catch(() => {
          if (generation === connectionGenerationRef.current) {
            robotStatusKnownRef.current = false
            setRobot(null)
          }
        })
      fetchOpStatus()
        .then((op) => {
          if (generation !== connectionGenerationRef.current) return
          if (op.running && op.session) {
            setSession(op.session)
            setSelectedOp(op.session.command as OperationId)
          }
          setPolicy(op.running ? op.policy : null)
          setLockout(op.lockout === true)
          setOpBusy(op.running)
        })
        .catch(() => {})
    },
    [refreshCameras, loadSettings]
  )

  useEffect(() => {
    loadServer(serverHost)
    // Only on mount — reconnects are explicit via the setup dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Poll the robot connection, configured CAN presence, and Quest-USB status
  // while online. CAN presence makes startup hardware-aware and also catches a
  // hub that enumerates shortly after the page loads.
  useEffect(() => {
    if (conn.state !== "ok") {
      robotStatusKnownRef.current = false
      canInventoryKnownRef.current = false
      voidCanInventoryPolls(canInventoryPollRef.current)
      return
    }
    // Guard against in-flight polls landing after a disconnect (which flips
    // conn.state and tears this effect down): a late response must not
    // repopulate a tile while the host tile shows disconnected.
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
            canServerInstanceIdRef.current = null
            setCanServerInstanceId(null)
            setLegacyCanInventory(true)
            return
          }
          canInventoryKnownRef.current = false
          setCanProfiles(null)
          setCanDiscovery(null)
          setLegacyCanInventory(false)
        })
      fetchUsbStatus()
        .then((u) => {
          if (active) setUsb(u)
        })
        .catch(() => {})
      fetchJellyStatus()
        .then((status) => {
          if (!active) return
          setJelly(status)
          setJellySupported(true)
        })
        .catch((error) => {
          if (!active) return
          // A host that predates the Jelly links answers 404: show the tiles
          // as unavailable rather than as a disconnected device.
          if (String(error).includes("HTTP 404")) setJellySupported(false)
          setJelly(null)
        })
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, installCanInventory])

  useEffect(() => {
    if (conn.state !== "ok") {
      sessionInventoryKnownRef.current = false
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setActiveCommandSession(null)
      return
    }
    let active = true
    const operationIds = new Set([
      ...OPERATIONS.map((operation) => operation.id),
      ...commands.filter((command) => command.isOperation).map((command) => command.id),
    ])
    const poll = () => {
      fetchSessions()
        .then((sessions) => {
          if (!active) return
          const liveSessions = sessions.filter(
            (candidate) =>
              candidate.status === "starting" ||
              candidate.status === "running" ||
              candidate.status === "stopping"
          )
          sessionInventoryKnownRef.current = true
          setSessionInventoryReady(true)
          setHardwareSessionBusy(liveSessions.length > 0)
          setActiveCommandSession(
            liveSessions.find((candidate) => !operationIds.has(candidate.command)) ?? null
          )
        })
        .catch(() => {
          if (!active) return
          sessionInventoryKnownRef.current = false
          setSessionInventoryReady(false)
        })
    }
    poll()
    const timer = window.setInterval(poll, 1500)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [conn.state, commands])

  // Poll the update indicator slowly while online (its server-side `ls-remote`
  // is debounced, so a tight interval would buy nothing). Paused while an
  // update is in flight — handleUpdate drives its own faster restart-watch poll.
  // Host identity (/api/info) rides along so a backend upgraded/restarted from
  // outside this tab (installer, another terminal) refreshes the commit the
  // version-mismatch check compares against.
  useEffect(() => {
    if (conn.state !== "ok" || updating) return
    let active = true
    const poll = () => {
      fetchUpdateStatus()
        .then((u) => {
          if (active) setUpdate(u)
        })
        .catch(() => {})
      fetchInfo()
        .then((info) => {
          if (!active) return
          setViewerPort(info.viewerPort)
          setHostInfo(info)
        })
        .catch(() => {})
    }
    poll()
    const t = setInterval(poll, 60_000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, updating])

  // A failed profile + mapping gets two delayed retries without being hammered
  // by the 2s presence poll. A mapping change creates a fresh signature.
  const autoRobotRef = useRef<string | null>(null)
  const autoRobotAttemptsRef = useRef(new Map<string, number>())
  const autoRobotRetryTimerRef = useRef<number | null>(null)
  const autoRobotMountedRef = useRef(true)
  const [autoRobotRetryRevision, setAutoRobotRetryRevision] = useState(0)
  // Manual connect/disconnect wins until the selected operation changes.
  const manualRobotOverrideRef = useRef(false)
  const previousDesiredProfileRef = useRef<HardwareProfile | null>(null)
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

  // A new backend process also clears its server-side manual-disconnect state.
  // Drop this tab's old retry/manual latches as soon as inventory proves that
  // the process changed, even when the restart fit between status polls.
  useEffect(() => {
    if (canServerInstanceId === null) return
    const previous = previousCanServerInstanceIdRef.current
    previousCanServerInstanceIdRef.current = canServerInstanceId
    if (previous === null || previous === canServerInstanceId) return
    resetAutoRobotRetry()
    manualRobotOverrideRef.current = false
    automaticCanDiscoveryAttemptsRef.current.clear()
    // The Jelly links restart disconnected with their manual-disconnect pause
    // forgotten too, so this tab's latches must not outlive the process.
    autoJellyAttemptsRef.current.clear()
    manualJellyOverrideRef.current.clear()
  }, [canServerInstanceId, resetAutoRobotRetry])

  // Auto-establish the Quest-over-USB tunnel as soon as an authorized headset
  // appears. The latch clears once the tunnel is up (so a later drop retries
  // once) or when the headset goes away, while preventing a per-poll retry loop
  // if `adb reverse` can't establish it.
  const autoUsbRef = useRef(false)
  const usbConnectClick = useCallback(
    async (generation = connectionGenerationRef.current) => {
      if (generation !== connectionGenerationRef.current) return
      setUsbBusy(true)
      try {
        const status = await usbConnect()
        if (generation !== connectionGenerationRef.current) return
        // Runs `adb reverse`; the first touch also pops the USB-debugging
        // authorization prompt on the headset.
        setUsb(status)
      } catch (e) {
        if (generation !== connectionGenerationRef.current) return
        toast.error(String(e))
      } finally {
        if (generation === connectionGenerationRef.current) setUsbBusy(false)
      }
    },
    [toast]
  )

  useEffect(() => {
    if (conn.state !== "ok") {
      // Clear the latch on disconnect so a reconnect gets a fresh auto-connect
      // attempt rather than inheriting a stuck "already tried" flag.
      autoUsbRef.current = false
      return
    }
    if (!usb || usb.state !== "device") {
      autoUsbRef.current = false
      return
    }
    if (usb.reverseActive) {
      autoUsbRef.current = false
      return
    }
    if (autoUsbRef.current || usbBusy) return
    autoUsbRef.current = true
    usbConnectClick()
  }, [conn.state, usb, usbBusy, usbConnectClick])

  function hostDisconnectClick() {
    // Purely client-side: drop this browser's view of the host without
    // touching server state. Other control panels on the network may be
    // driving the same host, so never stop a running op from here.
    connectionGenerationRef.current += 1
    setConnectionGeneration(connectionGenerationRef.current)
    setConn({ state: "idle" })
    setCommands([])
    setHostInfo(null)
    setRobot(null)
    setCanProfiles(null)
    setCanDevices(null)
    setJelly(null)
    setJellySupported(true)
    setJellyBusy({})
    autoJellyAttemptsRef.current.clear()
    manualJellyOverrideRef.current.clear()
    setCanDiscovery(null)
    canServerInstanceIdRef.current = null
    previousCanServerInstanceIdRef.current = null
    setCanServerInstanceId(null)
    setCanDiscoveryRetryBusy(false)
    voidCanInventoryPolls(canInventoryPollRef.current)
    setLegacyCanInventory(false)
    setUsb(null)
    setUsbBusy(false)
    setSession(null)
    setSessionInventoryReady(false)
    setHardwareSessionBusy(false)
    setActiveCommandSession(null)
    setPolicy(null)
    setBusy(false)
    setStartPhase(null)
    setCameraDevices(null)
    setCameraDetectError(null)
    setCameraDetecting(false)
    setSettingsSnap(null)
    setSettingsError(null)
    setUpdate(null)
    setStartingUpdate(false)
    setUpdateAbandoned(false)
    setUpdatePhase(null)
    setRobotBusy(false)
    robotStatusKnownRef.current = false
    canInventoryKnownRef.current = false
    sessionInventoryKnownRef.current = false
    automaticCanDiscoveryAttemptsRef.current.clear()
    resetAutoRobotRetry()
    manualRobotOverrideRef.current = false
    previousDesiredProfileRef.current = null
  }

  function updateServerHost(value: string) {
    setServerHost(value)
    if (value.trim()) localStorage.setItem("axolServerHost", value.trim())
    else localStorage.removeItem("axolServerHost")
  }

  function openSettings(tab: SettingsTab = "cameras") {
    navigate(`/control/settings/${encodeURIComponent(tab)}`)
  }

  function openOperation(op: OperationId) {
    selectOp(op)
    navigate(`/control/operations/${encodeURIComponent(op)}`)
  }

  // Back to the live run: its own panel, with its episode controls and Stop,
  // whatever operation was selected before.
  function openRunningOperation() {
    if (runningOp) openOperation(runningOp)
    else navigate("/control/operations")
  }

  // A connection tile opens its own settings scope on that scope's first tab.
  function openSettingsScope(scope: SettingsScope) {
    openSettings(defaultSettingsTab(scope))
  }

  // Whether the connected host's settings schema knows the device selection;
  // an older host keeps it per-browser rather than failing the save.
  const hostStoresHardwareProfile =
    settingsSnap != null &&
    !settingsError &&
    settingsSnap.schema.some((cat) => cat.settings.some((s) => s.key === HARDWARE_PROFILE_SETTING))
  // The host's value is the truth for every operator device (unset = Axol);
  // this browser's copy only applies to hosts that cannot store it.
  const hardwareProfile: HardwareProfile = hostStoresHardwareProfile
    ? (parseHardwareProfile(settingsSnap?.values[HARDWARE_PROFILE_SETTING]) ?? "axol")
    : localHardwareProfile
  // Axol and Mantis keep separate camera layouts: the prompt to assign cameras
  // is about the selected device's.
  const selectedHasCamera = cameraCount(cameras, hardwareProfile === "mantis") > 0

  // Resolves false only when the host rejected the save (the optimistic
  // selection was reverted), so the automatic selection can retry it.
  async function selectHardwareProfile(profile: HardwareProfile): Promise<boolean> {
    const generation = connectionGenerationRef.current
    setLocalHardwareProfile(profile)
    saveLocalHardwareProfile(profile)
    if (!hostStoresHardwareProfile || !settingsSnap) return true
    // Optimistic: the switch, the tiles, and the auto-connect target follow
    // immediately; a failed save restores the stored values.
    const priorValues = settingsSnap.values
    setSettingsSnap((prev) =>
      prev ? { ...prev, values: { ...prev.values, [HARDWARE_PROFILE_SETTING]: profile } } : prev
    )
    setHardwareProfileSaving(true)
    try {
      const snap = await saveSettings({ values: { [HARDWARE_PROFILE_SETTING]: profile } })
      if (generation !== connectionGenerationRef.current) return true
      setSettingsSnap((prev) => ({
        ...snap,
        schema: prev?.schema ?? snap.schema,
        advancedSchema: prev?.advancedSchema ?? snap.advancedSchema,
      }))
      if (snap.cameras) setCameras(snap.cameras)
      return true
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return true
      setSettingsSnap((prev) => (prev ? { ...prev, values: priorValues } : prev))
      toast.error(`Could not save the device selection: ${String(e).replace(/^Error:\s*/, "")}`)
      return false
    } finally {
      if (generation === connectionGenerationRef.current) setHardwareProfileSaving(false)
    }
  }

  // Persist a settings-dialog save: cameras also mirror to localStorage (the
  // fallback for old hosts / offline), everything else goes to the serve host.
  async function handleSettingsSave(
    patch: SettingsPatch,
    generation = connectionGenerationRef.current
  ) {
    if (generation !== connectionGenerationRef.current) return
    const changedChannelProfiles = (
      [
        ["axol", ["robot.left_channel", "robot.right_channel"]],
        ["mantis", ["mantis.left_channel", "mantis.right_channel"]],
      ] as const
    )
      .filter(([, keys]) =>
        keys.some((key) => Object.prototype.hasOwnProperty.call(patch.values ?? {}, key))
      )
      .map(([profile]) => profile)
    if (patch.cameras) {
      setCameras(patch.cameras)
      persistLocalCameras(patch.cameras)
    }
    if (settingsError) {
      // Old host: only the cameras section is editable, and it saved locally.
      return
    }
    const snap = await saveSettings(patch)
    if (generation !== connectionGenerationRef.current) return
    setSettingsSnap((prev) => ({
      ...snap,
      schema: prev?.schema ?? snap.schema,
      advancedSchema: prev?.advancedSchema ?? snap.advancedSchema,
    }))
    if (snap.cameras) setCameras(snap.cameras)
    if (changedChannelProfiles.length > 0) {
      resetAutoRobotRetry()
      const changedDesiredProfile = changedChannelProfiles.includes(desiredHardwareProfile)
      if (changedDesiredProfile) manualRobotOverrideRef.current = false
      const label =
        changedChannelProfiles.length === 2
          ? "CAN channels"
          : `${changedChannelProfiles[0] === "mantis" ? "Mantis" : "Axol"} CAN channels`
      if (!changedDesiredProfile) {
        toast.info(`${label} saved. They apply the next time that hardware connects.`)
      } else if (isLive || robot?.state === "busy") {
        toast.info(`${label} saved. The robot reconnects when this run stops.`)
      } else {
        toast.info(`${label} saved. Reconnecting when the CAN bus is detected…`)
      }
    }
  }

  // The operations this host offers on the selected device — Mantis hides the
  // Axol-only ones — and the selected one resolved against them: a stored
  // selection the host (or device) doesn't have lands on the first operation
  // instead of an empty panel. A running Axol-only operation stays visible
  // even if another operator selects Mantis mid-run, so it can be stopped.
  const liveCommand = [status, session].find(
    (s) => s && (s.status === "running" || s.status === "starting" || s.status === "stopping")
  )?.command
  const operations = useMemo(() => {
    const all = operationsFromCommands(commands)
    return hardwareProfile === "mantis"
      ? all.filter((o) => o.supportsMantis || o.id === liveCommand)
      : all
  }, [commands, hardwareProfile, liveCommand])
  // An operation's own screen (/control/operations/<id>) decides which
  // operation is selected; the launcher and other pages keep the last one.
  const routeOp = view === "operations" ? pathRest : null
  const currentOp = routeOp ?? selectedOp
  const meta = useMemo(
    () => operations.find((o) => o.id === currentOp) ?? catalogMeta(currentOp) ?? operations[0],
    [operations, currentOp]
  )
  const opId = meta.id
  // The host's own description of this operation, when it has one: the app's
  // catalog name and icon otherwise.
  const hostOffersOp = operations.some((o) => o.id === opId)
  useEffect(() => {
    if (!routeOp) return
    localStorage.setItem("axolOp", routeOp)
    rememberOperation(routeOp)
  }, [routeOp])

  // The newest datasets on the host, for Home; read again on each visit.
  const [recentDatasets, setRecentDatasets] = useState<DatasetInfo[] | null>(null)
  useEffect(() => {
    if (conn.state !== "ok" || view !== "overview") return
    let active = true
    fetchDatasets()
      .then((found) => {
        if (active) setRecentDatasets(found.slice(0, 4))
      })
      .catch(() => {
        if (active) setRecentDatasets(null)
      })
    return () => {
      active = false
    }
  }, [conn.state, view, connectionGeneration])

  function selectOp(op: OperationId) {
    setSelectedOp(op)
    localStorage.setItem("axolOp", op)
  }

  // -- per-operation settings --
  const settings = useMemo(() => settingsByOp[opId] ?? loadOpSettings(opId), [settingsByOp, opId])
  // The device selection is system-wide: a Mantis-capable operation runs on
  // Mantis whenever it is selected, and the idle link auto-connects to it.
  const mantisMode = meta.supportsMantis && hardwareProfile === "mantis"
  // An Axol is connected: its robot model can be picked (until then the
  // picker previews only, like the Jelly card until Jelly is attached).
  const axolConnected = robot?.state === "connected" && (robot?.profile ?? "axol") === "axol"
  // No robot model picked yet: a host that offers the pick (kinematics.mount)
  // and has none saved, on Axol without Jelly (Jelly is detected, never
  // picked). The pick shows above the settings and gates Axol runs until saved.
  //
  // It waits for CAN discovery to settle: hardware still being identified
  // may be the Jelly, which would make the pick moot. (Hosts that predate
  // discovery report none: settled.)
  const canDiscoverySettled =
    canDiscovery == null || canDiscovery.status === "ready" || canDiscovery.status === "configured"
  const needsMount =
    conn.state === "ok" &&
    canDiscoverySettled &&
    !settingsError &&
    hardwareProfile !== "mantis" &&
    settingsSnap?.robotModel !== "mobile" &&
    (settingsSnap?.schema ?? []).some((c) =>
      c.settings.some((f) => f.key === "kinematics.mount")
    ) &&
    settingsSnap?.values["kinematics.mount"] == null
  const desiredHardwareProfile: HardwareProfile = hardwareProfile
  // The robot sim runs simulate: this browser's last pick, else the robot this
  // host is (its model and fitted gripper).
  const simRobot: SimRobot = simRobotChoice ?? {
    model: settingsSnap?.robotModel ?? "classic",
    gripper: parseGripper(settingsSnap?.values["axol.gripper"]),
  }

  // The host derives the robot model (robotModel) from whether Jelly's CAN
  // interfaces are present — but the panel only fetches it with the
  // settings. Follow the 2 s CAN inventory instead of a one-off answer:
  // refetch whenever Jelly appears or goes, or discovery settles, so a Jelly
  // that comes up a moment after the panel connects turns the robot into
  // Axol on Jelly (and drops the "What is Axol mounted on?" pick) without a reload.
  const jellyPresent = canDevices
    ? Boolean(canDevices.wheels?.present || canDevices.lift?.present)
    : null
  const robotModelInputs = conn.state === "ok" ? `${jellyPresent}:${canDiscoverySettled}` : null
  const robotModelInputsRef = useRef<string | null>(null)
  useEffect(() => {
    const prev = robotModelInputsRef.current
    robotModelInputsRef.current = robotModelInputs
    if (prev !== null && robotModelInputs !== null && prev !== robotModelInputs) {
      void loadSettings()
    }
  }, [robotModelInputs, loadSettings])

  const updateSettings = useCallback((op: OperationId, next: Record<string, FormValue>) => {
    setSettingsByOp((prev) => ({ ...prev, [op]: next }))
    saveOpSettings(op, next)
  }, [])

  function setSetting(key: string, value: FormValue) {
    updateSettings(opId, { ...settings, [key]: value })
  }

  function resetSetting(key: string) {
    const next = { ...settings }
    delete next[key]
    updateSettings(opId, next)
  }

  function resetAll() {
    updateSettings(opId, {})
  }

  // -- robot connection --
  const robotConnectClick = useCallback(
    async (profile: HardwareProfile, automatic = false): Promise<boolean | null> => {
      if (!autoRobotMountedRef.current) return null
      const generation = connectionGenerationRef.current
      if (!automatic) {
        resetAutoRobotRetry()
      }
      setRobotBusy(true)
      try {
        const status = await robotConnect(undefined, profile, automatic)
        if (!autoRobotMountedRef.current || generation !== connectionGenerationRef.current)
          return null
        setRobot(status)
        if (!status.connected) {
          throw new Error(
            status.error ?? `Could not connect to ${profile === "mantis" ? "Mantis" : "Axol"}.`
          )
        }
        if (!automatic) manualRobotOverrideRef.current = true
        return true
      } catch (e) {
        if (!autoRobotMountedRef.current || generation !== connectionGenerationRef.current)
          return null
        if (!automatic) {
          toast.error(deviceErrorText(String(e), profile === "mantis" ? "Mantis" : "Axol"))
        }
        return false
      } finally {
        if (autoRobotMountedRef.current && generation === connectionGenerationRef.current)
          setRobotBusy(false)
      }
    },
    [resetAutoRobotRetry, toast]
  )

  async function robotDisconnectClick() {
    const generation = connectionGenerationRef.current
    resetAutoRobotRetry()
    setRobotBusy(true)
    try {
      // Kill any running task and wait for it to exit before releasing the
      // robot connection out from under it, then disconnect.
      if (!(await stopRunningOp(generation))) return
      if (generation !== connectionGenerationRef.current) return
      const status = await robotDisconnect()
      if (generation !== connectionGenerationRef.current) return
      setRobot(status)
      if (status.state !== "disconnected") {
        throw new Error(status.error ?? "Could not disconnect the robot.")
      }
      manualRobotOverrideRef.current = true
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(String(e))
    } finally {
      if (generation === connectionGenerationRef.current) setRobotBusy(false)
    }
  }

  // -- Jelly wheels / lift connections --
  const jellyConnectClick = useCallback(
    async (device: JellyDevice, automatic = false): Promise<boolean | null> => {
      const generation = connectionGenerationRef.current
      if (!automatic) manualJellyOverrideRef.current.add(device)
      setJellyBusy((prev) => ({ ...prev, [device]: true }))
      try {
        const status = await jellyConnect(device, automatic)
        if (generation !== connectionGenerationRef.current) return null
        setJelly(status)
        if (!status[device].connected) {
          throw new Error(
            status[device].error ?? `Could not connect to ${JELLY_DEVICE_LABELS[device]}.`
          )
        }
        return true
      } catch (e) {
        if (generation !== connectionGenerationRef.current) return null
        if (!automatic) toast.error(String(e))
        return false
      } finally {
        if (generation === connectionGenerationRef.current)
          setJellyBusy((prev) => ({ ...prev, [device]: false }))
      }
    },
    [toast]
  )

  const jellyDisconnectClick = useCallback(
    async (device: JellyDevice) => {
      const generation = connectionGenerationRef.current
      manualJellyOverrideRef.current.add(device)
      setJellyBusy((prev) => ({ ...prev, [device]: true }))
      try {
        const status = await jellyDisconnect(device)
        if (generation !== connectionGenerationRef.current) return
        setJelly(status)
        if (status[device].state !== "disconnected") {
          throw new Error(
            status[device].error ?? `Could not disconnect from ${JELLY_DEVICE_LABELS[device]}.`
          )
        }
      } catch (e) {
        if (generation !== connectionGenerationRef.current) return
        toast.error(String(e))
      } finally {
        if (generation === connectionGenerationRef.current)
          setJellyBusy((prev) => ({ ...prev, [device]: false }))
      }
    },
    [toast]
  )

  // -- operation lifecycle --
  // Liveness comes from two sources that can briefly disagree about the same
  // session: `session` (the REST start/stop responses) and `status` (the logs
  // WebSocket). On Stop the REST response is authoritative and immediately
  // reports "exited", but the WebSocket's final "exited" frame can be missed
  // (dropped sentinel on a full subscriber queue, or a flaky link to a remote
  // serve host), leaving its last-seen status stuck at "running". So treat the
  // op as live only when a source reports it active AND neither source reports
  // it finished — a terminal state from either side flips the button to Start.
  const sources = [status, session].filter((s): s is SessionInfo => s != null)
  // Display the most-advanced status across the two sources. The logs
  // WebSocket only ever reports "running" then "exited" — it never emits
  // "stopping" — so during a stop the REST/poll session is ahead; ranking it
  // higher makes the badge show "Stopping" instead of a stale "Running".
  const STATUS_RANK: Record<string, number> = {
    starting: 0,
    running: 1,
    stopping: 2,
    exited: 3,
    error: 3,
  }
  const rank = (s: SessionInfo) => STATUS_RANK[s.status] ?? 0
  const effectiveStatus = sources.reduce<SessionInfo | null>(
    (best, s) => (best && rank(best) >= rank(s) ? best : s),
    null
  )
  // Startup steps come from the status poll: the logs WebSocket sends its
  // session once, when the run starts, so its copy never advances.
  const runProgress =
    session && session.id === effectiveStatus?.id ? session.progress : effectiveStatus?.progress
  // What the running operation is doing while it starts (null once ready).
  const runningStartup = useStartupView(runProgress, {
    device: effectiveStatus?.args?.[HARDWARE_PROFILE_ARG] ? "Mantis" : "Axol",
    sim: Boolean(effectiveStatus?.args?.sim),
  })
  const isLive =
    sources.some(
      (s) => s.status === "running" || s.status === "starting" || s.status === "stopping"
    ) && !sources.some((s) => s.status === "exited" || s.status === "error")
  // The op has been asked to stop and is unwinding (its worker thread is still
  // tearing down / its children are being killed). The Stop button shows a
  // disabled "Stopping…" until a terminal status flips the op back to idle.
  const isStopping = isLive && sources.some((s) => s.status === "stopping")
  const runningOp = isLive ? (effectiveStatus?.command as OperationId) : null
  // Names shown to the operator: the command's label, not its id.
  const commandLabel = (id: string | null | undefined) =>
    id ? (commands.find((c) => c.id === id)?.label ?? id) : null
  const runningLabel = commandLabel(runningOp)
  const selectedLive = isLive && runningOp === opId
  const selectedStopping = isStopping && runningOp === opId

  // Whether the host is currently unsafe to restart / power off / update.
  // `isLive` is the immediate local operation signal and the shared session
  // inventory catches setup/diagnostics from other pages; `update.idle` is the
  // backstop for any remaining non-idle reason. The server guards each request
  // regardless. Mirrors _is_idle: only an operation/session blocks (a merely
  // connected robot is fine).
  // Shared by the update banner and the host tile's power confirmations.
  const hostBusy = isLive || hardwareSessionBusy || !(update?.idle ?? true)
  // Reason shown in the banner; capitalized clause, no trailing period.
  const updateBusyReason = isLive ? "Stop the running operation" : "The Host Computer is busy"
  const settledCanDiscovery = useSettledDiscovery(canDiscovery)
  const canDiscoveryNeedsRetry = discoveryNeedsRecheck(settledCanDiscovery)
  const canDiscoveryChecking = canDiscoveryRetryBusy || canDiscovery?.status === "running"

  // Bring up each detected Jelly device once per (device, interface, server
  // epoch), the way the Axol/Mantis link auto-connects: only from an idle
  // panel, only after CAN discovery has settled, and never for a device the
  // operator disconnected by hand (the server refuses those, and this tab
  // stops asking). Absence is not latched — a replug gets a fresh attempt.
  useEffect(() => {
    if (conn.state !== "ok" || !jelly || !jellySupported || !canDevices) return
    if (
      canDiscoveryBlocksAutoConnect(canDiscovery) ||
      isLive ||
      hardwareSessionBusy ||
      activeCommandSession ||
      !sessionInventoryReady
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
      void jellyConnectClick(device, true)
    }
  }, [
    activeCommandSession,
    canDevices,
    canDiscovery,
    conn.state,
    currentCanServerEpoch,
    hardwareSessionBusy,
    isLive,
    jelly,
    jellyBusy,
    jellyConnectClick,
    jellySupported,
    sessionInventoryReady,
  ])

  // Check plugged-in hardware that hasn't answered again. Quiet: the notice
  // on Overview and Devices says what is wrong until it answers, so neither
  // a pass that still finds it silent nor a refused request is news.
  const [recheckTick, setRecheckTick] = useState(0)
  const recheckCanHardware = useCallback(async () => {
    if (canDiscoveryRetryBusy) return
    const hostGeneration = connectionGenerationRef.current
    setCanDiscoveryRetryBusy(true)
    try {
      const inventory = await discoverCanHardware(true)
      if (!autoRobotMountedRef.current || hostGeneration !== connectionGenerationRef.current) return
      canInventoryKnownRef.current = true
      installCanInventory(inventory)
    } catch {
      // Busy, rate-limited or unreachable: the next check tries again.
    } finally {
      if (hostGeneration === connectionGenerationRef.current) {
        setCanDiscoveryRetryBusy(false)
        setRecheckTick((n) => n + 1)
      }
    }
  }, [canDiscoveryRetryBusy, installCanInventory])

  // Hardware that is plugged in but silent is nearly always switched off, and
  // turning it on changes nothing the host can see. So check it again on its
  // own, backing off from 5 s to a minute, while nothing could be disturbed:
  // the host idle and neither Axol nor Mantis connected (a check may briefly
  // take an idle link down). With a link up, "Check again" does it by hand.
  const recheckDelayRef = useRef<number | null>(null)
  const robotLinkFree = !robot || robot.state === "disconnected" || robot.state === "error"
  const canRecheckOnItsOwn =
    conn.state === "ok" &&
    canDiscoveryNeedsRetry &&
    robotLinkFree &&
    !hostBusy &&
    !activeCommandSession &&
    !robotBusy &&
    !canDiscoveryChecking
  useEffect(() => {
    if (!canDiscoveryNeedsRetry) recheckDelayRef.current = null
  }, [canDiscoveryNeedsRetry])
  useEffect(() => {
    if (!canRecheckOnItsOwn) return
    const delay = nextRecheckDelay(recheckDelayRef.current)
    const timer = window.setTimeout(() => {
      recheckDelayRef.current = delay
      void recheckCanHardware()
    }, delay)
    return () => window.clearTimeout(timer)
  }, [canRecheckOnItsOwn, recheckTick, recheckCanHardware])

  // A fresh Axol/Mantis hub or Jelly base/lift adapter initially appears only
  // as anonymous canX devices. Ask the server to probe and persist its role
  // before the ordinary profile chooser runs. The server is the cross-tab single-flight authority; this
  // latch merely keeps one tab's 2s poll from duplicating the request.
  useEffect(() => {
    if (conn.state !== "ok" || !canDiscovery) return
    if (canDiscovery.status === "ready" && canDiscovery.candidateCount === 0) {
      automaticCanDiscoveryAttemptsRef.current.clear()
      return
    }
    const signature = canDiscoveryAttemptSignature(canDiscovery)
    const idle = !hostBusy && !activeCommandSession && !robotBusy
    if (
      signature === null ||
      !shouldStartCanDiscovery(canDiscovery, robot?.state, autoRobotPollStateKnown(), idle)
    )
      return

    const hostGeneration = connectionGenerationRef.current
    const serverEpoch = currentCanServerEpoch()
    const attemptKey = `${hostGeneration}:${serverEpoch}:${signature}`
    if (automaticCanDiscoveryAttemptsRef.current.has(attemptKey)) return
    automaticCanDiscoveryAttemptsRef.current.add(attemptKey)

    void discoverCanHardware()
      .then((inventory) => {
        if (
          !autoRobotMountedRef.current ||
          hostGeneration !== connectionGenerationRef.current ||
          serverEpoch !== currentCanServerEpoch()
        )
          return
        installCanInventory(inventory)
      })
      .catch((error) => {
        if (
          !autoRobotMountedRef.current ||
          hostGeneration !== connectionGenerationRef.current ||
          serverEpoch !== currentCanServerEpoch()
        )
          return
        if (canDiscoveryRequestCanRetry(error)) {
          // Let the next coherent inventory poll retry after a session-launch
          // race or transient transport/server failure. The backend remains
          // the cross-tab single-flight and per-hardware rate-limit authority.
          automaticCanDiscoveryAttemptsRef.current.delete(attemptKey)
        }
        // Otherwise the result stands as the discovery state, which the
        // hardware notice turns into what to do.
      })
  }, [
    activeCommandSession,
    autoRobotPollStateKnown,
    canDiscovery,
    canServerInstanceId,
    conn.state,
    currentCanServerEpoch,
    hostBusy,
    installCanInventory,
    robot?.state,
    robotBusy,
    toast,
  ])

  // Pick from hardware actually present on the host. If only one configured
  // profile exists, connect it; if both exist, the selected operation wins.
  // Never guess arbitrary can0 devices or switch underneath a live operation.
  useEffect(() => {
    if (conn.state !== "ok") {
      resetAutoRobotRetry()
      manualRobotOverrideRef.current = false
      previousDesiredProfileRef.current = null
      return
    }

    const desiredChanged =
      previousDesiredProfileRef.current !== null &&
      previousDesiredProfileRef.current !== desiredHardwareProfile
    previousDesiredProfileRef.current = desiredHardwareProfile
    if (desiredChanged) {
      resetAutoRobotRetry()
      manualRobotOverrideRef.current = false
    }

    if (!robot || !sessionInventoryReady || (!canProfiles && !legacyCanInventory)) return
    if (
      canDiscoveryBlocksAutoConnect(canDiscovery) ||
      isLive ||
      hardwareSessionBusy ||
      activeCommandSession ||
      robotBusy ||
      robot.state === "busy" ||
      manualRobotOverrideRef.current ||
      !autoRobotPollStateKnown()
    )
      return

    const activeProfile = robot.profile ?? "axol"
    // A saved mapping can move away from every currently attached interface.
    // Do one normal connect onto that new mapping while the old link is still
    // open: the server first closes the stale buses, then returns an explicit
    // error for the absent replacement. This keeps Connected/Start from
    // continuing to describe the old mapping. Ordinary absence never triggers
    // a connect attempt.
    const target = canProfiles
      ? chooseAutoConnectTarget(
          canProfiles,
          desiredHardwareProfile,
          activeProfile,
          robot.channels,
          robot.state === "connected"
        )
      : desiredHardwareProfile
    if (target === null) {
      // Do not latch genuine absence: a later physical replug gets a fresh
      // bounded attempt budget. Driver/netdev cycling for a configured hub
      // remains present through its persisted USB identity.
      resetAutoRobotRetry()
      return
    }
    const targetPresence = canProfiles?.[target]
    const profileSignature = targetPresence
      ? autoConnectSignature(target, targetPresence)
      : `legacy:${target}`
    const serverEpoch = currentCanServerEpoch()
    const signature = `${profileSignature}:host-${serverEpoch}`
    const mappingChanged =
      activeProfile === target &&
      targetPresence !== undefined &&
      robot.channels !== undefined &&
      (robot.channels.left !== targetPresence.channels.left ||
        robot.channels.right !== targetPresence.channels.right)
    if (autoRobotRef.current === signature) return
    if (
      activeProfile !== target ||
      robot.state === "disconnected" ||
      robot.state === "error" ||
      mappingChanged
    ) {
      const attempts = nextAutoConnectAttempt(
        autoRobotAttemptsRef.current.get(signature) ?? 0,
        autoRobotPollStateKnown()
      )
      if (attempts === null) {
        // A down/up inventory oscillation can revisit an old signature. Keep
        // its exhausted budget latched instead of starting attempt 4+.
        autoRobotRef.current = signature
        return
      }
      const timer = window.setTimeout(() => {
        autoRobotRef.current = signature
        const generation = connectionGenerationRef.current
        void robotConnectClick(target, true).then((connected) => {
          if (
            connected !== false ||
            !autoRobotMountedRef.current ||
            generation !== connectionGenerationRef.current ||
            serverEpoch !== currentCanServerEpoch() ||
            manualRobotOverrideRef.current
          )
            return
          if (!autoRobotPollStateKnown()) {
            // A request that settles while inventory/session authority is
            // unknown is not evidence that this hardware target failed.
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
              generation === connectionGenerationRef.current &&
              serverEpoch === currentCanServerEpoch() &&
              autoRobotMountedRef.current &&
              !manualRobotOverrideRef.current &&
              autoRobotRef.current === signature
            ) {
              autoRobotRef.current = null
              // Recovery itself causes the rerender. Do not spend or schedule
              // an attempt while one of the authoritative polls is unknown.
              if (!autoRobotPollStateKnown()) return
              setAutoRobotRetryRevision((revision) => revision + 1)
            }
          }, delay)
        })
      }, 0)
      return () => window.clearTimeout(timer)
    }
    autoRobotRef.current = signature
  }, [
    activeCommandSession,
    autoRobotPollStateKnown,
    autoRobotRetryRevision,
    canProfiles,
    canDiscovery,
    canServerInstanceId,
    conn.state,
    desiredHardwareProfile,
    hardwareSessionBusy,
    isLive,
    legacyCanInventory,
    robot,
    robotBusy,
    robotConnectClick,
    currentCanServerEpoch,
    resetAutoRobotRetry,
    sessionInventoryReady,
  ])

  // Follow the hardware when only one of Axol / Mantis is attached: a saved
  // selection on the absent device would leave its tile marked "selected"
  // while the link auto-connects the other one. Once per inventory change
  // (and host), so an operator who deliberately picks the absent device keeps
  // that choice until the attached hardware changes. Waits for the host's
  // settings so a stored selection isn't judged by this browser's fallback,
  // and for CAN discovery to settle so a hub still being identified isn't
  // read as absent. A rejected save gets the same bounded retries as
  // auto-connect before the latch holds.
  const autoSelectedPresenceRef = useRef<string | null>(null)
  const autoSelectAttemptsRef = useRef(new Map<string, number>())
  useEffect(() => {
    if (conn.state !== "ok" || !canProfiles) return
    if (settingsSnap == null && settingsError == null) return
    if (canDiscoveryBlocksAutoConnect(canDiscovery)) return
    if (isLive || hardwareProfileSaving) return
    const signature = `${connectionGeneration}:${hardwarePresenceSignature(canProfiles)}`
    if (autoSelectedPresenceRef.current === signature) return
    const detected = chooseDetectedHardwareProfile(canProfiles, hardwareProfile)
    if (detected === null) {
      autoSelectedPresenceRef.current = signature
      return
    }
    const timer = window.setTimeout(() => {
      autoSelectedPresenceRef.current = signature
      void selectHardwareProfile(detected).then((saved) => {
        if (saved) return
        const attempts = (autoSelectAttemptsRef.current.get(signature) ?? 0) + 1
        autoSelectAttemptsRef.current.set(signature, attempts)
        const delay = autoConnectRetryDelay(attempts)
        if (delay === null) return
        // The 2 s inventory poll re-runs this effect once the latch clears.
        window.setTimeout(() => {
          if (autoSelectedPresenceRef.current === signature) autoSelectedPresenceRef.current = null
        }, delay)
      })
    }, 0)
    return () => window.clearTimeout(timer)
  }, [
    canDiscovery,
    canProfiles,
    conn.state,
    connectionGeneration,
    hardwareProfile,
    hardwareProfileSaving,
    isLive,
    selectHardwareProfile,
    settingsError,
    settingsSnap,
  ])

  // While an op is live (including the "stopping" window), poll the server's
  // authoritative op status so the panel reliably catches the transition to
  // exited even if the logs WebSocket drops its final status frame. The stop
  // itself returns immediately server-side, so this is what flips the button
  // back to Start once the op has actually torn down.
  //
  // Keep polling past the session's terminal status for as long as the server
  // still reports the runner busy. The session reads error before teardown
  // finishes (the stall watchdog marks it seconds earlier), and the cleanup
  // lockout is only decided at the end of that teardown — so a poll gated on
  // the live session alone would stop on `lockout: false` and never see the
  // flag flip, leaving Start returning 409 with no Re-check control shown.
  // A lockout keeps `op.running` true, so this also tracks it being cleared.
  useEffect(() => {
    if (conn.state !== "ok" || !(isLive || opBusy)) return
    let active = true
    const tick = () => {
      fetchOpStatus()
        .then((op) => {
          if (!active) return
          if (op.session) setSession(op.session)
          setPolicy(op.running ? op.policy : null)
          setLockout(op.lockout === true)
          setOpBusy(op.running)
        })
        .catch(() => {})
    }
    // Sample once up front so a run that ends within the first interval still
    // records the server as busy and the teardown is followed to its end.
    tick()
    const t = setInterval(tick, 1500)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, isLive, opBusy])

  // Refresh the update status the moment an operation starts or stops, so the
  // server's idle state (and thus the banner's blocked state) becomes current
  // without waiting for the slow 60s poll. Skipped while an update is applying
  // (handleUpdate drives its own watch poll then).
  useEffect(() => {
    if (conn.state !== "ok" || updating) return
    let active = true
    fetchUpdateStatus()
      .then((value) => {
        if (active) setUpdate(value)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [conn.state, updating, isLive])

  // Drive an in-flight update to completion on ANY connected computer: advance
  // the phase, surface a failure, and hard-reload once the backend is back on
  // the target release (the hosted front-end is on Vercel, so a reload also
  // pulls the latest UI and reconnects to the restarted server). Keys off the
  // server's "updating" state rather than a local click, so a second computer
  // that opens the panel mid-update behaves like the initiator. Replaces the
  // per-click watch loop handleUpdate used to run itself.
  useEffect(() => {
    if (conn.state !== "ok" || !updating) return
    const target = update?.remoteVersion ?? null
    // The server bounds its own upgrade + provision steps and reports a
    // timeout as an error, so only an unreachable server is given up on: the
    // window restarts on every answer and covers a restart or host reboot.
    const unreachableLimitMs = 10 * 60_000
    let deadline = Date.now() + unreachableLimitMs
    let active = true
    const t = setInterval(async () => {
      if (Date.now() > deadline) {
        clearInterval(t)
        if (active) {
          // Give up auto-watching so the banner leaves the spinner and offers a
          // retry, even if the server is still/again reporting "updating".
          setUpdateAbandoned(true)
          setStartingUpdate(false)
          setUpdatePhase(null)
          toast.error(
            "The robot did not come back online after the update. Reload the page to try again."
          )
        }
        return
      }
      try {
        const u = await fetchUpdateStatus()
        if (!active) return
        deadline = Date.now() + unreachableLimitMs
        setUpdate(u)
        // Real server state is known now — drop the optimistic bridge so a
        // failed status fetch in handleUpdate can't wedge `updating` on.
        setStartingUpdate(false)
        if (u.state === "error") {
          setUpdatePhase(null)
          toast.error(u.error ?? "Update failed.")
          return
        }
        // Reflect the server's current step so the banner shows progress.
        if (u.phase) setUpdatePhase(u.phase)
        // Back on the new release — done.
        if (target && u.version === target) window.location.reload()
      } catch {
        // Server stopped responding: it exited to relaunch (or is briefly
        // unreachable). Show "restarting" (unless the host said it is
        // rebooting) and keep watching for it to return.
        if (active) setUpdatePhase((phase) => (phase === "rebooting" ? phase : "restarting"))
      }
    }, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, updating, update?.remoteVersion, toast])

  // Optional provisioning steps that failed (Lighthouse tracking, camera
  // plugins): the robot runs without them, so warn once per backend run and
  // message rather than blocking anything or repeating on every poll.
  const shownUpdateWarningsRef = useRef(new Set<string>())
  const updateWarning = update?.warning ?? null
  useEffect(() => {
    if (conn.state !== "ok" || !updateWarning) return
    const key = `${connectionGeneration}:${canServerInstanceId ?? ""}:${updateWarning}`
    if (shownUpdateWarningsRef.current.has(key)) return
    shownUpdateWarningsRef.current.add(key)
    toast.warning(updateWarning)
  }, [canServerInstanceId, conn.state, connectionGeneration, toast, updateWarning])

  // Only an operation the host offers on the selected device can run: a
  // Mantis host's Axol-only operation opened by its address stays off.
  const spec = useMemo(
    () => (hostOffersOp ? (commands.find((c) => c.id === opId) ?? null) : null),
    [commands, opId, hostOffersOp]
  )

  // Stop the running task (if any) and wait for it to actually exit before
  // returning, so a disconnect never tears the host/robot link down mid-cleanup.
  // The server-side stop now returns immediately with "stopping" (it force-kills
  // a stuck op in the background), so we poll op status until the op is truly
  // gone rather than relying on the stop response to block.
  async function stopRunningOp(generation = connectionGenerationRef.current): Promise<boolean> {
    if (!isLive) return generation === connectionGenerationRef.current
    const stopped = await stopOperation()
    if (generation !== connectionGenerationRef.current) return false
    setSession(stopped)
    // Bounded so an unkillable op (abandoned server-side) can't wedge the UI;
    // we proceed best-effort after the deadline. Long enough for teleop /
    // collect-data's shutdown park (rest, then zero: up to ~20 s before the
    // server even starts force-killing) plus the dataset finalize.
    const deadline = Date.now() + 60_000
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 500))
      if (generation !== connectionGenerationRef.current) return false
      try {
        const op = await fetchOpStatus()
        if (generation !== connectionGenerationRef.current) return false
        if (op.session) setSession(op.session)
        if (!op.running) return true
      } catch {
        // Host unreachable — nothing left to wait on, but do not let a host
        // switch turn the caller's next step into a request to the new host.
        return generation === connectionGenerationRef.current
      }
    }
    return generation === connectionGenerationRef.current
  }

  async function handleStart() {
    const generation = connectionGenerationRef.current
    setBusy(true)
    try {
      // Only ops that actually require cameras (collect-data / run-policy) are
      // gated on them. Teleop streams whatever cameras are configured but must
      // never be blocked by camera detection, and sim never touches hardware.
      const mantisSelected = mantisMode
      if (hardwareProfile === "mantis" && !meta.supportsMantis) {
        toast.error(`${meta.label} runs on Axol only. Select the Axol tile first.`)
        return
      }
      // Sim is an Axol run mode; it is hidden (and ignored) on Mantis, so only
      // an Axol run can be hardware-free here.
      const isSimSelected = !mantisSelected && isSimRun(meta, settings)
      if (meta.requiresCameras && !isSimSelected) {
        // Reuse the detection we already ran (on connect / when the Cameras
        // dialog closed) instead of spawning a fresh enumeration on every start
        // — re-detecting isn't more accurate anyway (the ZED daemon caches its
        // device list until it's restarted), and the subprocess spawn is what
        // made "Starting" hang. Only detect on demand if we have no result yet.
        let devices = cameraDevices
        let detErr = cameraDetectError
        if (devices === null) {
          setStartPhase("Checking cameras…")
          const detect = await detectCameras()
          if (generation !== connectionGenerationRef.current) return
          setCameraDevices(detect.devices)
          setCameraDetectError(detect.error)
          devices = detect.devices
          detErr = detect.error
        }
        if (detErr) {
          toast.error(`Could not check the cameras: ${detErr}`)
          return
        }
        const missing = missingCameraSerials(cameras, devices ?? [], mantisSelected, {
          stream: meta.streamsVideo,
          record: true,
        })
        if (missing.length > 0) {
          toast.error(
            `Camera ${missing.length > 1 ? "serials" : "serial"} not detected: ${missing.join(
              ", "
            )}. Reconnect the camera, then restart the camera service in Cameras settings.`
          )
          return
        }
      }

      // Send only the panel's per-run fields — the shared settings (and any
      // advanced overrides) are folded in server-side, and stale keys from the
      // old per-op localStorage must not shadow them. On Mantis the Axol-only
      // sim mode is not a per-run field and stays out.
      const runKeys = new Set(
        spec ? perRunFields(spec, meta, hardwareProfile).map((f) => f.key) : []
      )
      const args: Record<string, FormValue> = Object.fromEntries(
        Object.entries(settings).filter(([k]) => runKeys.has(k))
      )
      // The system-wide device selection becomes this run's hardware flag.
      // For collection, snapshot the shared tracking source alongside it:
      // new hosts also return their fully merged args, but this keeps live
      // hints/reset controls tied to the actual run even against an older
      // serve host if the operator edits the saved source mid-run. Teleop
      // never tracks (Mantis teleop is grippers-only), so it takes no source.
      if (mantisSelected) {
        args[HARDWARE_PROFILE_ARG] = true
        if (meta.id === "collect-data") args.mantis_source = mantisSource
      }
      // A sim run simulates the robot picked for it (the "Sim robot" roster),
      // which may differ from the robot's saved model; a run's args override
      // the shared settings.
      if (isSimSelected) {
        args["kinematics.robot_model"] = simRobot.model
        if (simRobot.model === "mobile") args["axol.gripper"] = simRobot.gripper
      }
      // Send the camera spec whenever any serial is assigned — collect-data /
      // run-policy need at least one, while teleop streams whichever are set to
      // the headset (and runs fine with none in sim). Newer hosts also hold the
      // spec in their settings store; sending it stays compatible with old ones.
      const camSpec =
        meta.requiresCameras || cameraCount(cameras, mantisSelected) > 0 ? cameras : undefined
      if (generation !== connectionGenerationRef.current) return
      const result = await startOperation(opId, args, camSpec)
      if (generation !== connectionGenerationRef.current) return
      setSession(result)
      // Fresh run — clear any stale phase; the live poll repopulates it.
      setPolicy(null)
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(String(e))
    } finally {
      if (generation === connectionGenerationRef.current) {
        setStartPhase(null)
        setBusy(false)
      }
    }
  }

  async function handleStop() {
    const generation = connectionGenerationRef.current
    setBusy(true)
    // Reflect "Stopping…" immediately — the server stop returns right away and
    // teardown runs in the background, so don't wait for the response/next poll
    // to flip the button.
    setSession((s) => (s ? { ...s, status: "stopping" } : s))
    try {
      const stopped = await stopOperation()
      if (generation !== connectionGenerationRef.current) return
      setSession(stopped)
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(String(e))
    } finally {
      if (generation === connectionGenerationRef.current) setBusy(false)
    }
  }

  // Ask the server to re-check the motors and drop the hardware-cleanup
  // reservation. It refuses while any motor still answers and is not disabled,
  // so a failure here is the robot saying it is still torqued, not a UI error.
  async function handleClearLockout() {
    const generation = connectionGenerationRef.current
    setLockoutBusy(true)
    try {
      await clearOperationLockout()
      if (generation !== connectionGenerationRef.current) return
      setLockout(false)
      toast.success("The motors are off. The robot is available again.")
      fetchRobotStatus()
        .then((value) => {
          if (generation === connectionGenerationRef.current) setRobot(value)
        })
        .catch(() => {})
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(String(e))
    } finally {
      if (generation === connectionGenerationRef.current) setLockoutBusy(false)
    }
  }

  function handleEpisode(command: string) {
    const generation = connectionGenerationRef.current
    sendEpisodeCommand(command).catch((e) => {
      if (generation === connectionGenerationRef.current) toast.error(String(e))
    })
  }

  // Kick off the available update. The server upgrades and exits (systemd
  // relaunches it); the update-watcher effect — which runs on any computer while
  // the server reports an update in flight — then advances the phase and
  // hard-reloads once the backend is back on the new release.
  async function handleUpdate() {
    if (!update?.remoteVersion) return
    const generation = connectionGenerationRef.current
    setUpdateAbandoned(false)
    setStartingUpdate(true)
    setUpdatePhase("upgrading")
    try {
      await startUpdate()
      if (generation !== connectionGenerationRef.current) return
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(`Could not start the update: ${e}`)
      setStartingUpdate(false)
      setUpdatePhase(null)
      return
    }
    // Pull the now-"updating" status so `updating` derives true and the watcher
    // takes over; then drop the optimistic flag (server state carries it now).
    fetchUpdateStatus()
      .then((u) => {
        if (generation !== connectionGenerationRef.current) return
        setUpdate(u)
        setStartingUpdate(false)
      })
      .catch(() => {})
  }

  const viewerHost = serverHost || hostInfo?.lanIp || ""
  const mantisSource = String(settingsSnap?.values["mantis.source"] ?? "lighthouse")
  // Child settings actions can finish after their old-host tree unmounts.
  // Capture the generation represented by these callbacks so an old camera
  // daemon restart or diagnostics launch cannot refresh/repopulate a new host.
  const renderedConnectionGeneration = connectionGeneration

  // UI/backend skew warning (stale local bundle, or hosted UI on a different
  // release than the robot). Suppressed while the update banner covers the
  // same ground — an available update *is* the mismatch's remediation — and
  // while an update is applying (the page hard-reloads when it lands).
  const mismatch = useMemo(
    () => (conn.state === "ok" ? versionMismatch(hostInfo) : null),
    [conn.state, hostInfo]
  )

  // An unknown /control/... path lands on the overview instead of a blank page.
  if (view === null) return <Navigate to="/control" replace />

  const connected = conn.state === "ok"

  const hostDot: StatusDot =
    conn.state === "ok"
      ? "ok"
      : conn.state === "err"
        ? "err"
        : conn.state === "idle"
          ? "idle"
          : "warn"
  const deviceLabel = hardwareProfile === "mantis" ? "Mantis" : "Axol"
  const robotUp = robot?.state === "connected" || robot?.state === "busy"
  const host: HostSummary = {
    dot: hostDot,
    label:
      conn.state === "ok"
        ? hostInfo?.hostname || serverHost || "Connected"
        : conn.state === "err"
          ? "Can't reach the robot"
          : conn.state === "idle"
            ? "Not connected"
            : "Connecting…",
    detail: connected ? `${deviceLabel} ${robotUp ? "connected" : "not connected"}` : null,
    version: connected ? (update?.version ?? hostInfo?.version ?? null) : null,
    // Before a host answers, the pill connects; after, it opens the devices.
    onClick: connected ? () => navigate("/control/devices") : () => setSetupOpen(true),
  }

  const badges: NavBadges = {
    "/control/devices": needsMount || lockout || canDiscoveryNeedsRetry ? "attention" : undefined,
    "/control/operations": isLive ? "running" : undefined,
    "/control/settings": settingsDirty ? "attention" : undefined,
  }

  // A live run follows the operator around the app: every other page shows
  // what is running, with a way back to it.
  const runningChip =
    isLive && !(view === "operations" && routeOp === runningOp) ? (
      <Link
        to={
          runningOp ? `/control/operations/${encodeURIComponent(runningOp)}` : "/control/operations"
        }
        className="flex h-8 items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/[0.08] px-3 text-xs text-emerald-200 transition-colors hover:bg-emerald-400/[0.14]"
      >
        <span className="size-2 animate-pulse rounded-full bg-emerald-400" />
        <span className="max-w-40 truncate">
          {isStopping
            ? "Stopping"
            : runningStartup
              ? `Starting ${runningStartup.percent}%`
              : "Running"}
          : {runningLabel}
        </span>
      </Link>
    ) : null

  // Alerts that apply wherever the operator is: an update, a version skew, a
  // robot held after a failed shutdown, and hardware still to identify. Not on
  // Request data, which works without the robot. The connection itself shows
  // in the header pill, and on the pages that need it.
  const notices = (
    <>
      {update?.updateAvailable && (
        <UpdateBanner
          update={update}
          updating={updating}
          phase={updatePhase}
          blocked={hostBusy}
          busyReason={updateBusyReason}
          onUpdate={handleUpdate}
        />
      )}

      {mismatch && !updating && !update?.updateAvailable && (
        <VersionMismatchBanner mismatch={mismatch} />
      )}

      {lockout && (
        <div className="flex flex-col gap-2 rounded-lg border border-red-400/25 bg-red-400/[0.05] p-3 text-xs text-red-200/80 sm:flex-row sm:items-center sm:justify-between">
          <p>
            The robot is locked because the last operation could not confirm that the motors are
            off. Cut motor power with the E-Stop, then press Re-check motors.
          </p>
          <Button
            variant="outline"
            size="sm"
            className="shrink-0"
            disabled={lockoutBusy}
            onClick={handleClearLockout}
          >
            {lockoutBusy ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            Re-check motors
          </Button>
        </div>
      )}
    </>
  )

  const saveMount = async (mount: AxolMount) => {
    await handleSettingsSave(
      { values: { "kinematics.mount": mount } },
      renderedConnectionGeneration
    )
    toast.success("Mount saved. You can change it in Axol settings → Robot.")
  }
  const mountCard = needsMount && <MountRequired locked={!axolConnected} onSave={saveMount} />

  const page: Record<ControlView, { title: string; description?: string }> = {
    overview: { title: "Overview" },
    operations: { title: "Operations" },
    datasets: {
      title: "Datasets",
      description: "Watch recorded episodes, check their robot data and fix their task text.",
    },
    "request-data": {
      title: "Request data",
      description:
        "With Teleop Data Collection, Almond collects labeled data for your task and delivers it in LeRobot format, ready for training.",
    },
    devices: {
      title: "Devices",
      description: "The Host Computer and the hardware connected to it.",
    },
    settings: {
      title: "Settings",
      description: "Settings here apply to every operation on this robot and to every browser.",
    },
    logs: { title: "Logs" },
  }

  const settingsScope = settingsScopeForTab(settingsTab)

  // The operations as tiles: the app's catalog, with what the host says.
  const launcherItems: LauncherItem[] = launcherEntries({
    hostOperations: operations,
    connected,
    mantis: hardwareProfile === "mantis",
  }).map((entry) => {
    const hostMeta = operations.find((o) => o.id === entry.id) ?? null
    return {
      entry,
      status: operationStatus({
        entry,
        hostMeta,
        available: commands.find((c) => c.id === entry.id)?.available ?? false,
        connected,
        running: runningOp === entry.id,
        otherRunning: isLive && runningOp !== entry.id,
        needsMount,
        hasCameras: selectedHasCamera,
      }),
    }
  })
  const quickEntries = quickStartOperations(
    launcherItems.map((i) => i.entry),
    recentOperations()
  )
  const currentEntry = catalogEntry(opId, meta)
  // The app's own one-line summary leads the operation's screen; the host's
  // wording only for an operation the app doesn't know.
  const displayMeta =
    currentEntry.summary && currentEntry.summary !== meta.description
      ? { ...meta, description: currentEntry.summary }
      : meta
  const runningEntry = runningOp
    ? catalogEntry(
        runningOp,
        operations.find((o) => o.id === runningOp)
      )
    : null

  // What Home asks the operator to sort out, each with the page that fixes it.
  const attention: AttentionItem[] = []
  if (needsMount) {
    attention.push(
      axolConnected
        ? {
            key: "mount",
            text: "Choose what Axol is mounted on, so the arms don't hit it.",
            content: <MountChooser onSave={saveMount} />,
          }
        : {
            key: "mount",
            text: "Connect Axol, then choose what it is mounted on.",
            action: "Connect Axol",
            onAction: () => connectDevice("axol"),
          }
    )
  }
  if (connected && !selectedHasCamera && settingsSnap !== null) {
    attention.push({
      key: "cameras",
      text: "No cameras are assigned. Collect data and Run policy need at least one.",
      action: "Assign cameras",
      onAction: () => openSettings(camerasSettingsTab(hardwareProfile)),
    })
  }

  // The devices at a glance on Home. Mantis and Jelly show once they are
  // plugged in (or selected), so an Axol-only setup sees just Axol.
  const hardwareNotice = connected ? discoveryNotice(settledCanDiscovery) : null
  const deviceSummaries: DeviceSummary[] = [
    {
      key: "host",
      name: "Host Computer",
      dot: hostDot,
      status: connected
        ? `${hostInfo?.hostname || serverHost || "Connected"}`
        : conn.state === "err"
          ? "Can't reach it"
          : conn.state === "idle"
            ? "Not connected"
            : "Connecting…",
      connectable: !connected,
    },
  ]
  if (connected) {
    const linkProfile = robot?.profile ?? "axol"
    for (const profile of ["axol", "mantis"] as const) {
      const name = profile === "mantis" ? "Mantis" : "Axol"
      const present = canProfiles?.[profile].present ?? false
      const active = linkProfile === profile
      const state = active ? (robot?.state ?? "disconnected") : "disconnected"
      if (profile === "mantis" && !present && !active && hardwareProfile !== "mantis") continue
      const faults = active ? (robot?.faults ?? []) : []
      const canConnect = (state === "disconnected" || state === "error") && present && !isLive
      deviceSummaries.push({
        key: profile,
        name,
        dot:
          state === "connected"
            ? faults.length
              ? "err"
              : "ok"
            : state === "busy"
              ? "busy"
              : state === "connecting"
                ? "warn"
                : state === "error"
                  ? "err"
                  : present
                    ? "warn"
                    : "idle",
        status:
          state === "connected"
            ? faults.length
              ? "Motor fault"
              : "Connected"
            : state === "busy"
              ? "In use"
              : state === "connecting"
                ? "Connecting…"
                : state === "error"
                  ? deviceErrorText(robot?.error, name)
                  : present
                    ? "Plugged in"
                    : "Not plugged in",
        detail: state === "error" ? robot?.error : null,
        connectable: canConnect,
        connectBusy: robotBusy,
      })
    }
    if (jellySupported) {
      const views = JELLY_DEVICES.map((device) => ({
        device,
        view: jellyDeviceView(device, jelly, canDevices?.[device], jellySupported),
        present: canDevices?.[device]?.present ?? false,
      }))
      const live = views.filter((v) => v.view.state === "connected" || v.view.state === "busy")
      const plugged = views.filter(
        (v) => v.present && (v.view.state === "disconnected" || v.view.state === "error")
      )
      if (live.length || views.some((v) => v.present)) {
        deviceSummaries.push({
          key: "jelly",
          name: "Jelly",
          dot: views.some((v) => v.view.dot === "err") ? "err" : live.length ? "ok" : "warn",
          status:
            live.length === views.length
              ? "Wheels and lift connected"
              : live.length
                ? `${live[0].device === "wheels" ? "Wheels" : "Lift"} connected`
                : "Plugged in",
          connectable: plugged.length > 0 && !isLive,
          connectBusy: plugged.some((v) => jellyBusy[v.device]),
        })
      }
    }
  }
  // A row's Connect: the host dialog, the Axol/Mantis link, or every Jelly
  // device that is plugged in but not connected.
  function connectDevice(key: string) {
    if (key === "host") setSetupOpen(true)
    else if (key === "axol" || key === "mantis") void robotConnectClick(key)
    else if (key === "jelly") {
      for (const device of JELLY_DEVICES) {
        const state = jelly?.[device]?.state ?? "disconnected"
        if (canDevices?.[device]?.present && (state === "disconnected" || state === "error")) {
          void jellyConnectClick(device)
        }
      }
    }
  }

  // Checking by hand is only offered where the panel won't check on its own.
  const hardwareCheck =
    hardwareNotice && !robotLinkFree && !isLive ? () => void recheckCanHardware() : undefined
  const hardwareNoticeCard = hardwareNotice && (
    <HardwareNoticeCard
      notice={hardwareNotice}
      onCheckAgain={hardwareCheck}
      checking={canDiscoveryChecking}
    />
  )

  // The button beside each "before you start" item on an operation's screen.
  function fixFor(blocker: string): BlockerFix | null {
    if (blocker === "Choose what Axol is mounted on") {
      return { label: "Choose mount", onClick: () => navigate("/control/devices") }
    }
    if (/^Connect (Axol|Mantis)/.test(blocker)) {
      return { label: "Open Devices", onClick: () => navigate("/control/devices") }
    }
    if (blocker.startsWith("Wait:") && runningOp && runningOp !== opId) {
      return { label: "Open it", onClick: openRunningOperation }
    }
    if (blocker.includes("Cameras tab of Mantis settings")) {
      return { label: "Assign cameras", onClick: () => openSettings("mantis-cameras") }
    }
    if (blocker.includes("Cameras tab of Axol settings")) {
      return { label: "Assign cameras", onClick: () => openSettings("cameras") }
    }
    if (blocker.includes("Robot tab of Axol settings")) {
      return { label: "Open Axol settings", onClick: () => openSettings("robot") }
    }
    if (blocker.includes("Mantis settings") || blocker.includes("Mantis trackers")) {
      return { label: "Open Mantis settings", onClick: () => openSettings("mantis-tracking") }
    }
    if (blocker.startsWith("Fix the motor fault")) {
      return { label: "Open Diagnostics", onClick: () => navigate("/diagnostics") }
    }
    return null
  }

  // The newest run's error, if its output reported one after it started. The
  // log isn't on screen by default, so the operation page surfaces it.
  const lastStart = lastLineIndex(lines, "[serve] starting")
  const lastErrorAt = lastLineIndex(lines, "[serve] error:")
  const runError =
    lastErrorAt > lastStart ? lines[lastErrorAt].replace(/^\[serve\] error:\s*/, "") : null

  return (
    <>
      <AppShell
        title={page[view].title}
        back={routeOp ? "/control/operations" : undefined}
        description={page[view].description}
        actions={runningChip}
        notices={view === "request-data" ? undefined : notices}
        badges={badges}
        host={host}
      >
        {view === "overview" && (
          <>
            {hardwareNoticeCard}

            <AttentionList items={attention} />

            {(conn.state === "idle" || conn.state === "err") && (
              <ConnectHero failed={conn.state === "err"} onConnect={() => setSetupOpen(true)} />
            )}

            {isLive && runningEntry && (
              <RunningCard
                entry={runningEntry}
                startedAt={effectiveStatus?.status === "running" ? effectiveStatus.startedAt : null}
                startup={runningStartup}
                stopping={isStopping}
                busy={busy}
                onOpen={openRunningOperation}
                onStop={() => void handleStop()}
              />
            )}

            {connected && <DevicesRow devices={deviceSummaries} onConnect={connectDevice} />}

            <QuickStart entries={quickEntries} onOpen={openOperation} />

            {!isLive && <DataCollectionOffer operation="overview" />}

            {connected && recentDatasets && <RecentDatasets datasets={recentDatasets} />}
          </>
        )}

        {view === "devices" && (
          <>
            {hardwareNoticeCard}

            <ConnectionsBar
              conn={conn.state}
              host={serverHost}
              hostName={hostInfo?.hostname}
              version={update?.version ?? hostInfo?.version ?? null}
              onOpenSetup={() => setSetupOpen(true)}
              onHostDisconnect={hostDisconnectClick}
              opRunning={hostBusy}
              hostInstalling={update?.installing ?? false}
              robot={robot}
              robotBusy={robotBusy}
              canProfiles={canProfiles}
              onRobotConnect={robotConnectClick}
              onRobotDisconnect={robotDisconnectClick}
              jelly={jelly}
              jellySupported={jellySupported}
              jellyBusy={jellyBusy}
              canDevices={canDevices}
              onJellyConnect={(device) => void jellyConnectClick(device)}
              onJellyDisconnect={(device) => void jellyDisconnectClick(device)}
              selectedProfile={hardwareProfile}
              onSelectProfile={(profile) => void selectHardwareProfile(profile)}
              selectDisabled={isLive || hardwareProfileSaving}
              selectDisabledReason={
                isLive ? "Stop the running operation to change the device." : null
              }
              selectSaving={hardwareProfileSaving}
              onOpenSettings={openSettingsScope}
            />

            {mountCard}

            {connected && !selectedHasCamera && settingsSnap !== null && (
              <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-white/[0.02] p-4">
                <Camera className="size-4 shrink-0 text-white/40" />
                <p className="min-w-0 flex-1 text-sm text-white/60">
                  No cameras are assigned. Collect data and Run policy need at least one.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => openSettings(camerasSettingsTab(hardwareProfile))}
                >
                  Assign cameras
                </Button>
              </div>
            )}
          </>
        )}

        {view === "operations" && !routeOp && (
          <OperationLauncher items={launcherItems} onOpen={openOperation} />
        )}

        {view === "operations" && routeOp && (
          <div className="flex min-w-0 flex-col gap-4">
            <OperationPanel
              key={renderedConnectionGeneration}
              entry={currentEntry}
              fixFor={fixFor}
              onConnect={() => setSetupOpen(true)}
              meta={displayMeta}
              spec={spec}
              settings={settings}
              hardwareProfile={hardwareProfile}
              mantisSource={mantisSource}
              sharedValues={settingsSnap?.values ?? null}
              onChange={setSetting}
              onReset={resetSetting}
              onResetAll={resetAll}
              onOpenSettings={() => openSettingsScope(hardwareProfile)}
              cameras={cameras}
              robot={robot}
              live={selectedLive}
              stopping={selectedStopping}
              busy={busy}
              session={selectedLive ? effectiveStatus : null}
              progress={selectedLive ? runProgress : null}
              host={viewerHost}
              viewerPort={viewerPort}
              vrPort={hostInfo?.vrPort ?? 8000}
              startPhase={startPhase}
              hostBlocker={
                activeCommandSession
                  ? `${commandLabel(activeCommandSession.command)} is running`
                  : isLive && !selectedLive
                    ? `${runningLabel ?? "Another operation"} is running`
                    : null
              }
              needsMount={needsMount}
              simRobot={simRobot}
              onSimRobotChange={(next) => {
                setSimRobotChoice(next)
                try {
                  localStorage.setItem(SIM_ROBOT_KEY, JSON.stringify(next))
                } catch {
                  // Private mode / storage off: the choice lasts this page view.
                }
              }}
              connected={connected}
              policy={selectedLive ? policy : null}
              onStart={handleStart}
              onStop={handleStop}
              onEpisode={handleEpisode}
            />

            {lines.length > 0 && (
              <LogPeek
                line={lines[lines.length - 1]}
                failure={
                  runError && effectiveStatus?.command === opId
                    ? `${commandLabel(opId)} stopped with an error: ${runError}`
                    : null
                }
              />
            )}

            {(meta.id === "teleop" || meta.id === "collect-data") && !isLive && (
              <DataCollectionOffer operation={meta.id} />
            )}

            {showsDatasetPreview(meta.id) && connected && (
              <Link
                to="/control/datasets"
                className="flex w-fit items-center gap-1.5 text-xs text-white/40 transition-colors hover:text-white/70"
              >
                Review recorded episodes in Datasets
                <ArrowRight className="size-3.5" />
              </Link>
            )}
          </div>
        )}

        {view === "datasets" &&
          (connected ? (
            <DatasetPreview
              key={`datasets-${renderedConnectionGeneration}`}
              connected
              embedded
              robot={{
                model: settingsSnap?.robotModel,
                gripper: parseGripper(settingsSnap?.values["axol.gripper"]),
              }}
              liveDataset={policy?.dataset ?? null}
              episodesRecorded={policy?.episodesRecorded ?? null}
            />
          ) : (
            <EmptyState onConnect={() => setSetupOpen(true)}>
              Connect to the Host Computer to see its datasets.
            </EmptyState>
          ))}

        {view === "request-data" && (
          <RequestDataPage
            source={requestSource(new URLSearchParams(location.search).get("from"))}
          />
        )}

        {view === "settings" && !connected && (
          <EmptyState onConnect={() => setSetupOpen(true)}>
            Connect to the Host Computer to see its settings.
          </EmptyState>
        )}

        {view === "settings" && connected && (
          <div
            role="tablist"
            aria-label="Settings for"
            className="flex w-full gap-1 overflow-x-auto rounded-xl border border-white/10 bg-white/[0.02] p-1 sm:w-fit"
          >
            {SETTINGS_SCOPES.map((s) => (
              <button
                key={s.key}
                type="button"
                role="tab"
                aria-selected={s.key === settingsScope}
                onClick={() => {
                  if (s.key !== settingsScope) openSettings(defaultSettingsTab(s.key))
                }}
                className={cn(
                  "flex-1 rounded-lg px-4 py-1.5 text-sm whitespace-nowrap transition-colors sm:flex-none",
                  s.key === settingsScope
                    ? "bg-[#eff483]/10 text-[#eff483]"
                    : "text-white/55 hover:bg-white/[0.05] hover:text-white/85"
                )}
              >
                {s.label}
              </button>
            ))}
          </div>
        )}

        {/* Mounted on every page so staged edits survive a trip elsewhere;
            only its body is dropped while another page shows. */}
        {connected && (
          <div hidden={view !== "settings"}>
            <SettingsSection
              open={view === "settings"}
              embedded
              onOpenChange={() => {}}
              tab={settingsTab}
              onTabChange={(tab) =>
                navigate(`/control/settings/${encodeURIComponent(tab)}`, { replace: true })
              }
              onDirtyChange={setSettingsDirty}
              snapshot={settingsSnap}
              supportError={settingsError}
              cameras={cameras}
              onSave={(patch) => handleSettingsSave(patch, renderedConnectionGeneration)}
              devices={cameraDevices}
              detecting={cameraDetecting}
              onRefresh={() => void refreshCameras(renderedConnectionGeneration)}
              usb={usb}
              usbBusy={usbBusy}
              onUsbConnect={() => void usbConnectClick(renderedConnectionGeneration)}
              actionBlocker={
                isLive
                  ? `${runningLabel ?? "An operation"} is running`
                  : activeCommandSession
                    ? `${commandLabel(activeCommandSession.command)} is running`
                    : null
              }
              activeCommandSession={activeCommandSession}
              axolConnected={axolConnected}
              onCommandSessionChange={(next) => {
                if (connectionGenerationRef.current === renderedConnectionGeneration) {
                  setActiveCommandSession(next)
                }
              }}
            />
          </div>
        )}

        {view === "logs" && (
          <>
            <p className="-mt-1 text-sm text-white/50">
              {effectiveStatus
                ? `Output from ${commandLabel(effectiveStatus.command)}${isLive ? ", which is running" : ""}.`
                : "Output from the next operation you start shows here."}
            </p>
            <LogConsole lines={lines} fill />
          </>
        )}
      </AppShell>

      <SetupDialog
        open={setupOpen}
        onClose={() => setSetupOpen(false)}
        host={serverHost}
        hostHistory={hostHistory}
        onChangeHost={updateServerHost}
        conn={conn}
        onConnect={() => loadServer(serverHost)}
      />
    </>
  )
}

/** Index of the last line starting with `prefix`, or -1. */
function lastLineIndex(lines: string[], prefix: string): number {
  for (let i = lines.length - 1; i >= 0; i--) if (lines[i].startsWith(prefix)) return i
  return -1
}

/** A page that needs the Host Computer, before one is connected. */
function EmptyState({ children, onConnect }: { children: ReactNode; onConnect?: () => void }) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-white/10 p-6 text-center">
      <span className="flex size-11 items-center justify-center rounded-2xl bg-white/[0.04] text-white/40">
        <Plug className="size-5" />
      </span>
      <p className="max-w-sm text-sm text-white/55">{children}</p>
      {onConnect && <Button onClick={onConnect}>Connect</Button>}
    </div>
  )
}

/** The newest log line under the operation, linking to the full log. A run
 *  that failed shows its error instead, so it is seen without the log open. */
function LogPeek({ line, failure }: { line: string; failure: string | null }) {
  return (
    <Link
      to="/control/logs"
      className={cn(
        "group flex min-w-0 items-center gap-3 rounded-lg border px-3 py-2 transition-colors",
        failure
          ? "border-red-400/25 bg-red-400/[0.05] hover:border-red-400/40"
          : "border-white/10 bg-white/[0.02] hover:border-white/20"
      )}
    >
      {failure ? (
        <CircleAlert className="size-4 shrink-0 text-red-300" />
      ) : (
        <ScrollText className="size-4 shrink-0 text-white/35" />
      )}
      {failure ? (
        <span className="line-clamp-2 min-w-0 flex-1 text-xs text-red-200/85" title={failure}>
          {failure}
        </span>
      ) : (
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-white/50" title={line}>
          {line}
        </span>
      )}
      <span className="flex shrink-0 items-center gap-1 text-xs text-white/45 group-hover:text-white/80">
        View logs
        <ArrowRight className="size-3.5" />
      </span>
    </Link>
  )
}
