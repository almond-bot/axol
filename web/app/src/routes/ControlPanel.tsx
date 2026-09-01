import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import {
  OPERATIONS,
  cameraCount,
  detectCameras,
  fetchCanInterfaces,
  fetchCommands,
  fetchInfo,
  fetchOpStatus,
  fetchRobotStatus,
  fetchSessions,
  fetchSettings,
  fetchUpdateStatus,
  fetchUsbStatus,
  isSimRun,
  loadOpSettings,
  missingCameraSerials,
  operationsFromCommands,
  perRunFields,
  probeUpdateStatus,
  robotConnect,
  robotDisconnect,
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
  type CanProfileInventory,
  type CommandSpec,
  type FormValue,
  type HardwareProfile,
  type OperationId,
  type OperationMeta,
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
  chooseAutoConnectTarget,
  nextAutoConnectAttempt,
} from "@/lib/can-auto-connect"
import { InstallerMigrationBanner, UpdateBanner } from "@/components/update-banner"
import { VersionMismatchBanner } from "@/components/version-mismatch-banner"
import { requiresInstallerMigration } from "@/lib/update-migration"
import { versionMismatch } from "@/lib/version"
import { ConnectionsBar } from "@/components/connections-bar"
import { OperationPanel } from "@/components/operation-panel"
import { LogConsole } from "@/components/log-console"
import { SetupDialog, type ConnState } from "@/components/setup-dialog"
import { SettingsSection, type SettingsTab } from "@/components/settings/settings-section"
import { SiteNav } from "@/components/site-nav"
import { useToast } from "@/components/ui/toast"

type OpSettings = Record<OperationId, Record<string, FormValue>>

const DEFAULT_CAMERAS: CameraSpec = {
  serials: { overhead: "", left_arm: "", right_arm: "" },
  mantis_serials: { left_arm: "", right_arm: "" },
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
      return {
        ...DEFAULT_CAMERAS,
        ...parsed,
        serials: { ...DEFAULT_CAMERAS.serials, ...(parsed.serials ?? {}) },
        mantis_serials: {
          ...DEFAULT_CAMERAS.mantis_serials,
          ...(parsed.mantis_serials ?? {
            left_arm: parsed.serials?.left_arm ?? "",
            right_arm: parsed.serials?.right_arm ?? "",
          }),
        },
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

export default function ControlPanel() {
  const toast = useToast()
  const [commands, setCommands] = useState<CommandSpec[]>([])
  const [conn, setConn] = useState<{ state: ConnState; message?: string }>({ state: "loading" })
  const [serverHost, setServerHost] = useState<string>(
    () => localStorage.getItem("axolServerHost") ?? ""
  )
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
  // A successful response without profile summaries identifies an older host;
  // retain its historical selected-profile auto-connect behavior.
  const [legacyCanInventory, setLegacyCanInventory] = useState(false)
  const [usb, setUsb] = useState<UsbStatus | null>(null)
  const [usbBusy, setUsbBusy] = useState(false)
  const [cameras, setCameras] = useState<CameraSpec>(() => loadCameras())
  const [settingsOpen, setSettingsOpen] = useState(() => cameraCount(loadCameras()) === 0)
  // Shared settings stored on the serve host (~/.almond/settings.json); null
  // until fetched. settingsError marks a host too old for the settings API —
  // cameras then fall back to the legacy localStorage flow.
  const [settingsSnap, setSettingsSnap] = useState<SettingsSnapshot | null>(null)
  const [settingsError, setSettingsError] = useState<string | null>(null)
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
  const [busy, setBusy] = useState(false)
  // Short label shown on the Start button while a start is being prepared (e.g.
  // "Checking cameras…"), so the wait isn't an opaque spinner — mirrors the
  // update banner's phase display.
  const [startPhase, setStartPhase] = useState<string | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("cameras")
  // Anchor for the on-page settings card, so "…live in Settings" links can
  // scroll to it.
  const settingsRef = useRef<HTMLDivElement>(null)
  // Every API helper targets the module-level server base. A slow response
  // from the previous host must therefore never update this host's UI (or
  // trigger the legacy camera-settings migration against the new host).
  const connectionGenerationRef = useRef(0)
  const [connectionGeneration, setConnectionGeneration] = useState(0)

  const { lines, status } = useSessionLogs(session?.id ?? null)

  const hasConfiguredCamera = cameraCount(cameras) > 0
  useEffect(() => {
    // Keep setup in view until the first camera is assigned. Later camera
    // presence changes drive the default state, but manual toggles stay put.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSettingsOpen(!hasConfiguredCamera)
  }, [hasConfiguredCamera])

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
      try {
        // Must be the first API probe. v0.1.0-v0.1.2 have no update endpoint,
        // while their info/op-status endpoints start a destructive unpinned
        // upgrade from main merely by being read.
        const initialUpdate = await probeUpdateStatus()
        if (generation !== connectionGenerationRef.current) return
        if (initialUpdate === null) {
          setConn({ state: "migration" })
          setSetupOpen(false)
          return
        }
        setUpdate(initialUpdate)
        const cmds = await fetchCommands()
        if (generation !== connectionGenerationRef.current) return
        setCommands(cmds)
        setConn({ state: "ok" })
        setSetupOpen(false)
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
      fetchCanInterfaces()
        .then((inventory) => {
          if (!active) return
          canInventoryKnownRef.current = true
          if (inventory.profiles) {
            setCanProfiles(inventory.profiles)
            setLegacyCanInventory(false)
          } else {
            setCanProfiles(null)
            setLegacyCanInventory(true)
          }
        })
        .catch((error) => {
          if (!active) return
          if (String(error).includes("HTTP 404")) {
            canInventoryKnownRef.current = true
            setCanProfiles(null)
            setLegacyCanInventory(true)
            return
          }
          canInventoryKnownRef.current = false
          setCanProfiles(null)
          setLegacyCanInventory(false)
        })
      fetchUsbStatus()
        .then((u) => {
          if (active) setUsb(u)
        })
        .catch(() => {})
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state])

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
    setSettingsTab(tab)
    setSettingsOpen(true)
    settingsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
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
          ? "CAN mappings"
          : `${changedChannelProfiles[0] === "mantis" ? "Mantis" : "Axol"} CAN mapping`
      if (!changedDesiredProfile) {
        toast.info(`${label} saved. It will apply the next time that hardware connects.`)
      } else if (isLive || robot?.state === "busy") {
        toast.info(`${label} saved. The link will reconnect when this run stops.`)
      } else {
        toast.info(`${label} saved. Reconnecting the link when its configured bus is detected…`)
      }
    }
  }

  // The operations this host offers, and the selected one resolved against
  // them: a stored selection the host doesn't have lands on its first
  // operation instead of an empty panel.
  const operations = useMemo(() => operationsFromCommands(commands), [commands])
  const meta = useMemo(
    () => operations.find((o) => o.id === selectedOp) ?? operations[0],
    [operations, selectedOp]
  )
  const opId = meta.id

  function selectOp(op: OperationId) {
    setSelectedOp(op)
    localStorage.setItem("axolOp", op)
  }

  // -- per-operation settings --
  const settings = useMemo(() => settingsByOp[opId] ?? loadOpSettings(opId), [settingsByOp, opId])
  const mantisMode = meta.supportsMantis && Boolean(settings.mantis)
  const desiredHardwareProfile: HardwareProfile = mantisMode ? "mantis" : "axol"

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
          throw new Error(status.error ?? `Could not connect the ${profile} CAN link`)
        }
        if (!automatic) manualRobotOverrideRef.current = true
        return true
      } catch (e) {
        if (!autoRobotMountedRef.current || generation !== connectionGenerationRef.current)
          return null
        if (!automatic) toast.error(String(e))
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
        throw new Error(status.error ?? "Could not disconnect the robot CAN link")
      }
      manualRobotOverrideRef.current = true
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(String(e))
    } finally {
      if (generation === connectionGenerationRef.current) setRobotBusy(false)
    }
  }

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
  const isLive =
    sources.some(
      (s) => s.status === "running" || s.status === "starting" || s.status === "stopping"
    ) && !sources.some((s) => s.status === "exited" || s.status === "error")
  // The op has been asked to stop and is unwinding (its worker thread is still
  // tearing down / its children are being killed). The Stop button shows a
  // disabled "Stopping…" until a terminal status flips the op back to idle.
  const isStopping = isLive && sources.some((s) => s.status === "stopping")
  const runningOp = isLive ? (effectiveStatus?.command as OperationId) : null
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
  const updateBusyReason = isLive ? "Stop the running operation" : "The server is busy"

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
    const signature = `${profileSignature}:host-${robotStatusRecoveryEpochRef.current}`
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
    conn.state,
    desiredHardwareProfile,
    hardwareSessionBusy,
    isLive,
    legacyCanInventory,
    robot,
    robotBusy,
    robotConnectClick,
    resetAutoRobotRetry,
    sessionInventoryReady,
  ])

  // While an op is live (including the "stopping" window), poll the server's
  // authoritative op status so the panel reliably catches the transition to
  // exited even if the logs WebSocket drops its final status frame. The stop
  // itself returns immediately server-side, so this is what flips the button
  // back to Start once the op has actually torn down.
  useEffect(() => {
    if (conn.state !== "ok" || !isLive) return
    let active = true
    const t = setInterval(() => {
      fetchOpStatus()
        .then((op) => {
          if (!active) return
          if (op.session) setSession(op.session)
          setPolicy(op.running ? op.policy : null)
        })
        .catch(() => {})
    }, 1500)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, isLive])

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
    const deadline = Date.now() + 5 * 60_000
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
          toast.error("Update is taking longer than expected. Reload to retry.")
        }
        return
      }
      try {
        const u = await fetchUpdateStatus()
        if (!active) return
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
        // unreachable). Show "restarting" and keep watching for it to return.
        if (active) setUpdatePhase("restarting")
      }
    }, 2000)
    return () => {
      active = false
      clearInterval(t)
    }
  }, [conn.state, updating, update?.remoteVersion, toast])

  const spec = useMemo(() => commands.find((c) => c.id === opId) ?? null, [commands, opId])

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
    // we proceed best-effort after the deadline.
    const deadline = Date.now() + 30_000
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
      const isSimSelected = isSimRun(meta, settings)
      const mantisSelected = mantisMode
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
          toast.error(`Can't verify cameras: ${detErr}`)
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
            )}. Reconnect, then Refresh (or Restart daemon) in the Cameras dialog.`
          )
          return
        }
      }

      // Send only the panel's per-run fields — the shared settings (and any
      // advanced overrides) are folded in server-side, and stale keys from the
      // old per-op localStorage must not shadow them.
      const runKeys = new Set(spec ? perRunFields(spec, meta).map((f) => f.key) : [])
      const args: Record<string, FormValue> = Object.fromEntries(
        Object.entries(settings).filter(([k]) => runKeys.has(k))
      )
      // Snapshot the shared source into the session request. New hosts also
      // return their fully merged args, but this keeps live hints/reset
      // controls tied to the actual run even against an older serve host if
      // the operator edits the saved source mid-run.
      if (mantisSelected) args.mantis_source = mantisSource
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
    if (requiresInstallerMigration(update.version)) {
      toast.error(
        "This server needs the one-time hosted-installer migration; run the command shown on the robot."
      )
      return
    }
    const generation = connectionGenerationRef.current
    setUpdateAbandoned(false)
    setStartingUpdate(true)
    setUpdatePhase("upgrading")
    try {
      await startUpdate()
      if (generation !== connectionGenerationRef.current) return
    } catch (e) {
      if (generation !== connectionGenerationRef.current) return
      toast.error(`Update failed to start: ${e}`)
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
  const connectedReleaseVersion = update?.version ?? hostInfo?.version ?? null
  const installerMigrationRequired =
    conn.state === "ok" &&
    connectedReleaseVersion !== null &&
    (hostInfo?.releaseInstall ?? update?.enabled ?? false) &&
    requiresInstallerMigration(connectedReleaseVersion)
  const mantisSource = String(settingsSnap?.values["teleop.mantis_source"] ?? "lighthouse")
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

  return (
    <div className="min-h-screen">
      <SiteNav current="control" />
      <main className="safe-x mx-auto flex max-w-5xl flex-col gap-6 py-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:py-8">
        {conn.state === "migration" && <InstallerMigrationBanner version={null} />}

        {installerMigrationRequired && (
          <InstallerMigrationBanner version={connectedReleaseVersion} />
        )}

        {update?.updateAvailable && !installerMigrationRequired && (
          <UpdateBanner
            update={update}
            updating={updating}
            phase={updatePhase}
            blocked={hostBusy}
            busyReason={updateBusyReason}
            onUpdate={handleUpdate}
          />
        )}

        {mismatch && !updating && !update?.updateAvailable && !installerMigrationRequired && (
          <VersionMismatchBanner mismatch={mismatch} />
        )}

        <ConnectionsBar
          conn={conn.state}
          host={serverHost}
          hostName={hostInfo?.hostname}
          version={update?.version ?? hostInfo?.version ?? null}
          onOpenSetup={() => setSetupOpen(true)}
          onHostDisconnect={hostDisconnectClick}
          opRunning={hostBusy}
          robot={robot}
          robotBusy={robotBusy}
          canProfiles={canProfiles}
          onRobotConnect={robotConnectClick}
          onRobotDisconnect={robotDisconnectClick}
        />

        {conn.state === "ok" && (
          <div ref={settingsRef} className="scroll-mt-18 sm:scroll-mt-20">
            <SettingsSection
              open={settingsOpen}
              onOpenChange={setSettingsOpen}
              tab={settingsTab}
              onTabChange={setSettingsTab}
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
                  ? `${runningOp ?? "An operation"} is running`
                  : activeCommandSession
                    ? `${activeCommandSession.command} is running`
                    : null
              }
              activeCommandSession={activeCommandSession}
              onCommandSessionChange={(next) => {
                if (connectionGenerationRef.current === renderedConnectionGeneration) {
                  setActiveCommandSession(next)
                }
              }}
            />
          </div>
        )}

        <OperationSelector
          operations={operations}
          selected={opId}
          runningOp={runningOp}
          onSelect={selectOp}
        />

        {isLive && !selectedLive && (
          <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
            <span className="font-mono text-amber-200">{runningOp}</span> is currently running. Stop
            it before starting another operation.
          </p>
        )}

        <OperationPanel
          key={renderedConnectionGeneration}
          meta={meta}
          spec={spec}
          settings={settings}
          mantisSource={mantisSource}
          onChange={setSetting}
          onReset={resetSetting}
          onResetAll={resetAll}
          onOpenSettings={() => openSettings(mantisMode ? "mantis" : "robot")}
          cameras={cameras}
          robot={robot}
          live={selectedLive}
          stopping={selectedStopping}
          busy={busy}
          session={selectedLive ? effectiveStatus : null}
          host={viewerHost}
          viewerPort={viewerPort}
          vrPort={hostInfo?.vrPort ?? 8000}
          startPhase={startPhase}
          hostBlocker={
            activeCommandSession
              ? `${activeCommandSession.command} setup/diagnostic is running`
              : isLive && !selectedLive
                ? `${runningOp ?? "Another operation"} is running`
                : null
          }
          connected={conn.state === "ok"}
          policy={selectedLive ? policy : null}
          onStart={handleStart}
          onStop={handleStop}
          onEpisode={handleEpisode}
        />

        <LogConsole lines={lines} />
      </main>

      <SetupDialog
        open={setupOpen}
        onClose={() => setSetupOpen(false)}
        host={serverHost}
        onChangeHost={updateServerHost}
        conn={conn}
        onConnect={() => loadServer(serverHost)}
      />
    </div>
  )
}

/** One column per operation on wide screens, spelled out so Tailwind emits the
 *  classes; a host offering more than five wraps rather than shrinking. */
const WIDE_COLUMNS: Record<number, string> = {
  1: "lg:grid-cols-1",
  2: "lg:grid-cols-2",
  3: "lg:grid-cols-3",
  4: "lg:grid-cols-4",
  5: "lg:grid-cols-5",
}

function OperationSelector({
  operations,
  selected,
  runningOp,
  onSelect,
}: {
  operations: OperationMeta[]
  selected: OperationId
  runningOp: OperationId | null
  onSelect: (op: OperationId) => void
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 gap-3 sm:grid-cols-3",
        WIDE_COLUMNS[Math.min(operations.length, 5)]
      )}
    >
      {operations.map((op) => {
        const active = op.id === selected
        const running = op.id === runningOp
        return (
          <button
            key={op.id}
            type="button"
            onClick={() => onSelect(op.id)}
            className={cn(
              "rounded-xl border p-3 text-left transition-all",
              active
                ? "border-[#eff483]/40 bg-[#eff483]/10"
                : "border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.05]"
            )}
          >
            <div className="flex items-center gap-2">
              <span className={cn("text-sm font-medium", !active && "text-white/85")}>
                {op.label}
              </span>
              {running && <span className="size-2 animate-pulse rounded-full bg-emerald-400" />}
            </div>
          </button>
        )
      })}
    </div>
  )
}
