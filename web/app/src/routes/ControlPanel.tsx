import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import {
  OPERATIONS,
  cameraCount,
  detectCameras,
  fetchCommands,
  fetchInfo,
  fetchOpStatus,
  fetchRobotStatus,
  fetchSettings,
  fetchUpdateStatus,
  fetchUsbStatus,
  loadOpSettings,
  missingCameraSerials,
  operationMeta,
  perRunFields,
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
  type CommandSpec,
  type FormValue,
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
import { UpdateBanner } from "@/components/update-banner"
import { VersionMismatchBanner } from "@/components/version-mismatch-banner"
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

function loadAllOpSettings(): OpSettings {
  return OPERATIONS.reduce((acc, op) => {
    acc[op.id] = loadOpSettings(op.id)
    return acc
  }, {} as OpSettings)
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
    () => (localStorage.getItem("axolOp") as OperationId) || "teleop"
  )
  const [settingsByOp, setSettingsByOp] = useState<OpSettings>(() => loadAllOpSettings())

  const [session, setSession] = useState<SessionInfo | null>(null)
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
  const refreshCameras = useCallback(async () => {
    setCameraDetecting(true)
    try {
      const result = await detectCameras()
      setCameraDevices(result.devices)
      setCameraDetectError(result.error)
    } catch (e) {
      setCameraDevices(null)
      setCameraDetectError(String(e).replace(/^Error:\s*/, ""))
    } finally {
      setCameraDetecting(false)
    }
  }, [])

  // Pull the shared settings from the serve host. A host whose stored camera
  // spec is empty gets this browser's legacy localStorage spec migrated up
  // once, so nobody has to re-enter serials after updating.
  const loadSettings = useCallback(async () => {
    try {
      const snap = await fetchSettings()
      setSettingsError(null)
      if (snap.cameras) {
        setCameras(snap.cameras)
        persistLocalCameras(snap.cameras)
        setSettingsSnap(snap)
      } else {
        const local = loadCameras()
        if (cameraCount(local) > 0) {
          try {
            await saveSettings({ cameras: local, camerasSet: true })
            setSettingsSnap({ ...snap, cameras: local })
          } catch {
            setSettingsSnap(snap)
          }
        } else {
          setSettingsSnap(snap)
        }
      }
    } catch (e) {
      // Old serve host without /api/settings: keep the localStorage camera
      // flow; the settings dialog explains the needed update.
      setSettingsSnap(null)
      setSettingsError(String(e).replace(/^Error:\s*/, ""))
    }
  }, [])

  const loadServer = useCallback(
    async (host: string) => {
      setServerBase(host)
      setConn({ state: "loading" })
      try {
        const cmds = await fetchCommands()
        setCommands(cmds)
        setConn({ state: "ok" })
        setSetupOpen(false)
      } catch (e) {
        setCommands([])
        setConn({ state: "err", message: String(e) })
        return
      }
      refreshCameras()
      loadSettings()
      fetchInfo()
        .then((info) => {
          setViewerPort(info.viewerPort)
          setHostInfo(info)
        })
        .catch(() => {})
      // Force a synchronous remote check on connect/page load so the banner
      // reflects reality immediately; the steady-state poll below stays cheap.
      fetchUpdateStatus(true)
        .then(setUpdate)
        .catch(() => {})
      fetchRobotStatus()
        .then(setRobot)
        .catch(() => {})
      fetchOpStatus()
        .then((op) => {
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
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadServer(serverHost)
    // Only on mount — reconnects are explicit via the setup dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Poll the robot connection + Quest-USB status while online.
  useEffect(() => {
    if (conn.state !== "ok") return
    // Guard against in-flight polls landing after a disconnect (which flips
    // conn.state and tears this effect down): a late response must not
    // repopulate a tile while the host tile shows disconnected.
    let active = true
    const poll = () => {
      fetchRobotStatus()
        .then((r) => {
          if (active) setRobot(r)
        })
        .catch(() => {})
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

  // Auto-connect Axol once after the host comes online, if it's sitting idle.
  // The ref makes it fire at most once per host session, so a manual robot
  // disconnect afterwards isn't immediately undone.
  const autoRobotRef = useRef(false)
  useEffect(() => {
    if (conn.state !== "ok") {
      autoRobotRef.current = false
      return
    }
    if (autoRobotRef.current || !robot) return
    autoRobotRef.current = true
    if (robot.state === "disconnected" && !robotBusy) {
      robotConnectClick()
    }
    // robotConnectClick is stable enough (only uses state setters / fetch).
  }, [conn.state, robot, robotBusy])

  // Auto-establish the Quest-over-USB tunnel as soon as an authorized headset
  // appears. The latch clears once the tunnel is up (so a later drop retries
  // once) or when the headset goes away, while preventing a per-poll retry loop
  // if `adb reverse` can't establish it.
  const autoUsbRef = useRef(false)
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
    // usbConnectClick is stable (only uses state setters / fetch).
  }, [conn.state, usb, usbBusy])

  async function hostDisconnectClick() {
    // Kill any running task and wait for it to exit before dropping the host,
    // so disconnecting never leaves an orphaned op running server-side. Only
    // then tear down the (client-side) host connection.
    setBusy(true)
    try {
      await stopRunningOp()
    } catch (e) {
      toast.error(String(e))
      setBusy(false)
      return
    }
    setBusy(false)
    setConn({ state: "idle" })
    setCommands([])
    setRobot(null)
    setSession(null)
    setPolicy(null)
    setCameraDevices(null)
    setCameraDetectError(null)
    setSettingsSnap(null)
    setSettingsError(null)
    setUpdate(null)
    setStartingUpdate(false)
    setUpdateAbandoned(false)
    setUpdatePhase(null)
    autoRobotRef.current = false
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
  async function handleSettingsSave(patch: SettingsPatch) {
    if (patch.cameras) {
      setCameras(patch.cameras)
      persistLocalCameras(patch.cameras)
    }
    if (settingsError) {
      // Old host: only the cameras section is editable, and it saved locally.
      return
    }
    const snap = await saveSettings(patch)
    setSettingsSnap((prev) => ({
      ...snap,
      schema: prev?.schema ?? snap.schema,
      advancedSchema: prev?.advancedSchema ?? snap.advancedSchema,
    }))
    if (snap.cameras) setCameras(snap.cameras)
  }

  function selectOp(op: OperationId) {
    setSelectedOp(op)
    localStorage.setItem("axolOp", op)
  }

  // -- per-operation settings --
  const settings = settingsByOp[selectedOp] ?? {}

  const updateSettings = useCallback((op: OperationId, next: Record<string, FormValue>) => {
    setSettingsByOp((prev) => ({ ...prev, [op]: next }))
    saveOpSettings(op, next)
  }, [])

  function setSetting(key: string, value: FormValue) {
    updateSettings(selectedOp, { ...settings, [key]: value })
  }

  function resetSetting(key: string) {
    const next = { ...settings }
    delete next[key]
    updateSettings(selectedOp, next)
  }

  function resetAll() {
    updateSettings(selectedOp, {})
  }

  // -- robot connection --
  async function robotConnectClick() {
    setRobotBusy(true)
    try {
      setRobot(await robotConnect())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  async function robotDisconnectClick() {
    setRobotBusy(true)
    try {
      // Kill any running task and wait for it to exit before releasing the
      // robot connection out from under it, then disconnect.
      await stopRunningOp()
      setRobot(await robotDisconnect())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  // -- quest over usb (adb reverse pose tunnel) --
  async function usbConnectClick() {
    setUsbBusy(true)
    try {
      // Runs `adb reverse`; the first touch also pops the USB-debugging
      // authorization prompt on the headset.
      setUsb(await usbConnect())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setUsbBusy(false)
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
  const selectedLive = isLive && runningOp === selectedOp
  const selectedStopping = isStopping && runningOp === selectedOp

  // Whether an update is currently unsafe to apply. `isLive` is the immediate,
  // local signal (reacts the instant an op starts/stops) so the banner blocks
  // without waiting for the slow status poll; the server's `update.idle` is the
  // backstop for any other non-idle reason (and the server guards the request
  // regardless). Mirrors the server's _is_idle: only a running operation blocks
  // a restart (a connected robot is fine).
  const updateBlocked = isLive || !(update?.idle ?? true)
  // Reason shown in the banner; capitalized clause, no trailing period.
  const updateBusyReason = isLive ? "Stop the running operation" : "The server is busy"

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
    fetchUpdateStatus()
      .then(setUpdate)
      .catch(() => {})
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

  const meta = operationMeta(selectedOp)
  const spec = useMemo(
    () => commands.find((c) => c.id === selectedOp) ?? null,
    [commands, selectedOp]
  )

  // Stop the running task (if any) and wait for it to actually exit before
  // returning, so a disconnect never tears the host/robot link down mid-cleanup.
  // The server-side stop now returns immediately with "stopping" (it force-kills
  // a stuck op in the background), so we poll op status until the op is truly
  // gone rather than relying on the stop response to block.
  async function stopRunningOp() {
    if (!isLive) return
    setSession(await stopOperation())
    // Bounded so an unkillable op (abandoned server-side) can't wedge the UI;
    // we proceed best-effort after the deadline.
    const deadline = Date.now() + 30_000
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 500))
      try {
        const op = await fetchOpStatus()
        if (op.session) setSession(op.session)
        if (!op.running) return
      } catch {
        return // host unreachable — nothing left to wait on
      }
    }
  }

  async function handleStart() {
    setBusy(true)
    try {
      // Only ops that actually require cameras (collect-data / run-policy) are
      // gated on them. Teleop streams whatever cameras are configured but must
      // never be blocked by camera detection, and sim never touches hardware.
      const isSimSelected = selectedOp === "teleop" && Boolean(settings.sim)
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
          setCameraDevices(detect.devices)
          setCameraDetectError(detect.error)
          devices = detect.devices
          detErr = detect.error
        }
        if (detErr) {
          toast.error(`Can't verify cameras: ${detErr}`)
          return
        }
        const missing = missingCameraSerials(cameras, devices ?? [])
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
      const args = Object.fromEntries(Object.entries(settings).filter(([k]) => runKeys.has(k)))
      // Send the camera spec whenever any serial is assigned — collect-data /
      // run-policy need at least one, while teleop streams whichever are set to
      // the headset (and runs fine with none in sim). Newer hosts also hold the
      // spec in their settings store; sending it stays compatible with old ones.
      const camSpec = meta.requiresCameras || cameraCount(cameras) > 0 ? cameras : undefined
      const result = await startOperation(selectedOp, args, camSpec)
      setSession(result)
      // Fresh run — clear any stale phase; the live poll repopulates it.
      setPolicy(null)
    } catch (e) {
      toast.error(String(e))
    } finally {
      setStartPhase(null)
      setBusy(false)
    }
  }

  async function handleStop() {
    setBusy(true)
    // Reflect "Stopping…" immediately — the server stop returns right away and
    // teardown runs in the background, so don't wait for the response/next poll
    // to flip the button.
    setSession((s) => (s ? { ...s, status: "stopping" } : s))
    try {
      setSession(await stopOperation())
    } catch (e) {
      toast.error(String(e))
    } finally {
      setBusy(false)
    }
  }

  function handleEpisode(command: string) {
    sendEpisodeCommand(command).catch((e) => toast.error(String(e)))
  }

  // Kick off the available update. The server upgrades and exits (systemd
  // relaunches it); the update-watcher effect — which runs on any computer while
  // the server reports an update in flight — then advances the phase and
  // hard-reloads once the backend is back on the new release.
  async function handleUpdate() {
    if (!update?.remoteVersion) return
    setUpdateAbandoned(false)
    setStartingUpdate(true)
    setUpdatePhase("upgrading")
    try {
      await startUpdate()
    } catch (e) {
      toast.error(`Update failed to start: ${e}`)
      setStartingUpdate(false)
      setUpdatePhase(null)
      return
    }
    // Pull the now-"updating" status so `updating` derives true and the watcher
    // takes over; then drop the optimistic flag (server state carries it now).
    fetchUpdateStatus()
      .then((u) => {
        setUpdate(u)
        setStartingUpdate(false)
      })
      .catch(() => {})
  }

  const viewerHost = serverHost || hostInfo?.lanIp || ""

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
      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-8">
        {update?.updateAvailable && (
          <UpdateBanner
            update={update}
            updating={updating}
            phase={updatePhase}
            blocked={updateBlocked}
            busyReason={updateBusyReason}
            onUpdate={handleUpdate}
          />
        )}

        {mismatch && !updating && !update?.updateAvailable && (
          <VersionMismatchBanner mismatch={mismatch} />
        )}

        <ConnectionsBar
          conn={conn.state}
          host={serverHost}
          hostName={hostInfo?.hostname}
          version={update?.version ?? hostInfo?.version ?? null}
          onOpenSetup={() => setSetupOpen(true)}
          onHostDisconnect={hostDisconnectClick}
          robot={robot}
          robotBusy={robotBusy}
          onRobotConnect={() => robotConnectClick()}
          onRobotDisconnect={robotDisconnectClick}
        />

        {conn.state === "ok" && (
          <div ref={settingsRef} className="scroll-mt-4">
            <SettingsSection
              open={settingsOpen}
              onOpenChange={setSettingsOpen}
              tab={settingsTab}
              onTabChange={setSettingsTab}
              snapshot={settingsSnap}
              supportError={settingsError}
              cameras={cameras}
              onSave={handleSettingsSave}
              devices={cameraDevices}
              detecting={cameraDetecting}
              onRefresh={refreshCameras}
              usb={usb}
              usbBusy={usbBusy}
              onUsbConnect={() => usbConnectClick()}
            />
          </div>
        )}

        <OperationSelector selected={selectedOp} runningOp={runningOp} onSelect={selectOp} />

        {isLive && !selectedLive && (
          <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
            <span className="font-mono text-amber-200">{runningOp}</span> is currently running. Stop
            it before starting another operation.
          </p>
        )}

        <OperationPanel
          meta={meta}
          spec={spec}
          settings={settings}
          onChange={setSetting}
          onReset={resetSetting}
          onResetAll={resetAll}
          onOpenSettings={() => openSettings(settingsTab === "cameras" ? "robot" : settingsTab)}
          cameras={cameras}
          robot={robot}
          live={selectedLive}
          stopping={selectedStopping}
          busy={busy}
          session={selectedLive ? effectiveStatus : null}
          host={viewerHost}
          viewerPort={viewerPort}
          startPhase={startPhase}
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

function OperationSelector({
  selected,
  runningOp,
  onSelect,
}: {
  selected: OperationId
  runningOp: OperationId | null
  onSelect: (op: OperationId) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
      {OPERATIONS.map((op) => {
        const active = op.id === selected
        const running = op.id === runningOp
        return (
          <button
            key={op.id}
            type="button"
            onClick={() => onSelect(op.id)}
            className={cn(
              "flex flex-col gap-1 rounded-xl border p-3 text-left transition-all",
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
            <span className="text-xs text-white/40">
              {op.requiresCameras ? "Axol + Cameras" : op.simCapable ? "Axol or Sim" : "Axol"}
            </span>
          </button>
        )
      })}
    </div>
  )
}
