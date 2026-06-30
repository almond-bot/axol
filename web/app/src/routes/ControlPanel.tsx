import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import {
  OPERATIONS,
  cameraCount,
  exportOpSettings,
  fetchCommands,
  fetchInfo,
  fetchOpStatus,
  fetchRobotStatus,
  fetchUpdateStatus,
  fetchUsbStatus,
  loadOpSettings,
  operationMeta,
  parseImportedSettings,
  robotConnect,
  robotDisconnect,
  saveOpSettings,
  sendEpisodeCommand,
  setServerBase,
  startOperation,
  startUpdate,
  stopOperation,
  usbConnect,
  useSessionLogs,
  type CameraSpec,
  type CommandSpec,
  type FormValue,
  type OperationId,
  type RobotStatus,
  type ServerInfo,
  type SessionInfo,
  type UpdateStatus,
  type UsbStatus,
} from "@/lib/supervisor"
import { UpdateBanner } from "@/components/update-banner"
import { ConnectionsBar } from "@/components/connections-bar"
import { OperationPanel } from "@/components/operation-panel"
import { LogConsole } from "@/components/log-console"
import { SetupDialog, type ConnState } from "@/components/setup-dialog"
import { CamerasDialog } from "@/components/cameras-dialog"
import { SiteNav } from "@/components/site-nav"

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

function loadAllOpSettings(): OpSettings {
  return OPERATIONS.reduce((acc, op) => {
    acc[op.id] = loadOpSettings(op.id)
    return acc
  }, {} as OpSettings)
}

export default function ControlPanel() {
  const [commands, setCommands] = useState<CommandSpec[]>([])
  const [conn, setConn] = useState<{ state: ConnState; message?: string }>({ state: "loading" })
  const [serverHost, setServerHost] = useState<string>(
    () => localStorage.getItem("axolServerHost") ?? ""
  )
  const [hostInfo, setHostInfo] = useState<ServerInfo | null>(null)
  const [viewerPort, setViewerPort] = useState(8002)
  const [update, setUpdate] = useState<UpdateStatus | null>(null)
  // Drives the banner's "Updating…" state while the server upgrades + restarts.
  const [updating, setUpdating] = useState(false)

  const [robot, setRobot] = useState<RobotStatus | null>(null)
  const [robotBusy, setRobotBusy] = useState(false)
  const [usb, setUsb] = useState<UsbStatus | null>(null)
  const [usbBusy, setUsbBusy] = useState(false)
  const [cameras, setCameras] = useState<CameraSpec>(() => loadCameras())

  const [selectedOp, setSelectedOp] = useState<OperationId>(
    () => (localStorage.getItem("axolOp") as OperationId) || "teleop"
  )
  const [settingsByOp, setSettingsByOp] = useState<OpSettings>(() => loadAllOpSettings())

  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)
  const [camerasDialogOpen, setCamerasDialogOpen] = useState(false)

  const { lines, status } = useSessionLogs(session?.id ?? null)

  const loadServer = useCallback(async (host: string) => {
    setServerBase(host)
    setConn({ state: "loading" })
    setError(null)
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
      })
      .catch(() => {})
  }, [])

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
  useEffect(() => {
    if (conn.state !== "ok" || updating) return
    let active = true
    const poll = () => {
      fetchUpdateStatus()
        .then((u) => {
          if (active) setUpdate(u)
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
    setError(null)
    try {
      await stopRunningOp()
    } catch (e) {
      setError(String(e))
      setBusy(false)
      return
    }
    setBusy(false)
    setConn({ state: "idle" })
    setCommands([])
    setRobot(null)
    setSession(null)
    setUpdate(null)
    setUpdating(false)
    autoRobotRef.current = false
  }

  function updateServerHost(value: string) {
    setServerHost(value)
    if (value.trim()) localStorage.setItem("axolServerHost", value.trim())
    else localStorage.removeItem("axolServerHost")
  }

  function saveCameras(spec: CameraSpec) {
    setCameras(spec)
    try {
      localStorage.setItem("axolCameraSpec", JSON.stringify(spec))
    } catch {
      // ignore storage failures
    }
  }

  function selectOp(op: OperationId) {
    setSelectedOp(op)
    localStorage.setItem("axolOp", op)
    setError(null)
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

  function importSettings(text: string) {
    try {
      updateSettings(selectedOp, parseImportedSettings(text))
      setError(null)
    } catch (e) {
      setError(`Import failed: ${e}`)
    }
  }

  // -- robot connection --
  async function robotConnectClick() {
    setRobotBusy(true)
    try {
      setRobot(await robotConnect())
    } catch (e) {
      setError(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  async function robotDisconnectClick() {
    setRobotBusy(true)
    setError(null)
    try {
      // Kill any running task and wait for it to exit before releasing the
      // robot connection out from under it, then disconnect.
      await stopRunningOp()
      setRobot(await robotDisconnect())
    } catch (e) {
      setError(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  // -- quest over usb (adb reverse pose tunnel) --
  async function usbConnectClick() {
    setUsbBusy(true)
    setError(null)
    try {
      // Runs `adb reverse`; the first touch also pops the USB-debugging
      // authorization prompt on the headset.
      setUsb(await usbConnect())
    } catch (e) {
      setError(String(e))
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
          if (active && op.session) setSession(op.session)
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
    setError(null)
    try {
      // Send the camera spec whenever any serial is assigned — collect-data /
      // run-policy need at least one, while teleop streams whichever are set to
      // the headset (and runs fine with none in sim).
      const spec = meta.requiresCameras || cameraCount(cameras) > 0 ? cameras : undefined
      const result = await startOperation(selectedOp, settings, spec)
      setSession(result)
    } catch (e) {
      setError(String(e))
    } finally {
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
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  function handleEpisode(command: string) {
    sendEpisodeCommand(command).catch((e) => setError(String(e)))
  }

  // Apply the available update, then hard-reload once the server is back on the
  // new commit. The server upgrades and exits (systemd relaunches it), so the
  // start request and the watch polls below tolerate it briefly going away.
  async function handleUpdate() {
    const target = update?.remoteCommit
    if (!target) return
    setError(null)
    setUpdating(true)
    try {
      await startUpdate()
    } catch (e) {
      setError(`Update failed to start: ${e}`)
      setUpdating(false)
      return
    }
    // The hosted UI is served by Vercel, so a hard reload also pulls the latest
    // front-end; reconnecting to the restarted backend re-runs loadServer.
    const reload = () => window.location.reload()
    const deadline = Date.now() + 5 * 60_000
    const watch = setInterval(async () => {
      if (Date.now() > deadline) {
        clearInterval(watch)
        setUpdating(false)
        setError("Update is taking longer than expected. Reload to retry.")
        return
      }
      try {
        const u = await fetchUpdateStatus()
        setUpdate(u)
        if (u.state === "error") {
          clearInterval(watch)
          setUpdating(false)
          setError(u.error ?? "Update failed.")
          return
        }
        // Server is back and now running the target commit — done.
        if (u.commit === target) {
          clearInterval(watch)
          reload()
        }
      } catch {
        // Server is mid-restart (or briefly unreachable); keep watching.
      }
    }, 2000)
  }

  const viewerHost = serverHost || hostInfo?.lanIp || ""

  return (
    <div className="min-h-screen">
      <SiteNav current="control" />
      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-8">
        {update?.updateAvailable && (
          <UpdateBanner
            update={update}
            updating={updating}
            blocked={updateBlocked}
            busyReason={updateBusyReason}
            onUpdate={handleUpdate}
          />
        )}

        <ConnectionsBar
          conn={conn.state}
          host={serverHost}
          hostName={hostInfo?.hostname}
          commit={update?.commit ?? hostInfo?.commit ?? null}
          onOpenSetup={() => setSetupOpen(true)}
          onHostDisconnect={hostDisconnectClick}
          robot={robot}
          robotBusy={robotBusy}
          onRobotConnect={() => robotConnectClick()}
          onRobotDisconnect={robotDisconnectClick}
          cameras={cameras}
          onConfigureCameras={() => setCamerasDialogOpen(true)}
          usb={usb}
          usbBusy={usbBusy}
          onUsbConnect={() => usbConnectClick()}
        />

        <OperationSelector selected={selectedOp} runningOp={runningOp} onSelect={selectOp} />

        {isLive && !selectedLive && (
          <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
            <span className="font-mono text-amber-200">{runningOp}</span> is currently running. Stop
            it before starting another operation.
          </p>
        )}
        {error && <p className="text-sm text-red-400">{error}</p>}

        <OperationPanel
          meta={meta}
          spec={spec}
          settings={settings}
          onChange={setSetting}
          onReset={resetSetting}
          onResetAll={resetAll}
          onExport={() => exportOpSettings(selectedOp, settings)}
          onImport={importSettings}
          cameras={cameras}
          robot={robot}
          live={selectedLive}
          stopping={selectedStopping}
          busy={busy}
          session={selectedLive ? effectiveStatus : null}
          host={viewerHost}
          viewerPort={viewerPort}
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
      <CamerasDialog
        open={camerasDialogOpen}
        onClose={() => setCamerasDialogOpen(false)}
        initial={cameras}
        onSave={saveCameras}
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
