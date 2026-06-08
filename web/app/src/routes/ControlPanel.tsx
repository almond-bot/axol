import { useCallback, useEffect, useMemo, useState } from "react"
import { cn } from "@/lib/utils"
import {
  OPERATIONS,
  exportOpSettings,
  fetchCommands,
  fetchInfo,
  fetchOpStatus,
  fetchRobotStatus,
  fetchZedStatus,
  loadOpSettings,
  operationMeta,
  parseImportedSettings,
  robotConnect,
  robotDisconnect,
  saveOpSettings,
  sendEpisodeCommand,
  setServerBase,
  startOperation,
  stopOperation,
  useSessionLogs,
  zedDisconnect,
  type CommandSpec,
  type FormValue,
  type OperationId,
  type RobotStatus,
  type ServerInfo,
  type SessionInfo,
  type ZedLinkStatus,
  type ZedSpec,
} from "@/lib/supervisor"
import { ConnectionsBar } from "@/components/connections-bar"
import { OperationPanel } from "@/components/operation-panel"
import { LogConsole } from "@/components/log-console"
import { SetupDialog, type ConnState } from "@/components/setup-dialog"
import { ZedConnectDialog } from "@/components/zed-connect-dialog"
import { SudoDialog } from "@/components/sudo-dialog"
import { SiteNav } from "@/components/site-nav"

type OpSettings = Record<OperationId, Record<string, FormValue>>

const DEFAULT_ZED: ZedSpec = {
  enabled: false,
  boxUrl: "",
  cameras: { overhead: "", left_arm: "", right_arm: "" },
}

function loadZed(): ZedSpec {
  try {
    const raw = localStorage.getItem("axolZedSpec")
    if (raw) return { ...DEFAULT_ZED, ...JSON.parse(raw) }
  } catch {
    // ignore malformed storage
  }
  return DEFAULT_ZED
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
  const [viewerPort, setViewerPort] = useState(8080)

  const [robot, setRobot] = useState<RobotStatus | null>(null)
  const [robotBusy, setRobotBusy] = useState(false)
  const [sudoOpen, setSudoOpen] = useState(false)
  const [sudoError, setSudoError] = useState<string | null>(null)
  const [zedLink, setZedLink] = useState<ZedLinkStatus | null>(null)
  const [zedSettings, setZedSettings] = useState<ZedSpec>(() => loadZed())

  const [selectedOp, setSelectedOp] = useState<OperationId>(
    () => (localStorage.getItem("axolOp") as OperationId) || "teleop"
  )
  const [settingsByOp, setSettingsByOp] = useState<OpSettings>(() => loadAllOpSettings())

  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)
  const [zedDialogOpen, setZedDialogOpen] = useState(false)

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
    fetchRobotStatus()
      .then(setRobot)
      .catch(() => {})
    fetchZedStatus()
      .then(setZedLink)
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

  // Poll the detached connections while online.
  useEffect(() => {
    if (conn.state !== "ok") return
    const t = setInterval(() => {
      fetchRobotStatus()
        .then(setRobot)
        .catch(() => {})
      // Keep the ZED link (and its PTP clock-sync state) fresh so the badge
      // settles from "syncing" to "locked" after connecting the box.
      fetchZedStatus()
        .then(setZedLink)
        .catch(() => {})
    }, 2000)
    return () => clearInterval(t)
  }, [conn.state])

  function updateServerHost(value: string) {
    setServerHost(value)
    if (value.trim()) localStorage.setItem("axolServerHost", value.trim())
    else localStorage.removeItem("axolServerHost")
  }

  function patchZed(patch: Partial<ZedSpec>) {
    setZedSettings((prev) => {
      const next = { ...prev, ...patch }
      try {
        localStorage.setItem("axolZedSpec", JSON.stringify(next))
      } catch {
        // ignore storage failures
      }
      return next
    })
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
  async function robotConnectClick(password?: string) {
    setRobotBusy(true)
    try {
      const next = await robotConnect(password)
      setRobot(next)
      if (next.needsSudo) {
        // CAN is down and passwordless sudo isn't available — prompt for it.
        setSudoOpen(true)
        setSudoError(null)
      } else if (password && next.state === "error") {
        // A password was supplied but bring-up still failed (e.g. wrong pw).
        setSudoOpen(true)
        setSudoError(next.error ?? "CAN bring-up failed.")
      } else {
        setSudoOpen(false)
        setSudoError(null)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  async function robotDisconnectClick() {
    setRobotBusy(true)
    try {
      setRobot(await robotDisconnect())
    } catch (e) {
      setError(String(e))
    } finally {
      setRobotBusy(false)
    }
  }

  async function zedDisconnectClick() {
    try {
      setZedLink(await zedDisconnect())
    } catch (e) {
      setError(String(e))
    }
  }

  // -- operation lifecycle --
  const effectiveStatus = status ?? session
  const isLive = effectiveStatus?.status === "running" || effectiveStatus?.status === "starting"
  const runningOp = isLive ? (effectiveStatus?.command as OperationId) : null
  const selectedLive = isLive && runningOp === selectedOp

  const meta = operationMeta(selectedOp)
  const spec = useMemo(
    () => commands.find((c) => c.id === selectedOp) ?? null,
    [commands, selectedOp]
  )

  function buildZedSpec(): ZedSpec {
    return {
      ...zedSettings,
      enabled: true,
      boxUrl: zedLink?.boxUrl ?? zedSettings.boxUrl,
    }
  }

  async function handleStart() {
    setBusy(true)
    setError(null)
    try {
      const zed = meta.requiresZed ? buildZedSpec() : undefined
      const result = await startOperation(selectedOp, settings, zed)
      setSession(result)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    setBusy(true)
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

  const viewerHost = serverHost || hostInfo?.lanIp || ""

  return (
    <div className="min-h-screen">
      <SiteNav current="control" />
      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-8">
        <ConnectionsBar
          conn={conn.state}
          host={serverHost}
          onOpenSetup={() => setSetupOpen(true)}
          robot={robot}
          robotBusy={robotBusy}
          onRobotConnect={() => robotConnectClick()}
          onRobotDisconnect={robotDisconnectClick}
          zed={zedLink}
          onZedConnect={() => setZedDialogOpen(true)}
          onZedDisconnect={zedDisconnectClick}
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
          zedSettings={zedSettings}
          onZedChange={patchZed}
          zedLink={zedLink}
          robot={robot}
          live={selectedLive}
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
      <ZedConnectDialog
        open={zedDialogOpen}
        onClose={() => setZedDialogOpen(false)}
        initial={zedLink}
        onConnected={setZedLink}
      />
      <SudoDialog
        open={sudoOpen}
        busy={robotBusy}
        error={sudoError}
        onClose={() => {
          setSudoOpen(false)
          setSudoError(null)
        }}
        onSubmit={(password) => robotConnectClick(password)}
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
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
              {op.requiresZed ? "Axol + ZED" : op.simCapable ? "Axol or Sim" : "Axol"}
            </span>
          </button>
        )
      })}
    </div>
  )
}
