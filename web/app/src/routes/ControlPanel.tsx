import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, Square, Terminal, ExternalLink, AlertTriangle } from "lucide-react"
import {
  computeArgs,
  fetchCommands,
  fetchInfo,
  fetchSessions,
  flattenFields,
  missingRequired,
  runCommand,
  setServerBase,
  stopSession,
  useSessionLogs,
  zedMissing,
  type CommandSpec,
  type FormValue,
  type ServerInfo,
  type SessionInfo,
  type ZedSpec,
} from "@/lib/supervisor"
import { ConfigForm } from "@/components/config-form"
import { ZedOrchestration } from "@/components/zed-orchestration"
import { CommandCatalog } from "@/components/command-catalog"
import { SetupDialog, ConnectionPill, type ConnState } from "@/components/setup-dialog"
import { SiteNav } from "@/components/site-nav"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type Overrides = Record<string, FormValue>

const ZED_COMMANDS = new Set(["collect-data", "run-policy"])

const DEFAULT_ZED: ZedSpec = {
  enabled: false,
  boxUrl: "",
  topology: "direct",
  hostIface: "",
  boxIface: "",
  zedHost: "192.168.10.1",
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

export default function ControlPanel() {
  const [commands, setCommands] = useState<CommandSpec[]>([])
  const [selectedId, setSelectedId] = useState<string>("")
  const [overridesByCmd, setOverridesByCmd] = useState<Record<string, Overrides>>({})
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [serverHost, setServerHost] = useState<string>(
    () => localStorage.getItem("axolServerHost") ?? ""
  )
  const [viewerPort, setViewerPort] = useState(8080)
  const [hostInfo, setHostInfo] = useState<ServerInfo | null>(null)
  const [zed, setZed] = useState<ZedSpec>(() => loadZed())
  const [conn, setConn] = useState<{ state: ConnState; message?: string }>({ state: "loading" })
  const [setupOpen, setSetupOpen] = useState(false)

  const { lines, status } = useSessionLogs(session?.id ?? null)

  const loadServer = useCallback(async (host: string) => {
    setServerBase(host)
    setConn({ state: "loading" })
    setError(null)
    try {
      const cmds = await fetchCommands()
      setCommands(cmds)
      setSelectedId((prev) => prev || cmds.find((c) => c.available)?.id || cmds[0]?.id || "")
      setConn({ state: "ok" })
      setSetupOpen(false)
    } catch (e) {
      setCommands([])
      setConn({ state: "err", message: String(e) })
      return
    }
    // Best-effort extras once the connection is known good.
    fetchSessions()
      .then((sessions) => {
        const live = sessions.find((s) => s.status === "running" || s.status === "starting")
        if (live) {
          setSelectedId(live.command)
          setSession(live)
        }
      })
      .catch(() => {})
    fetchInfo()
      .then((info) => {
        setViewerPort(info.viewerPort)
        setHostInfo(info)
        if (info.ethIface) {
          setZed((prev) => (prev.hostIface ? prev : { ...prev, hostIface: info.ethIface! }))
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    loadServer(serverHost)
    // Only on mount — reconnects are explicit via the setup dialog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function updateServerHost(value: string) {
    setServerHost(value)
    if (value.trim()) localStorage.setItem("axolServerHost", value.trim())
    else localStorage.removeItem("axolServerHost")
  }

  function patchZed(patch: Partial<ZedSpec>) {
    setZed((prev) => {
      const next = { ...prev, ...patch }
      try {
        localStorage.setItem("axolZedSpec", JSON.stringify(next))
      } catch {
        // ignore storage failures
      }
      return next
    })
  }

  const selected = useMemo(
    () => commands.find((c) => c.id === selectedId) ?? null,
    [commands, selectedId]
  )
  const overrides = overridesByCmd[selectedId] ?? {}

  const effectiveStatus = status ?? session
  const isLive = effectiveStatus?.status === "running" || effectiveStatus?.status === "starting"
  // A live session belongs to whichever command spawned it, not the one being viewed.
  const liveHere = isLive && effectiveStatus?.command === selected?.id

  const isZedCmd = !!selected && ZED_COMMANDS.has(selected.id)
  const zedActive = isZedCmd && zed.enabled

  const fields = useMemo(() => (selected ? flattenFields(selected.schema) : []), [selected])
  const missing = useMemo(() => {
    const base = missingRequired(fields, overrides)
    return zedActive ? [...base, ...zedMissing(zed)] : base
  }, [fields, overrides, zedActive, zed])
  const editedCount = useMemo(
    () => Object.keys(computeArgs(fields, overrides)).length,
    [fields, overrides]
  )

  function setOverride(key: string, value: FormValue) {
    setOverridesByCmd((prev) => ({
      ...prev,
      [selectedId]: { ...prev[selectedId], [key]: value },
    }))
  }

  function resetOverride(key: string) {
    setOverridesByCmd((prev) => {
      const next = { ...(prev[selectedId] ?? {}) }
      delete next[key]
      return { ...prev, [selectedId]: next }
    })
  }

  function resetAll() {
    setOverridesByCmd((prev) => ({ ...prev, [selectedId]: {} }))
  }

  async function handleStart() {
    if (!selected) return
    if (missing.length > 0) {
      setError(`Fill required fields: ${missing.join(", ")}`)
      return
    }
    setBusy(true)
    setError(null)
    try {
      const result = await runCommand(
        selected.id,
        computeArgs(fields, overrides),
        zedActive ? zed : undefined
      )
      setSession(result)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    if (!session) return
    setBusy(true)
    try {
      const result = await stopSession(session.id)
      setSession(result)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const viewerHost = serverHost || hostInfo?.lanIp || ""

  return (
    <div className="min-h-screen">
      <SiteNav
        current="control"
        right={
          <ConnectionPill state={conn.state} host={serverHost} onClick={() => setSetupOpen(true)} />
        }
      />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
          <CommandCatalog
            commands={commands}
            selectedId={selectedId}
            disabled={isLive}
            connected={conn.state === "ok"}
            onSelect={setSelectedId}
            onOpenSetup={() => setSetupOpen(true)}
          />

          <div className="flex min-w-0 flex-col gap-6">
            {selected ? (
              <>
                <Card className="gap-0 p-0">
                  <div className="flex flex-col gap-4 border-b border-white/10 p-5 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <h2 className="font-heading text-lg font-semibold">{selected.label}</h2>
                        <StatusBadge session={liveHere ? effectiveStatus : null} />
                        {selected.simCapable && <Badge variant="neutral">sim</Badge>}
                      </div>
                      <div className="mt-0.5 font-mono text-xs text-white/40">
                        axol {selected.cli}
                      </div>
                      <p className="mt-2 max-w-prose text-sm text-white/55">
                        {selected.description}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {editedCount > 0 && !liveHere && selected.available && (
                        <button
                          type="button"
                          onClick={resetAll}
                          className="text-xs whitespace-nowrap text-white/40 hover:text-white/70"
                        >
                          Reset ({editedCount})
                        </button>
                      )}
                      {liveHere ? (
                        <Button variant="destructive" onClick={handleStop} disabled={busy}>
                          {busy ? <Loader2 className="animate-spin" /> : <Square />}
                          Stop
                        </Button>
                      ) : (
                        <Button
                          onClick={handleStart}
                          disabled={busy || !selected.available || missing.length > 0 || isLive}
                        >
                          {busy ? <Loader2 className="animate-spin" /> : <Play />}
                          Start
                        </Button>
                      )}
                    </div>
                  </div>

                  <CardContent className="gap-4 p-5">
                    {isLive && !liveHere && (
                      <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
                        <span className="font-mono text-amber-200">{effectiveStatus?.command}</span>{" "}
                        is currently running. Stop it before starting another command.
                      </p>
                    )}

                    {selected.available ? (
                      <ConfigForm
                        schema={selected.schema}
                        overrides={overrides}
                        disabled={isLive}
                        onChange={setOverride}
                        onReset={resetOverride}
                      />
                    ) : (
                      <UnavailableNotice cmd={selected} />
                    )}

                    {missing.length > 0 && !liveHere && selected.available && (
                      <p className="text-xs text-white/40">Required: {missing.join(", ")}</p>
                    )}
                    {error && <p className="text-sm text-red-400">{error}</p>}

                    <RunningHints
                      session={liveHere ? effectiveStatus : null}
                      overrides={overrides}
                      host={viewerHost}
                      viewerPort={viewerPort}
                    />
                  </CardContent>
                </Card>

                {isZedCmd && selected.available && (
                  <ZedOrchestration
                    spec={zed}
                    onChange={patchZed}
                    hostInfo={hostInfo}
                    disabled={isLive}
                  />
                )}

                <LogConsole lines={lines} />
              </>
            ) : (
              <EmptyState connected={conn.state === "ok"} onOpenSetup={() => setSetupOpen(true)} />
            )}
          </div>
        </div>
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

function EmptyState({ connected, onOpenSetup }: { connected: boolean; onOpenSetup: () => void }) {
  return (
    <Card>
      <CardContent className="items-center gap-3 py-16 text-center">
        <Terminal className="size-8 text-white/20" />
        <p className="text-sm text-white/45">
          {connected
            ? "Pick a command from the catalog to configure and run it."
            : "Connect to a machine running axol serve to get started."}
        </p>
        {!connected && (
          <Button variant="outline" size="sm" onClick={onOpenSetup}>
            Open setup
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

function RunningHints({
  session,
  overrides,
  host,
  viewerPort,
}: {
  session: SessionInfo | null
  overrides: Overrides
  host: string
  viewerPort: number
}) {
  if (!session || session.status !== "running") return null
  const simRunning = session.args?.sim === true || overrides.sim === true
  const viewerUrl = host ? `http://${host}:${viewerPort}` : ""

  return (
    <div className="flex flex-col gap-3">
      {simRunning && viewerUrl && (
        <a
          href={viewerUrl}
          target="_blank"
          rel="noreferrer"
          className={cn(buttonVariants({ variant: "outline", size: "sm" }), "w-fit")}
        >
          <ExternalLink />
          Open 3D viewer
        </a>
      )}
      {session.command === "teleop" && (
        <p className="rounded-lg border border-white/10 bg-white/[0.02] p-3 text-xs leading-relaxed text-white/45">
          Put on the headset, open <span className="text-white/70">axol.almond.bot</span>, and
          connect to <span className="font-mono text-[#eff483]">{host || "this machine"}</span>.
        </p>
      )}
    </div>
  )
}

function UnavailableNotice({ cmd }: { cmd: CommandSpec }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-4 text-sm">
      <div className="flex items-center gap-2 font-medium text-amber-300/90">
        <AlertTriangle className="size-4" />
        Not available on this server
      </div>
      <p className="text-white/55">
        This command needs dependencies that aren&apos;t installed on the connected machine (e.g.
        the ZED SDK / <span className="font-mono">lerobot</span> extra, or robot hardware).
      </p>
      {cmd.error && (
        <code className="rounded bg-black/30 p-2 text-xs break-words text-white/45">
          {cmd.error}
        </code>
      )}
    </div>
  )
}

function StatusBadge({ session }: { session: SessionInfo | null }) {
  if (!session) return null
  switch (session.status) {
    case "starting":
      return <Badge variant="warning">Starting</Badge>
    case "running":
      return <Badge variant="success">Running</Badge>
    case "error":
      return <Badge variant="destructive">Error</Badge>
    case "exited":
      return <Badge variant={session.exitCode === 0 ? "neutral" : "destructive"}>Exited</Badge>
    default:
      return <Badge variant="neutral">{session.status}</Badge>
  }
}

function LogConsole({ lines }: { lines: string[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  return (
    <Card className="min-h-0 flex-1 gap-3 p-0">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
        <Terminal className="size-4 text-white/40" />
        <span className="font-heading text-sm font-semibold">Logs</span>
      </div>
      <div
        ref={scrollRef}
        className="max-h-[60vh] min-h-[280px] overflow-auto px-4 pb-4 font-mono text-xs leading-relaxed"
      >
        {lines.length === 0 ? (
          <p className="text-white/30">No output yet.</p>
        ) : (
          lines.map((line, i) => (
            <div
              key={i}
              className={cn(
                "break-words whitespace-pre-wrap",
                line.startsWith("[serve]") ? "text-[#eff483]/70" : "text-white/70"
              )}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
