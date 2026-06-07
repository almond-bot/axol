import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Loader2,
  Play,
  Square,
  Terminal,
  Boxes,
  ExternalLink,
  AlertTriangle,
  Rocket,
  Copy,
  Check,
  ChevronRight,
  Plug,
} from "lucide-react"
import {
  computeArgs,
  fetchCommands,
  fetchInfo,
  fetchSessions,
  flattenFields,
  missingRequired,
  runCommand,
  serverHttpBase,
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
import { SiteNav } from "@/components/site-nav"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
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
  const [conn, setConn] = useState<{
    state: "loading" | "ok" | "err"
    message?: string
  }>({ state: "loading" })

  const { lines, status } = useSessionLogs(session?.id ?? null)

  const loadServer = useCallback(async (host: string) => {
    setServerBase(host)
    setConn({ state: "loading" })
    setError(null)
    try {
      const cmds = await fetchCommands()
      setCommands(cmds)
      setSelectedId((prev) => prev || cmds[0]?.id || "")
      setConn({ state: "ok" })
    } catch (e) {
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
        // Default the host's ZED-link interface to the detected one (once).
        if (info.ethIface) {
          setZed((prev) => (prev.hostIface ? prev : { ...prev, hostIface: info.ethIface! }))
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    loadServer(serverHost)
    // Only on mount — reconnects are explicit via the Server card.
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

  return (
    <div className="min-h-screen">
      <SiteNav current="control" />
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-6 py-8 lg:grid-cols-[400px_1fr]">
        <div className="flex flex-col gap-6">
          <Quickstart />

          <Card>
            <CardHeader>
              <CardTitle>Command</CardTitle>
              <CardDescription>Choose what to run on the robot.</CardDescription>
            </CardHeader>
            <CardContent className="gap-2">
              {commands.map((cmd) => (
                <CommandOption
                  key={cmd.id}
                  cmd={cmd}
                  selected={cmd.id === selectedId}
                  disabled={isLive}
                  onSelect={() => setSelectedId(cmd.id)}
                />
              ))}
            </CardContent>
          </Card>

          {isZedCmd && selected?.available && (
            <ZedOrchestration
              spec={zed}
              onChange={patchZed}
              hostInfo={hostInfo}
              disabled={isLive}
            />
          )}

          {selected && (
            <Card>
              <CardHeader className="flex-row items-start justify-between gap-3">
                <div className="min-w-0">
                  <CardTitle>Configure</CardTitle>
                  <CardDescription>{selected.description}</CardDescription>
                </div>
                {editedCount > 0 && !isLive && selected.available && (
                  <button
                    type="button"
                    onClick={resetAll}
                    className="shrink-0 text-xs whitespace-nowrap text-white/40 hover:text-white/70"
                  >
                    Reset all ({editedCount})
                  </button>
                )}
              </CardHeader>
              <CardContent>
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

                <div className="flex flex-col gap-2 pt-2">
                  {isLive ? (
                    <Button variant="destructive" onClick={handleStop} disabled={busy}>
                      {busy ? <Loader2 className="animate-spin" /> : <Square />}
                      Stop
                    </Button>
                  ) : (
                    <Button
                      onClick={handleStart}
                      disabled={busy || !selected.available || missing.length > 0}
                    >
                      {busy ? <Loader2 className="animate-spin" /> : <Play />}
                      Start
                    </Button>
                  )}
                  {missing.length > 0 && !isLive && selected.available && (
                    <p className="text-xs text-white/40">Required: {missing.join(", ")}</p>
                  )}
                  {error && <p className="text-sm text-red-400">{error}</p>}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-6">
          <ServerCard
            host={serverHost}
            onChange={updateServerHost}
            conn={conn}
            onConnect={() => loadServer(serverHost)}
          />
          <StatusCard
            command={selected}
            session={effectiveStatus}
            overrides={overrides}
            host={serverHost || hostInfo?.lanIp || ""}
            viewerPort={viewerPort}
          />
          <LogConsole lines={lines} />
        </div>
      </main>
    </div>
  )
}

const QUICKSTART: { label: string; hint?: string; cmd: string }[] = [
  {
    label: "1. Install uv",
    cmd: "curl -LsSf https://astral.sh/uv/install.sh | sh",
  },
  {
    label: "2. Install the Axol CLI globally",
    hint: "straight from GitHub",
    cmd: 'uv tool install --python 3.13 "almond-axol[sim] @ git+ssh://git@github.com/almond-bot/axol.git"',
  },
  {
    label: "3. Launch this control panel",
    cmd: "axol serve",
  },
]

function Quickstart() {
  const [open, setOpen] = useState(false)
  return (
    <Card className="gap-0 p-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-5 py-4 text-left"
      >
        <Rocket className="size-4 shrink-0 text-[#eff483]" />
        <div className="min-w-0 flex-1">
          <div className="font-heading text-sm font-semibold">Quickstart</div>
          <div className="text-xs text-white/40">Install uv, the CLI, and run the server</div>
        </div>
        <ChevronRight
          className={cn("size-4 shrink-0 text-white/40 transition-transform", open && "rotate-90")}
        />
      </button>
      {open && (
        <div className="flex flex-col gap-4 border-t border-white/10 p-5">
          {QUICKSTART.map((step) => (
            <div key={step.label} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm font-medium text-white/80">{step.label}</span>
                {step.hint && <span className="shrink-0 text-xs text-white/35">{step.hint}</span>}
              </div>
              <CommandLine cmd={step.cmd} />
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

function CommandLine({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(cmd)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard unavailable (e.g. non-secure context) — ignore.
    }
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/30 px-3 py-2">
      <code className="flex-1 overflow-x-auto font-mono text-xs whitespace-pre text-white/80">
        {cmd}
      </code>
      <button
        type="button"
        onClick={copy}
        title="Copy"
        className="shrink-0 text-white/40 transition-colors hover:text-white/80"
      >
        {copied ? <Check className="size-4 text-[#eff483]" /> : <Copy className="size-4" />}
      </button>
    </div>
  )
}

function CommandOption({
  cmd,
  selected,
  disabled,
  onSelect,
}: {
  cmd: CommandSpec
  selected: boolean
  disabled: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled && !selected}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all",
        selected
          ? "border-[#eff483]/40 bg-[#eff483]/10"
          : "border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.05]",
        disabled && !selected && "cursor-not-allowed opacity-40"
      )}
    >
      <Boxes className={cn("size-4 shrink-0", selected ? "text-[#eff483]" : "text-white/50")} />
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium">{cmd.label}</div>
        <div className="truncate font-mono text-xs text-white/40">axol {cmd.cli}</div>
      </div>
      {cmd.simCapable && <Badge variant="neutral">sim</Badge>}
      {!cmd.available && <Badge variant="warning">unavailable</Badge>}
    </button>
  )
}

function UnavailableNotice({ cmd }: { cmd: CommandSpec }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-4 text-sm">
      <div className="flex items-center gap-2 font-medium text-amber-300/90">
        <AlertTriangle className="size-4" />
        Not available in this environment
      </div>
      <p className="text-white/55">
        This command needs hardware-only dependencies that aren&apos;t installed here (e.g. the ZED
        SDK / <span className="font-mono">lerobot</span> extra).
      </p>
      {cmd.error && (
        <code className="rounded bg-black/30 p-2 text-xs break-words text-white/45">
          {cmd.error}
        </code>
      )}
    </div>
  )
}

function ServerCard({
  host,
  onChange,
  conn,
  onConnect,
}: {
  host: string
  onChange: (value: string) => void
  conn: { state: "loading" | "ok" | "err"; message?: string }
  onConnect: () => void
}) {
  const base = serverHttpBase(host)
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div className="min-w-0">
          <CardTitle>Server</CardTitle>
          <CardDescription>
            Address of the machine running <span className="font-mono">axol serve</span>. The
            control panel sends every command there.
          </CardDescription>
        </div>
        <ConnBadge state={conn.state} />
      </CardHeader>
      <CardContent className="gap-1.5">
        <Label htmlFor="server-host">Server address</Label>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            onConnect()
          }}
        >
          <Input
            id="server-host"
            value={host}
            onChange={(e) => onChange(e.target.value)}
            placeholder="192.168.1.42"
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
          />
          <Button
            type="submit"
            variant="outline"
            size="sm"
            className="shrink-0"
            disabled={conn.state === "loading"}
          >
            {conn.state === "loading" ? <Loader2 className="animate-spin" /> : <Plug />}
            Connect
          </Button>
        </form>
        {conn.state === "err" && (
          <div className="mt-1 flex flex-col gap-1 text-xs text-red-400">
            <span className="flex items-center gap-1.5">
              <AlertTriangle className="size-3" />
              Can&apos;t reach {base || "the server"}.
            </span>
            {base && (
              <span className="text-white/45">
                If it&apos;s running, the TLS certificate may need a one-time approval — open{" "}
                <a
                  href={base}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[#eff483] underline underline-offset-2"
                >
                  {base}
                </a>{" "}
                in a new tab, accept the warning, then Connect again.
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ConnBadge({ state }: { state: "loading" | "ok" | "err" }) {
  if (state === "ok") return <Badge variant="success">Connected</Badge>
  if (state === "err") return <Badge variant="destructive">Offline</Badge>
  return <Badge variant="warning">Connecting…</Badge>
}

function StatusCard({
  command,
  session,
  overrides,
  host,
  viewerPort,
}: {
  command: CommandSpec | null
  session: SessionInfo | null
  overrides: Overrides
  host: string
  viewerPort: number
}) {
  const simRunning =
    session?.status === "running" && (session.args?.sim === true || overrides.sim === true)
  const viewerUrl = host ? `http://${host}:${viewerPort}` : ""

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Status</CardTitle>
        <StatusBadge session={session} />
      </CardHeader>
      <CardContent className="gap-3">
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
          <dt className="text-white/40">Command</dt>
          <dd className="text-right font-mono text-white/80">
            {session?.command ?? command?.id ?? "—"}
          </dd>
          <dt className="text-white/40">PID</dt>
          <dd className="text-right font-mono text-white/80">{session?.pid ?? "—"}</dd>
          {session?.exitCode != null && (
            <>
              <dt className="text-white/40">Exit code</dt>
              <dd
                className={cn(
                  "text-right font-mono",
                  session.exitCode === 0 ? "text-white/80" : "text-red-400"
                )}
              >
                {session.exitCode}
              </dd>
            </>
          )}
        </dl>

        {simRunning && (
          <a
            href={viewerUrl}
            target="_blank"
            rel="noreferrer"
            className={cn(buttonVariants({ variant: "outline", size: "sm" }), "w-full")}
          >
            <ExternalLink />
            Open 3D viewer
          </a>
        )}

        {session?.status === "running" && session.command === "teleop" && (
          <p className="rounded-lg border border-white/10 bg-white/[0.02] p-3 text-xs leading-relaxed text-white/45">
            Put on the headset, open <span className="text-white/70">axol.almond.bot</span>, and
            connect to <span className="font-mono text-[#eff483]">{host || "this machine"}</span>.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function StatusBadge({ session }: { session: SessionInfo | null }) {
  if (!session) return <Badge variant="neutral">Idle</Badge>
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
