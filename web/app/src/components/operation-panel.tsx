import { useMemo, useRef, useState } from "react"
import {
  AlertTriangle,
  ChevronRight,
  Download,
  ExternalLink,
  Loader2,
  Play,
  RotateCcw,
  Square,
  Upload,
} from "lucide-react"
import {
  filterSchema,
  flattenFields,
  isModified,
  type CommandSpec,
  type FormValue,
  type OperationMeta,
  type RobotStatus,
  type SchemaNode,
  type SessionInfo,
  type ZedLinkStatus,
  type ZedSpec,
} from "@/lib/supervisor"
import { ConfigForm, CuratedForm } from "@/components/config-form"
import { Card, CardContent } from "@/components/ui/card"
import { Button, buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export function OperationPanel({
  meta,
  spec,
  settings,
  onChange,
  onReset,
  onResetAll,
  onExport,
  onImport,
  zedSettings,
  zedLink,
  robot,
  live,
  busy,
  session,
  host,
  viewerPort,
  onStart,
  onStop,
  onEpisode,
}: {
  meta: OperationMeta
  spec: CommandSpec | null
  settings: Record<string, FormValue>
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
  onResetAll: () => void
  onExport: () => void
  onImport: (text: string) => void
  zedSettings: ZedSpec
  zedLink: ZedLinkStatus | null
  robot: RobotStatus | null
  live: boolean
  busy: boolean
  session: SessionInfo | null
  host: string
  viewerPort: number
  onStart: () => void
  onStop: () => void
  onEpisode: (command: string) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  // Required fields stay visible; every optional field lives in the dropdown.
  const allFields = useMemo(() => (spec ? flattenFields(spec.schema) : []), [spec])
  const requiredFields = useMemo(() => allFields.filter((f) => f.required), [allFields])
  const optionalSchema = useMemo(
    () => (spec ? filterSchema(spec.schema, new Set(requiredFields.map((f) => f.key))) : []),
    [spec, requiredFields]
  )

  const isSim = meta.id === "teleop" && Boolean(settings.sim)
  const robotOk = robot?.state === "connected"
  const zedOk = Boolean(zedLink?.connected)
  const camCount = Object.values(zedSettings.cameras).filter((s) => s.trim()).length

  const blockers: string[] = []
  if (meta.requiresRobot && !isSim && !robotOk) blockers.push("Connect Axol")
  if (meta.requiresZed && !zedOk) blockers.push("Connect the ZED box")
  // ZED frame timestamps are only valid once both machines' clocks are
  // PTP-locked, so collect-data / run-policy can't start until then.
  if (meta.requiresZed && zedOk && !zedLink?.ptp?.locked) {
    if (zedLink?.ptp?.needsSudo) blockers.push("Enter the sudo password to start PTP clock sync")
    else if (zedLink?.ptp?.error) blockers.push(`PTP clock sync failed: ${zedLink.ptp.error}`)
    else blockers.push("Wait for PTP clock sync to lock")
  }
  if (meta.requiresZed && camCount === 0)
    blockers.push("Add at least one camera serial in the ZED connection")
  for (const f of allFields) {
    if (f.required) {
      const v = settings[f.key]
      if (v === undefined || String(v).trim() === "") blockers.push(`Set ${f.label}`)
    }
  }

  const editedCount = Object.keys(settings).length
  const available = spec?.available ?? false

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    file.text().then(onImport)
    e.target.value = ""
  }

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <Card className="gap-0 p-0">
        <div className="flex flex-col gap-4 border-b border-white/10 p-5 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="font-heading text-lg font-semibold">{meta.label}</h2>
              <StatusBadge session={live ? session : null} />
            </div>
            <p className="mt-2 max-w-prose text-sm text-white/55">{meta.description}</p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {live ? (
              <Button variant="destructive" onClick={onStop} disabled={busy}>
                {busy ? <Loader2 className="animate-spin" /> : <Square />}
                Stop
              </Button>
            ) : (
              <Button onClick={onStart} disabled={busy || !available || blockers.length > 0}>
                {busy ? <Loader2 className="animate-spin" /> : <Play />}
                Start
              </Button>
            )}
          </div>
        </div>

        <CardContent className="gap-5 p-5">
          {!available ? (
            <Unavailable spec={spec} />
          ) : (
            <>
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs tracking-widest text-white/40 uppercase">
                  Settings
                </span>
                <div className="flex items-center gap-1">
                  {editedCount > 0 && !live && (
                    <button
                      type="button"
                      onClick={onResetAll}
                      className="flex items-center gap-1 px-2 text-xs text-white/40 hover:text-white/70"
                    >
                      <RotateCcw className="size-3" />
                      Reset
                    </button>
                  )}
                  <Button variant="ghost" size="sm" onClick={onExport}>
                    <Download />
                    Export
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => fileRef.current?.click()}
                    disabled={live}
                  >
                    <Upload />
                    Import
                  </Button>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="application/json"
                    className="hidden"
                    onChange={handleFile}
                  />
                </div>
              </div>

              {requiredFields.length > 0 && (
                <CuratedForm
                  fields={requiredFields}
                  overrides={settings}
                  disabled={live}
                  onChange={onChange}
                  onReset={onReset}
                />
              )}

              {optionalSchema.length > 0 && (
                <OptionalSettings
                  schema={optionalSchema}
                  overrides={settings}
                  disabled={live}
                  onChange={onChange}
                  onReset={onReset}
                />
              )}

              {requiredFields.length === 0 && optionalSchema.length === 0 && (
                <p className="text-sm text-white/40">No settings — just press Start.</p>
              )}

              {blockers.length > 0 && !live && (
                <div className="flex flex-col gap-1 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
                  <span className="font-medium">Before you can start:</span>
                  <ul className="list-inside list-disc">
                    {blockers.map((b) => (
                      <li key={b}>{b}</li>
                    ))}
                  </ul>
                </div>
              )}

              {meta.id === "run-policy" && live && <EpisodeControls onEpisode={onEpisode} />}

              <RunningHints
                op={meta.id}
                session={live ? session : null}
                isSim={isSim}
                host={host}
                viewerPort={viewerPort}
              />
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function OptionalSettings({
  schema,
  overrides,
  disabled,
  onChange,
  onReset,
}: {
  schema: SchemaNode[]
  overrides: Record<string, FormValue>
  disabled: boolean
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
}) {
  const [open, setOpen] = useState(false)
  const leaves = useMemo(() => flattenFields(schema), [schema])
  const editedCount = leaves.filter((f) => isModified(f, overrides[f.key])).length

  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        <ChevronRight
          className={cn("size-4 shrink-0 text-white/40 transition-transform", open && "rotate-90")}
        />
        <span className="text-sm font-medium">Optional</span>
        <span className="text-xs text-white/30">{leaves.length}</span>
        {editedCount > 0 && (
          <span className="ml-auto rounded-full bg-[#eff483]/15 px-2 py-0.5 font-mono text-[0.65rem] text-[#eff483]">
            {editedCount} edited
          </span>
        )}
      </button>
      {open && (
        <div className="border-t border-white/10 p-3">
          <ConfigForm
            schema={schema}
            overrides={overrides}
            disabled={disabled}
            onChange={onChange}
            onReset={onReset}
          />
        </div>
      )}
    </div>
  )
}

function EpisodeControls({ onEpisode }: { onEpisode: (command: string) => void }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[#eff483]/25 bg-[#eff483]/[0.04] p-3">
      <span className="font-mono text-xs tracking-widest text-[#eff483]/80 uppercase">
        Episode control
      </span>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={() => onEpisode("start")}>
          Start episode
        </Button>
        <Button variant="outline" size="sm" onClick={() => onEpisode("s")}>
          Save + next
        </Button>
        <Button variant="outline" size="sm" onClick={() => onEpisode("r")}>
          Re-record
        </Button>
        <Button variant="destructive" size="sm" onClick={() => onEpisode("q")}>
          Quit episodes
        </Button>
      </div>
    </div>
  )
}

function RunningHints({
  op,
  session,
  isSim,
  host,
  viewerPort,
}: {
  op: string
  session: SessionInfo | null
  isSim: boolean
  host: string
  viewerPort: number
}) {
  if (!session || session.status !== "running") return null
  const viewerUrl = host ? `http://${host}:${viewerPort}` : ""
  return (
    <div className="flex flex-col gap-3">
      {isSim && viewerUrl && (
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
      {op === "teleop" && (
        <p className="rounded-lg border border-white/10 bg-white/[0.02] p-3 text-xs leading-relaxed text-white/45">
          Put on the headset, open <span className="text-white/70">axol.almond.bot</span>, and
          connect to <span className="font-mono text-[#eff483]">{host || "this machine"}</span>.
        </p>
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

function Unavailable({ spec }: { spec: CommandSpec | null }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-4 text-sm">
      <div className="flex items-center gap-2 font-medium text-amber-300/90">
        <AlertTriangle className="size-4" />
        Not available on this server
      </div>
      <p className="text-white/55">
        This operation needs dependencies that aren&apos;t installed on the connected machine (e.g.
        the <span className="font-mono">lerobot</span> / ZED extras, or Axol hardware).
      </p>
      {spec?.error && (
        <code className="rounded bg-black/30 p-2 text-xs break-words text-white/45">
          {spec.error}
        </code>
      )}
    </div>
  )
}
