import { useMemo } from "react"
import {
  AlertTriangle,
  ExternalLink,
  Loader2,
  Play,
  RotateCcw,
  Settings2,
  Square,
} from "lucide-react"
import {
  cameraCount,
  motorFaultLabel,
  perRunFields,
  type CameraSpec,
  type CommandSpec,
  type FormValue,
  type OperationMeta,
  type PolicyState,
  type RobotStatus,
  type SessionInfo,
} from "@/lib/supervisor"
import { CuratedForm } from "@/components/config-form"
import { ArmJointPicker } from "@/components/arm-joint-picker"
import { Card, CardContent } from "@/components/ui/card"
import { Button, buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

/**
 * One operation's panel: just its per-run inputs (dataset / task / policy
 * identity) and Start/Stop. Everything reusable across runs — cameras, arm
 * behaviour, recording, inference — lives in the shared Settings dialog and is
 * folded in server-side at start.
 */
export function OperationPanel({
  meta,
  spec,
  settings,
  onChange,
  onReset,
  onResetAll,
  onOpenSettings,
  cameras,
  robot,
  live,
  stopping,
  busy,
  session,
  host,
  viewerPort,
  startPhase,
  policy,
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
  onOpenSettings: () => void
  cameras: CameraSpec
  robot: RobotStatus | null
  live: boolean
  stopping: boolean
  busy: boolean
  session: SessionInfo | null
  host: string
  viewerPort: number
  /** Progress label shown on the Start button while preparing (e.g. camera check). */
  startPhase: string | null
  /** run-policy episode phase/count (null unless run-policy is the live op). */
  policy: PolicyState | null
  onStart: () => void
  onStop: () => void
  onEpisode: (command: string) => void
}) {
  // Per-run inputs: every required field plus the op's curated run-identity
  // fields (repo id, task, policy path, episode, …) — required ones first.
  const runFields = useMemo(() => (spec ? perRunFields(spec, meta) : []), [spec, meta])
  // Gravity-comp's joint subset gets a proper picker instead of a text field.
  const jointField = runFields.find((f) => f.key === "free_joints")
  const textFields = useMemo(() => runFields.filter((f) => f.key !== "free_joints"), [runFields])

  const isSim = meta.id === "teleop" && Boolean(settings.sim)
  // UMI mode runs on the handheld rig's own CAN buses — the persistent Axol
  // robot connection is not involved.
  const isUmi =
    (meta.id === "teleop" || meta.id === "collect-data") && Boolean(settings.umi)
  const robotOk = robot?.state === "connected"
  const camCount = cameraCount(cameras)

  const blockers: string[] = []
  if (meta.requiresRobot && !isSim && !isUmi && !robotOk) blockers.push("Connect Axol")
  // A faulted motor blocks every hardware operation (the server refuses the
  // start too) — driving through an over-temp / stalled / unreachable motor
  // risks the arm. Sim never touches the motors, and UMI ops run on the
  // handheld rig's own buses rather than the Axol's.
  if (!isSim && !isUmi) {
    for (const f of robot?.faults ?? []) {
      blockers.push(`Fix motor fault: ${motorFaultLabel(f)}`)
    }
  }
  // Collect-data / run-policy record whichever camera slots are assigned, so
  // at least one serial must be set before starting (the rest are optional).
  if (meta.requiresCameras && camCount < 1) {
    blockers.push("Assign at least one camera serial in the Cameras settings tab")
  }
  for (const f of runFields) {
    if (f.required) {
      const v = settings[f.key]
      if (v === undefined || String(v).trim() === "") blockers.push(`Set ${f.label}`)
    }
  }

  const editedCount = Object.keys(settings).length
  const available = spec?.available ?? false

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
            {stopping ? (
              <Button variant="destructive" disabled>
                <Loader2 className="animate-spin" />
                Stopping…
              </Button>
            ) : live ? (
              <Button variant="destructive" onClick={onStop} disabled={busy}>
                {busy ? <Loader2 className="animate-spin" /> : <Square />}
                Stop
              </Button>
            ) : (
              <Button onClick={onStart} disabled={busy || !available || blockers.length > 0}>
                {busy ? <Loader2 className="animate-spin" /> : <Play />}
                {busy && startPhase ? startPhase : "Start"}
              </Button>
            )}
          </div>
        </div>

        <CardContent className="gap-5 p-5">
          {!available ? (
            <Unavailable spec={spec} />
          ) : (
            <>
              {runFields.length > 0 && (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs tracking-widest text-white/40 uppercase">
                      This run
                    </span>
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
                  </div>
                  {textFields.length > 0 && (
                    <CuratedForm
                      fields={textFields}
                      overrides={settings}
                      disabled={live}
                      onChange={onChange}
                      onReset={onReset}
                    />
                  )}
                  {jointField && (
                    <ArmJointPicker
                      value={settings[jointField.key]}
                      disabled={live}
                      onChange={(v) => onChange(jointField.key, v)}
                      onReset={() => onReset(jointField.key)}
                    />
                  )}
                </>
              )}

              <button
                type="button"
                onClick={onOpenSettings}
                className="flex w-fit items-center gap-1.5 text-xs text-white/40 transition-colors hover:text-white/70"
              >
                <Settings2 className="size-3.5" />
                {runFields.length > 0
                  ? "Cameras, arm behaviour, recording and everything else live in Settings"
                  : "No per-run inputs — configure everything in Settings, then press Start"}
              </button>

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

              {meta.id === "run-policy" && live && (
                <EpisodeControls policy={policy} onEpisode={onEpisode} />
              )}

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

function EpisodeControls({
  policy,
  onEpisode,
}: {
  policy: PolicyState | null
  onEpisode: (command: string) => void
}) {
  const phase = policy?.phase ?? "preparing"
  const ready = phase === "ready" // between-episode gate: start the next episode
  // Episode in flight, or the safety-cap decision after it — Save/Discard apply.
  const canChoose = phase === "recording" || phase === "deciding"
  // Status line so both the initiator and any other computer see what's happening.
  const status =
    phase === "recording"
      ? "Recording — Save to keep, Discard to re-record."
      : phase === "deciding"
        ? "Time cap reached — Save to keep, Discard to re-record."
        : phase === "ready"
          ? "Reset the scene, then start the episode."
          : phase === "resetting"
            ? "Returning to rest…"
            : "Preparing…"
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[#eff483]/25 bg-[#eff483]/[0.04] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs tracking-widest text-[#eff483]/80 uppercase">
          Episode control
        </span>
        <span className="font-mono text-[0.65rem] text-white/40">
          {policy?.episodesRecorded ?? 0} saved
        </span>
      </div>
      <span className="text-xs text-white/50">{status}</span>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" disabled={!ready} onClick={() => onEpisode("start")}>
          Start episode
        </Button>
        <Button variant="outline" size="sm" disabled={!canChoose} onClick={() => onEpisode("s")}>
          Save
        </Button>
        <Button variant="outline" size="sm" disabled={!canChoose} onClick={() => onEpisode("r")}>
          Discard
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
    case "stopping":
      return <Badge variant="warning">Stopping</Badge>
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
