import { useEffect, useMemo, useState } from "react"
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
  fetchDatasets,
  isRobotFreeRun,
  isSimRun,
  motorFaultLabel,
  perRunFields,
  type CameraSpec,
  type CommandSpec,
  type DatasetInfo,
  type EpisodeControlSpec,
  type FormValue,
  type OperationMeta,
  type PolicyState,
  type RobotStatus,
  type SessionInfo,
} from "@/lib/supervisor"
import { CuratedForm, type FieldSuggestion } from "@/components/config-form"
import { ArmJointPicker } from "@/components/arm-joint-picker"
import { CameraFeeds, type VrHud } from "@/components/camera-feeds"
import { Card, CardContent } from "@/components/ui/card"
import { Button, buttonVariants } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
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
  vrPort,
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
  /** VR/teleop WebSocket port on the serve host (camera-feed signaling). */
  vrPort: number
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

  // Dataset picker: datasets found on the serve host, offered as a datalist
  // under the repo id field — typing a fresh id or a path stays possible.
  // Every operation with a repo_id gets the same picker: replay selects an
  // input dataset, while collect-data, DAgger, and run-policy can resume an
  // existing output dataset or type a new id. Refetched whenever the panel is
  // editable (a run that just ended may have added one). Older hosts without
  // /api/datasets simply leave the field a plain input.
  const wantsDatasets = runFields.some((f) => f.key === "repo_id")
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  useEffect(() => {
    if (!wantsDatasets || live) return
    let cancelled = false
    fetchDatasets()
      .then((found) => {
        if (!cancelled) setDatasets(found)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [wantsDatasets, live])
  const suggestions = useMemo<Record<string, FieldSuggestion[]> | undefined>(() => {
    if (!wantsDatasets || datasets.length === 0) return undefined
    return {
      repo_id: datasets.map((d) => ({
        value: d.repoId,
        label:
          d.episodes != null ? `${d.episodes} episode${d.episodes === 1 ? "" : "s"}` : undefined,
      })),
    }
  }, [wantsDatasets, datasets])

  const isSim = isSimRun(meta, settings)
  // Sim, or a run that never touches the arms (teleop's cart-only mode, or
  // the Mantis on its own CAN buses): either way the
  // arm-connection and motor-fault gates don't apply.
  const robotFree = isRobotFreeRun(meta, settings)
  const robotOk = robot?.state === "connected"
  const camCount = cameraCount(cameras)

  const blockers: string[] = []
  if (meta.requiresRobot && !robotFree && !robotOk) blockers.push("Connect Axol")
  // A faulted motor blocks every hardware operation (the server refuses the
  // start too) — driving through an over-temp / stalled / unreachable motor
  // risks the arm. Sim, cart-only, and Mantis runs never touch the arm motors.
  if (!robotFree) {
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
  // Teleop's run modes are mutually exclusive (the server refuses the start
  // too); catch the combination before the Start button instead of after.
  const modeFlags = ["sim", "mantis", "cart_only"].filter(
    (f) => meta.fields.includes(f) && Boolean(settings[f])
  )
  if (modeFlags.length > 1) {
    blockers.push("Sim, Mantis, and Cart only are mutually exclusive — enable only one")
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
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            {live &&
              Boolean(settings.mantis) &&
              (settings.mantis_source ?? "quest") !== "quest" && (
                <>
                  <Button
                    variant="outline"
                    onClick={() => onEpisode("bridge-toggle")}
                    disabled={busy}
                  >
                    Toggle tracking
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => onEpisode("bridge-reset")}
                    disabled={busy}
                  >
                    <RotateCcw /> Reset
                  </Button>
                </>
              )}
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
                      suggestions={suggestions}
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

              {/* Everything the operator watches during a session — episode
                  status/controls, the mirrored headset popups, and the live
                  camera feeds — grouped so it can expand to a fullscreen
                  operator view (the headset-off replacement for the HUD). */}
              {/* Skipped for robot-free runs: sim has the browser viewer, and
                  cart-only has no camera relay or HUD popups to mirror. */}
              {live && (meta.episodeControl || (meta.usesHeadset && !robotFree)) && (
                <OperatorDeck
                  label={meta.label}
                  episodeControl={meta.episodeControl}
                  showFeeds={meta.usesHeadset && !robotFree}
                  policy={policy}
                  onEpisode={onEpisode}
                  host={host}
                  vrPort={vrPort}
                />
              )}

              <RunningHints
                usesHeadset={meta.usesHeadset}
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

/** How long an armed confirm button waits for its second click (ms). */
const CONFIRM_ARM_MS = 5000

/**
 * The panel's built-in run-policy button set, used when the server's snapshot
 * doesn't carry its own `controls` (hosts predating server-driven controls).
 */
function legacyControls(phase: string): EpisodeControlSpec[] {
  if (phase === "ready") return [{ command: "start", label: "Start episode" }]
  if (phase === "recording" || phase === "deciding") {
    return [
      { command: "s", label: "Save" },
      { command: "r", label: "Discard" },
    ]
  }
  return []
}

function legacyStatus(phase: string): string {
  switch (phase) {
    case "recording":
      return "Recording — Save to keep, Discard to re-record."
    case "deciding":
      return "Time cap reached — Save to keep, Discard to re-record."
    case "ready":
      return "Reset the scene, then start the episode."
    case "resetting":
      return "Returning to rest…"
    default:
      return "Preparing…"
  }
}

/**
 * How each episode phase is presented: a short label, its colour scheme, and
 * whether the badge pulses (live activity) or spins (the server is working).
 * Unknown phases (from newer hosts) fall back to a neutral badge.
 */
const PHASE_STYLES: Record<string, { label: string; cls: string; kind: "dot" | "pulse" | "spin" }> =
  {
    preparing: {
      label: "Preparing",
      cls: "border-white/20 bg-white/[0.06] text-white/70",
      kind: "spin",
    },
    ready: { label: "Ready", cls: "border-sky-400/40 bg-sky-400/10 text-sky-200", kind: "dot" },
    countdown: {
      label: "Starting",
      cls: "border-amber-400/50 bg-amber-400/10 text-amber-200",
      kind: "pulse",
    },
    recording: {
      label: "Recording",
      cls: "border-red-500/60 bg-red-500/15 text-red-200",
      kind: "pulse",
    },
    deciding: {
      label: "Decide",
      cls: "border-amber-400/50 bg-amber-400/10 text-amber-200",
      kind: "pulse",
    },
    // The guarded return's limp gravity-comp hold: the arms hit something and
    // are waiting to be cleared by hand, so this needs the operator, not a
    // spinner.
    contact: {
      label: "Contact",
      cls: "border-orange-400/60 bg-orange-400/15 text-orange-200",
      kind: "pulse",
    },
    // run-policy's discard cleanup: the arms are limp for hand-repositioning
    // until the operator sends them back to rest.
    limp: {
      label: "Limp",
      cls: "border-orange-400/60 bg-orange-400/15 text-orange-200",
      kind: "pulse",
    },
    saving: {
      label: "Saving",
      cls: "border-emerald-400/50 bg-emerald-400/10 text-emerald-200",
      kind: "spin",
    },
    resetting: {
      label: "Resetting",
      cls: "border-sky-400/40 bg-sky-400/10 text-sky-200",
      kind: "spin",
    },
  }

function PhaseBadge({ phase }: { phase: string }) {
  const s = PHASE_STYLES[phase] ?? {
    label: phase,
    cls: "border-white/20 bg-white/[0.06] text-white/70",
    kind: "dot" as const,
  }
  return (
    <span
      className={`flex items-center gap-2 rounded-md border px-2.5 py-1 font-mono text-xs font-semibold tracking-widest uppercase ${s.cls}`}
    >
      {s.kind === "spin" ? (
        <Loader2 className="size-3 animate-spin" />
      ) : (
        <span className="relative flex size-2">
          {s.kind === "pulse" && (
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-60" />
          )}
          <span className="relative inline-flex size-2 rounded-full bg-current" />
        </span>
      )}
      {s.label}
    </span>
  )
}

/**
 * The label to headline the mirrored controller-confirmation popup with: the
 * matching panel button's label when the server drives the controls ("save"
 * pairs with the `s` command, "discard" with `r` — e.g. axol-pi's "Save
 * success" / "Save failure"), a generic fallback otherwise.
 */
function confirmLabel(action: "save" | "discard", policy: PolicyState | null): string {
  const controls = policy?.controls ?? legacyControls(policy?.phase ?? "recording")
  const command = action === "save" ? "s" : "r"
  const label = controls.find((c) => c.command === command)?.label
  return label ?? (action === "save" ? "Save episode" : "Discard episode")
}

/**
 * Mirror of the in-headset confirmation popup, driven by the relayed HUD
 * state: the operator armed a stop (save/discard) on the VR controller and
 * the headset is waiting for the second, confirming press.
 */
function ConfirmPopup({
  action,
  policy,
}: {
  action: "save" | "discard"
  policy: PolicyState | null
}) {
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
      <div className="flex w-full max-w-md flex-col items-center gap-3 rounded-xl border-2 border-amber-400/60 bg-[#151515] p-8 text-center shadow-2xl">
        <AlertTriangle className="size-7 text-amber-300" />
        <p className="font-heading text-2xl font-semibold">{confirmLabel(action, policy)}?</p>
        <p className="text-sm leading-relaxed text-white/60">
          Press the same controller button again to confirm — the other button cancels and keeps
          recording.
        </p>
      </div>
    </div>
  )
}

/**
 * Everything the operator watches during a live session, grouped so it can
 * expand to a fullscreen view: the episode status/controls, the camera
 * feeds, and the mirrored headset popups (confirmation dialog and record
 * countdown). This is the headset-off replacement for the in-headset HUD —
 * the VR controllers still drive the robot, and whatever they arm shows up
 * here.
 */
function OperatorDeck({
  label,
  episodeControl,
  showFeeds,
  policy,
  onEpisode,
  host,
  vrPort,
}: {
  label: string
  episodeControl: boolean
  showFeeds: boolean
  policy: PolicyState | null
  onEpisode: (command: string) => void
  host: string
  vrPort: number
}) {
  const [fullscreen, setFullscreen] = useState(false)
  // Relayed headset HUD state (armed confirm popup / record countdown), from
  // the camera-feed socket. Null when nothing is armed or no headset drives.
  const [hud, setHud] = useState<VrHud | null>(null)

  function toggleFullscreen() {
    const next = !fullscreen
    setFullscreen(next)
    if (next) {
      // Best-effort browser fullscreen on top of the overlay (we're in a
      // click handler, so the user-gesture requirement is met).
      document.documentElement.requestFullscreen?.().catch(() => {})
    }
  }

  useEffect(() => {
    if (!fullscreen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false)
    }
    // Leaving browser fullscreen (Esc there never reaches our key handler)
    // also collapses the overlay, so the two stay in sync.
    const onFsChange = () => {
      if (!document.fullscreenElement) setFullscreen(false)
    }
    window.addEventListener("keydown", onKey)
    document.addEventListener("fullscreenchange", onFsChange)
    return () => {
      window.removeEventListener("keydown", onKey)
      document.removeEventListener("fullscreenchange", onFsChange)
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {})
    }
  }, [fullscreen])

  const body = (
    <>
      {episodeControl && <EpisodeControls policy={policy} onEpisode={onEpisode} hud={hud} />}
      {showFeeds && (
        <CameraFeeds
          host={host}
          vrPort={vrPort}
          expanded={fullscreen}
          onToggleFullscreen={toggleFullscreen}
          onHud={setHud}
        />
      )}
      {hud?.confirm && <ConfirmPopup action={hud.confirm} policy={policy} />}
    </>
  )

  if (!fullscreen) return <div className="flex flex-col gap-5">{body}</div>
  return (
    // overflow-hidden: the feed grid shrinks to the leftover height instead,
    // so the episode state and every camera stay visible without scrolling.
    <div className="fixed inset-0 z-50 flex flex-col gap-3 overflow-hidden bg-[#0a0a0a] p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">
          {label} — operator view
        </span>
        <Button variant="outline" size="sm" onClick={toggleFullscreen}>
          Exit fullscreen
        </Button>
      </div>
      {body}
    </div>
  )
}

function EpisodeControls({
  policy,
  onEpisode,
  hud,
}: {
  policy: PolicyState | null
  onEpisode: (command: string) => void
  /** Relayed headset HUD state (drives the VR record-countdown readout). */
  hud: VrHud | null
}) {
  const phase = policy?.phase ?? "preparing"
  // Server-driven status/buttons when the op provides them (mirrors what the
  // headset HUD would show, so a session can be driven with the headset off);
  // the built-in run-policy set otherwise.
  const controls = policy?.controls ?? legacyControls(phase)
  const status = policy?.message ?? legacyStatus(phase)
  // Which confirm-gated command is armed and awaiting its second click — the
  // panel's stand-in for the headset's double-press save/discard confirmation.
  const [armed, setArmed] = useState<string | null>(null)
  useEffect(() => {
    if (armed == null) return
    const t = setTimeout(() => setArmed(null), CONFIRM_ARM_MS)
    return () => clearTimeout(t)
  }, [armed])
  // A phase change invalidates a pending confirmation (the episode moved on).
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setArmed(null)
  }, [phase])

  // A record countdown started from the VR controller (A press) reaches us
  // through the relayed HUD, not the op's snapshot — the backend only learns
  // about it when the headset flips to recording 3 s later. Anchor a local
  // deadline when the message arrives and tick the remaining seconds down.
  const [vrCountdownS, setVrCountdownS] = useState<number | null>(null)
  useEffect(() => {
    const remaining = hud?.countdownRemainingMs
    const deadline = remaining != null ? Date.now() + remaining : null
    const update = () =>
      setVrCountdownS(
        deadline != null ? Math.max(0, Math.ceil((deadline - Date.now()) / 1000)) : null
      )
    update()
    if (deadline == null) return
    const t = setInterval(update, 250)
    return () => clearInterval(t)
  }, [hud])

  const vrCounting = vrCountdownS != null && vrCountdownS > 0 && phase !== "recording"
  const displayPhase = vrCounting ? "countdown" : phase
  const displayStatus = vrCounting
    ? `Recording starts in ${vrCountdownS} s — controller countdown; press A again to cancel.`
    : status

  function click(control: EpisodeControlSpec) {
    if (control.confirm && armed !== control.command) {
      setArmed(control.command)
      return
    }
    setArmed(null)
    onEpisode(control.command)
  }

  const buttons = controls.filter((c) => !c.input)
  const inputs = controls.filter((c) => c.input)

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-[#eff483]/25 bg-[#eff483]/[0.04] p-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <PhaseBadge phase={displayPhase} />
          {policy?.episode != null && (
            <span className="font-heading text-lg font-semibold text-white/90">
              Episode {policy.episode}
            </span>
          )}
        </div>
        <span className="font-mono text-[0.65rem] text-white/40">
          {policy?.episodesRecorded ?? 0} saved
        </span>
      </div>
      <span className="text-sm text-white/60">{displayStatus}</span>
      {buttons.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {buttons.map((c, i) => (
            <Button
              key={c.command}
              variant={armed === c.command ? "destructive" : i === 0 ? "default" : "outline"}
              size="sm"
              onClick={() => click(c)}
            >
              {armed === c.command ? `Confirm: ${c.label}?` : c.label}
            </Button>
          ))}
        </div>
      )}
      {inputs.map((c) => (
        // Keyed on the placeholder too: a new target (e.g. the next saved
        // episode) resets the field even though the command stays the same.
        <EpisodeInputControl
          key={`${c.command}:${c.placeholder ?? ""}`}
          control={c}
          onEpisode={onEpisode}
        />
      ))}
    </div>
  )
}

/**
 * A server-driven episode control rendered as a text field + submit button
 * (`input: true` on the spec): submitting posts `${command} ${text}` to
 * /api/op/episode — e.g. a downstream op's post-episode notes, where the op
 * attaches the text to the just-saved episode. The field starts from the spec's
 * `value` (the current server-side text), so an earlier submission survives
 * phase changes and can be edited; submit is disabled while the text matches
 * what the server already has.
 */
function EpisodeInputControl({
  control,
  onEpisode,
}: {
  control: EpisodeControlSpec
  onEpisode: (command: string) => void
}) {
  const [text, setText] = useState(control.value ?? "")
  const serverValue = control.value ?? ""
  const unchanged = text.trim() === serverValue.trim()

  function submit() {
    if (unchanged || !text.trim()) return
    onEpisode(`${control.command} ${text.trim()}`)
  }

  return (
    <div className="flex items-center gap-2">
      <Input
        value={text}
        placeholder={control.placeholder}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit()
        }}
        className="h-8 flex-1"
      />
      <Button variant="outline" size="sm" disabled={unchanged || !text.trim()} onClick={submit}>
        {control.label}
      </Button>
    </div>
  )
}

function RunningHints({
  usesHeadset,
  session,
  isSim,
  host,
  viewerPort,
}: {
  usesHeadset: boolean
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
      {usesHeadset && (
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
