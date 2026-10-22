import { useEffect, useRef, useState, type ReactNode } from "react"
import {
  Bluetooth,
  Check,
  ChevronLeft,
  ChevronRight,
  Clipboard,
  Download,
  ExternalLink,
  ListChecks,
  Loader2,
  RadioTower,
  Save,
  Square,
} from "lucide-react"
import {
  fetchTrackerBindings,
  saveLighthouseQualityMode,
  sendSessionInput,
  stopSession,
  useSessionLogs,
  type LighthouseQualityMode,
  type SessionInfo,
  type TrackerBackend,
  type TrackerBinding,
  type TrackerSourceReadiness,
} from "@/lib/supervisor"
import { startDiagnosticsRun } from "@/lib/telemetry"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/toast"
import { cn } from "@/lib/utils"
import { UltimateWifiPanel } from "./ultimate-wifi-panel"

type SetupCommand =
  | "tracker.install"
  | "tracker.pair"
  | "tracker.lighthouse.check"
  | "tracker.lighthouse.validate"
  | "tracker.lighthouse.calibrate"
  | "tracker.identify"
  | "tracker.ultimate.install"
  | "tracker.ultimate.check"

type StepKey = "runtime" | "stations" | "headset" | "dongle" | "wifi" | "identify" | "validation"

interface FlowStep {
  key: StepKey
  title: string
  /** True once the host confirms this step; Continue stays disabled otherwise. */
  done: boolean
  /** Shown next to the disabled Continue button while the step is unresolved. */
  pending: string
  body: ReactNode
}

function backendFor(source: string): TrackerBackend | null {
  if (source === "lighthouse") return "survive"
  if (source === "ultimate") return "ultimate"
  return null
}

function trackerSessionBackend(session: SessionInfo): TrackerBackend | null {
  if (
    session.command === "tracker.install" ||
    session.command === "tracker.pair" ||
    session.command === "tracker.lighthouse.check" ||
    session.command === "tracker.lighthouse.validate" ||
    session.command === "tracker.lighthouse.calibrate"
  )
    return "survive"
  if (
    session.command === "tracker.ultimate.install" ||
    session.command === "tracker.ultimate.check"
  )
    return "ultimate"
  if (session.command === "tracker.identify") {
    return session.args.backend === "survive" || session.args.backend === "ultimate"
      ? session.args.backend
      : null
  }
  return null
}

/** Which flow step owns a running setup command, so the flow stays on it. */
function stepForCommand(command: string | undefined): StepKey | null {
  switch (command) {
    case "tracker.install":
    case "tracker.ultimate.install":
      return "runtime"
    case "tracker.pair":
    case "tracker.lighthouse.check":
      return "stations"
    case "tracker.lighthouse.validate":
    case "tracker.lighthouse.calibrate":
      return "validation"
    case "tracker.identify":
      return "identify"
    case "tracker.ultimate.check":
      // The optional host check is launched from the CLI / Diagnostics now;
      // while it runs, hold the flow on the step it validates.
      return "dongle"
    default:
      return null
  }
}

function latestPrompt(lines: string[]): { index: number; text: string } | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]
    if (line.startsWith("[prompt] ")) {
      return { index: i, text: line.slice("[prompt] ".length).trim() }
    }
  }
  return null
}

function ago(epochSeconds: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds)
  if (seconds < 90) return "just now"
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`
  return `${Math.round(seconds / 86400)} d ago`
}

/**
 * Step-by-step Mantis tracker setup in Settings → Mantis.
 *
 * One step is shown at a time; each resolves from host readiness rather than
 * operator say-so, and Continue only unlocks once the host confirms it. Every
 * fix is an in-panel action (install, pair, check, identify, calibrate, save)
 * so the terminal is never required.
 */
export function TrackerBindingPanel({
  source,
  sourceSaved,
  calibrationContextSaved,
  onQuestKeySelect,
  onSaveSettings,
  savingSettings = false,
  blockedReason,
  hostSession,
  onHostSessionChange,
}: {
  source: string
  /** False while a source selection is still persisting to the host (the
   * dropdown saves on change) or its write failed; setup actions hold. */
  sourceSaved: boolean
  /** Source plus profile-scoped Quest datum match the persisted host settings. */
  calibrationContextSaved: boolean
  /** Fill the shared Quest datum field; the enclosing Settings save persists it. */
  onQuestKeySelect?: (key: string) => void
  /** Save the pending Settings draft in place; undefined when nothing is pending. */
  onSaveSettings?: () => void
  savingSettings?: boolean
  /** Why a different host owner prevents a new setup action. */
  blockedReason: string | null
  /** Active generic session polled by the control panel (cross-tab owner). */
  hostSession: SessionInfo | null
  onHostSessionChange: (session: SessionInfo | null) => void
}) {
  const toast = useToast()
  const backend = backendFor(source)
  const [bindings, setBindings] = useState<Partial<Record<TrackerBackend, TrackerBinding>>>({})
  const [bindingsLoaded, setBindingsLoaded] = useState(false)
  const [bindingsError, setBindingsError] = useState(false)
  const [readiness, setReadiness] = useState<TrackerSourceReadiness | null>(null)
  const [checkedAt, setCheckedAt] = useState(() => Date.now() / 1000)
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [expectedStationChoice, setExpectedStationChoice] = useState<number | null>(null)
  const [qualityModeChoice, setQualityModeChoice] = useState<LighthouseQualityMode | null>(null)
  const [dismissed, setDismissed] = useState<{ id: string; promptIndex: number } | null>(null)
  // Keep ownership of an in-flight setup session even if the operator edits
  // the source selector. Otherwise its progress/prompt/Stop controls vanish
  // while the old subprocess continues blocking the host.
  const externalTrackerSession =
    hostSession && trackerSessionBackend(hostSession) !== null ? hostSession : null
  const selectedSession = externalTrackerSession ?? session
  const activeSession =
    selectedSession && trackerSessionBackend(selectedSession) !== null ? selectedSession : null
  const sessionBackend = activeSession ? trackerSessionBackend(activeSession) : null
  const displayedBackend = sessionBackend ?? backend
  const { lines, status } = useSessionLogs(activeSession?.id ?? null)
  const completionRef = useRef<string | null>(null)

  const [retries, setRetries] = useState(0)
  useEffect(() => {
    let cancelled = false
    fetchTrackerBindings()
      .then(({ bindings: found, sources }) => {
        if (cancelled) return
        setBindings(found)
        setReadiness(sources ?? null)
        setCheckedAt(Date.now() / 1000)
        setBindingsLoaded(true)
        setBindingsError(false)
      })
      .catch(() => {
        if (cancelled) return
        setBindingsLoaded(true)
        setBindingsError(true)
      })
    return () => {
      cancelled = true
    }
  }, [retries])

  // Keep runtime, binding, transform-file, survey, and (for Quest) live WebXR
  // datum readiness current while Settings is open. Setup runs and edits to
  // the calibration file happen outside React state, so a one-shot fetch
  // otherwise leaves an already-fixed step looking blocked until reload.
  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      // An expired report must stop appearing current even if this host read hangs.
      setCheckedAt(Date.now() / 1000)
      fetchTrackerBindings()
        .then(({ bindings: found, sources }) => {
          if (cancelled) return
          setBindings(found)
          setReadiness(sources ?? null)
          setCheckedAt(Date.now() / 1000)
          setBindingsLoaded(true)
          setBindingsError(false)
        })
        .catch(() => {
          if (!cancelled) setBindingsError(true)
        })
    }
    const timer = window.setInterval(refresh, 3000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [backend])

  const currentLines = !status || status.id === activeSession?.id ? lines : []
  const prompt = latestPrompt(currentLines)
  const pendingPrompt =
    prompt &&
    !(dismissed && dismissed.id === activeSession?.id && dismissed.promptIndex === prompt.index)
      ? prompt.text
      : null
  const checkingUltimate = activeSession?.command === "tracker.ultimate.check"
  const checkingLighthouse = activeSession?.command === "tracker.lighthouse.check"
  const validatingLighthouse =
    activeSession?.command === "tracker.lighthouse.validate" ||
    activeSession?.command === "tracker.lighthouse.calibrate"
  const checking = checkingUltimate || checkingLighthouse
  const transcriptLimit = checking || validatingLighthouse ? 24 : 6
  const transcript = currentLines
    .filter((line) => line.trim() && !line.startsWith("[serve]") && !line.startsWith("[prompt] "))
    .slice(-transcriptLimit)
  const activeLine = transcript.at(-1) ?? null
  // useSessionLogs can also retain the previous session's terminal status for
  // one render. Never let it complete a new run before that run has started.
  const current = status?.id === activeSession?.id ? status : activeSession
  const running =
    current?.status === "starting" ||
    current?.status === "running" ||
    current?.status === "stopping"
  const terminal = current?.status === "exited" || current?.status === "error"
  const succeeded = terminal && current?.status === "exited" && (current.exitCode ?? 0) === 0
  const pairing = activeSession?.command === "tracker.pair"
  const installing =
    activeSession?.command === "tracker.install" ||
    activeSession?.command === "tracker.ultimate.install"
  const sessionStep = stepForCommand(activeSession?.command)

  useEffect(() => {
    if (!terminal || !activeSession || completionRef.current === activeSession.id) return
    completionRef.current = activeSession.id
    const paired = activeSession.command === "tracker.pair"
    const installed =
      activeSession.command === "tracker.install" ||
      activeSession.command === "tracker.ultimate.install"
    const checkedUltimate = activeSession.command === "tracker.ultimate.check"
    const checkedLighthouse = activeSession.command === "tracker.lighthouse.check"
    const validatedLighthouse =
      activeSession.command === "tracker.lighthouse.validate" ||
      activeSession.command === "tracker.lighthouse.calibrate"
    fetchTrackerBindings()
      .then(({ bindings: found, sources }) => {
        setBindings(found)
        setReadiness(sources ?? null)
        setCheckedAt(Date.now() / 1000)
        setBindingsLoaded(true)
        setBindingsError(false)
      })
      .catch(() => setBindingsError(true))
    if (current?.status === "exited" && (current.exitCode ?? 0) === 0) {
      toast.success(
        validatedLighthouse
          ? "Lighthouse check finished. Review the tracking quality result."
          : checkedLighthouse
            ? "Base stations are on distinct channels and both trackers report."
            : checkedUltimate
              ? "Ultimate host setup is ready."
              : installed
                ? displayedBackend === "ultimate"
                  ? "Ultimate Linux runtime installed."
                  : "Lighthouse tracking support installed."
                : paired
                  ? "Lighthouse tracker paired."
                  : "Mantis tracker binding saved."
      )
    } else {
      toast.error(
        validatedLighthouse
          ? "Lighthouse validation failed. Review the results before recording."
          : checkedLighthouse
            ? "Base-station check found setup issues. Review the results in the step."
            : checkedUltimate
              ? "Ultimate readiness check found setup issues. Review the results in the step."
              : installed
                ? "Tracker runtime installation failed. See the status in the step."
                : paired
                  ? "Tracker pairing failed. See the status in the step."
                  : "Tracker identification failed. See the status in the step."
      )
    }
    // Persist the terminal status for same-source feedback, but release an old
    // source immediately if the selector changed while this run was active.
    // Queueing avoids a synchronous state write inside the effect body.
    queueMicrotask(() => {
      setSession((selected) => {
        if (selected?.id !== activeSession.id) return selected
        return sessionBackend === backend ? (current ?? activeSession) : null
      })
    })
  }, [terminal, activeSession, displayedBackend, current, toast, sessionBackend, backend])

  // Each run gets a fresh session id, so the completion guard needs no reset.
  async function launch(command: SetupCommand, args: Record<string, unknown> = {}) {
    if (busy || blockedReason !== null || running) return
    setBusy(true)
    setDismissed(null)
    try {
      const { session: started } = await startDiagnosticsRun(command, args)
      setSession(started)
      onHostSessionChange(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function capture() {
    if (!activeSession || !prompt) return
    setDismissed({ id: activeSession.id, promptIndex: prompt.index })
    try {
      await sendSessionInput(activeSession.id)
    } catch (error) {
      setDismissed(null)
      toast.error(String(error))
    }
  }

  async function stop() {
    if (!activeSession) return
    setBusy(true)
    try {
      const stopped = await stopSession(activeSession.id)
      setSession(stopped)
      onHostSessionChange(null)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  const label =
    source === "quest" ? "Quest / WebXR" : source === "ultimate" ? "VIVE Ultimate" : "Lighthouse"
  const actionsLocked = busy || blockedReason !== null || running
  const canAct = !actionsLocked && (sessionBackend === null || sessionBackend === backend)
  const expectedStationCount =
    expectedStationChoice ?? readiness?.lighthouse.expectedBaseStations ?? 2
  const qualityMode = qualityModeChoice ?? readiness?.lighthouse.qualityMode

  async function chooseQualityMode(mode: LighthouseQualityMode) {
    setBusy(true)
    try {
      const saved = await saveLighthouseQualityMode(mode)
      setQualityModeChoice(saved.mode)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  if (!bindingsLoaded) {
    return (
      <FlowCard title={`${label} setup`}>
        <p className="flex items-center gap-2 text-sm text-white/55">
          <Loader2 className="size-4 animate-spin" /> Checking this host…
        </p>
      </FlowCard>
    )
  }
  if (readiness === null) {
    return (
      <FlowCard title={`${label} setup`}>
        <p className="text-sm text-red-300/85">
          {bindingsError
            ? "Setup status could not be read from the host."
            : "This host does not report tracker setup status; update Axol on the host."}
        </p>
        {bindingsError && (
          <Button
            variant="outline"
            size="sm"
            className="self-start"
            onClick={() => setRetries((n) => n + 1)}
          >
            Retry
          </Button>
        )}
      </FlowCard>
    )
  }

  /** Live progress/result of the setup run owned by a step, rendered inside it. */
  const progressFor = (key: StepKey): ReactNode => {
    if (sessionStep !== key || !activeSession || (!running && !terminal)) return null
    return (
      <div className="flex flex-col gap-2 rounded-md border border-white/10 bg-black/20 p-3">
        {transcript.length > 0 && (
          <div
            className="flex max-h-40 flex-col gap-1 overflow-y-auto font-mono text-[11px] leading-relaxed text-white/45"
            aria-live="polite"
          >
            {transcript.map((line, index) => (
              <p key={`${index}:${line}`}>{line}</p>
            ))}
          </div>
        )}
        {running ? (
          <>
            {pendingPrompt && !pairing && !installing && !checking ? (
              <>
                <p className="text-sm text-amber-100/85">{pendingPrompt}</p>
                <Button size="sm" className="self-start" onClick={capture}>
                  {validatingLighthouse ? "Continue" : "Start 3-second capture"}
                </Button>
              </>
            ) : dismissed?.id === activeSession.id ? (
              <>
                {validatingLighthouse && prompt && (
                  <p className="text-sm text-amber-100/85">{prompt.text}</p>
                )}
                <p className="flex items-center gap-2 text-sm text-white/60">
                  <Loader2 className="size-4 animate-spin" />
                  {validatingLighthouse
                    ? "Running tracking check… Follow the current step's instructions."
                    : "Capturing motion…"}
                </p>
              </>
            ) : (
              <p className="flex items-center gap-2 text-sm text-white/60">
                <Loader2 className="size-4 animate-spin" />
                {installing
                  ? `Installing ${displayedBackend === "ultimate" ? "Ultimate" : "Lighthouse"} tracking support…`
                  : checkingUltimate
                    ? "Running the non-invasive Ultimate readiness check…"
                    : checkingLighthouse
                      ? "Listening to libsurvive for base stations and trackers…"
                      : validatingLighthouse
                        ? "Preparing the Lighthouse tracking check…"
                        : pairing
                          ? "Pairing remains active…"
                          : "Waiting for both trackers…"}
              </p>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="self-start"
              onClick={stop}
              disabled={busy || current?.status === "stopping"}
            >
              <Square /> Stop
            </Button>
          </>
        ) : (
          <p className={succeeded ? "text-xs text-emerald-300/80" : "text-xs text-red-300/80"}>
            {succeeded
              ? checkingUltimate
                ? "Ultimate host setup passed."
                : validatingLighthouse
                  ? "Check finished. Review the tracking quality result above before recording."
                  : checkingLighthouse
                    ? "Every base station is on its own channel and both trackers report."
                    : installing
                      ? "Tracking support installed."
                      : pairing
                        ? "Tracker paired. Repeat for the other tracker if needed."
                        : "Binding saved."
              : (current?.error ??
                [...transcript].reverse().find((line) => /^\s*FAIL\b/.test(line)) ??
                (activeLine?.startsWith("Saved to ") ? null : activeLine) ??
                (validatingLighthouse
                  ? "Lighthouse validation failed. Restore visibility and check the base-station setup."
                  : null) ??
                (checking
                  ? "The check found setup issues."
                  : installing
                    ? "Installation failed."
                    : pairing
                      ? "Pairing failed."
                      : "Identification failed."))}
          </p>
        )}
      </div>
    )
  }

  const identifyStep = (
    backendKey: TrackerBackend,
    blocker: string | null,
    runtimeReady: boolean
  ): FlowStep => {
    const binding = bindings[backendKey]
    return {
      key: "identify",
      title: "Identify sides",
      done: Boolean(binding?.complete),
      pending: "Bind both trackers to a Mantis side",
      body: (
        <StepBody>
          <p>
            The tracker addresses do not say which rig they are on. Identify watches which tracker
            moves for each side and saves that mapping. Move only the requested Mantis during each
            3-second capture.
          </p>
          {binding?.complete && (
            <p className="font-mono text-[11px] text-white/45">
              Left {binding.left} · Right {binding.right}
            </p>
          )}
          {blocker && !running && (
            <p className="text-xs text-amber-300/80">Identify is waiting: {blocker}.</p>
          )}
          <Button
            variant="outline"
            size="sm"
            className="self-start"
            onClick={() => launch("tracker.identify", { backend: backendKey, web_prompts: true })}
            disabled={!canAct || !sourceSaved || !runtimeReady || blocker !== null}
          >
            {busy ? <Loader2 className="animate-spin" /> : <RadioTower />}
            {binding?.complete ? "Identify again" : "Identify trackers"}
          </Button>
          {progressFor("identify")}
        </StepBody>
      ),
    }
  }

  let steps: FlowStep[]
  if (source === "quest") {
    const q = readiness.quest
    const live = q.liveDatum ?? null
    const liveKey = live?.commonKey ?? null
    const liveMismatch = Boolean(
      live?.live && live.commonKey && q.calibrationKey && live.commonKey !== q.calibrationKey
    )
    const headsetDone =
      calibrationContextSaved &&
      q.datumStatus === "configured" &&
      q.poseSpace === "grip" &&
      !liveMismatch
    steps = [
      {
        key: "headset",
        title: "Headset",
        done: headsetDone,
        pending: !calibrationContextSaved
          ? "Save the Quest calibration key"
          : q.datumStatus === "configured" && q.poseSpace !== "grip"
            ? "The saved datum must be a grip-space key"
            : q.datumStatus === "ambiguous"
              ? "Choose one Quest calibration key"
              : liveMismatch
                ? "The live controllers report a different key than the saved one"
                : "Connect the headset and use its reported calibration key",
        body: (
          <StepBody>
            <p>
              Put the Quest and this host on the same LAN (or connect USB-C with Developer Mode and
              use the General settings → Quest tab). Start a Quest bring-up collection run — Collect
              data with mantis_allow_uncalibrated set under Advanced (Mantis teleop never starts the
              VR server) — then in the Quest browser open{" "}
              <span className="text-white/70">axol.almond.bot</span>, enter this host, connect, and
              choose Enter VR. Hold both Touch controllers.
            </p>
            {live ? (
              <div className="flex flex-col gap-2 rounded-md border border-white/10 bg-black/20 p-2.5">
                <span className="text-[11px] text-white/55">
                  {live.live ? "Live" : `Last reported ${live.ageSeconds.toFixed(0)} s ago`} — left:{" "}
                  {live.left.profile ?? "profile missing"} ·{" "}
                  {live.left.poseSpace ?? "space missing"}; right:{" "}
                  {live.right.profile ?? "profile missing"} ·{" "}
                  {live.right.poseSpace ?? "space missing"}
                </span>
                {live.live && liveKey ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <CopyableCommand command={liveKey} />
                    {onQuestKeySelect && liveKey !== q.calibrationKey && (
                      <Button size="sm" onClick={() => onQuestKeySelect(liveKey)}>
                        <Check /> Use this key
                      </Button>
                    )}
                  </div>
                ) : !live.live ? (
                  <span className="text-[11px] text-amber-300/80">
                    Reconnect or resume the headset; a stale report cannot be used.
                  </span>
                ) : live.left.poseSpace === "target-ray" ||
                  live.right.poseSpace === "target-ray" ? (
                  <span className="text-[11px] text-red-300/85">
                    This WebXR runtime only supplied target-ray poses. Mantis requires gripSpace;
                    update or restart the Quest browser and controllers.
                  </span>
                ) : (
                  <span className="text-[11px] text-amber-300/80">
                    The two controllers do not report one matching calibration datum.
                  </span>
                )}
              </div>
            ) : (
              <p className="text-xs text-white/45">No controller report from a headset yet.</p>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <ReadinessBadge
                tone={
                  q.datumStatus === "configured"
                    ? q.poseSpace === "grip"
                      ? "ready"
                      : "warning"
                    : q.datumStatus === "ambiguous"
                      ? "error"
                      : "warning"
                }
              >
                {q.datumStatus === "configured"
                  ? `Saved: ${q.controllerProfile} · ${q.poseSpace}`
                  : q.datumStatus === "ambiguous"
                    ? "Multiple Quest calibrations saved"
                    : q.datumStatus === "invalid"
                      ? "Saved Quest key is invalid"
                      : "No Quest calibration key saved"}
              </ReadinessBadge>
              {!calibrationContextSaved && (
                <Button
                  size="sm"
                  onClick={onSaveSettings}
                  disabled={!onSaveSettings || savingSettings}
                >
                  {savingSettings ? <Loader2 className="animate-spin" /> : <Save />}
                  Save settings
                </Button>
              )}
            </div>
          </StepBody>
        ),
      },
    ]
  } else if (source === "ultimate") {
    const u = readiness.ultimate
    const partiallyPresent = u.pythonHid || u.apiCompatible || u.pinnedPyvut || u.udevReady
    const dongleDone = u.operatorAccess && u.dongleConnected && u.endpointStatus === "accessible"
    const identifyBlocker = !u.installed
      ? "install or repair the Linux runtime"
      : u.wifiConfig !== "valid"
        ? "save a valid private shared-map Wi-Fi configuration"
        : !u.dongleConnected
          ? "connect the paired HTC wireless dongle"
          : !u.operatorAccess
            ? "grant this operator durable dongle access"
            : u.endpointStatus !== "accessible"
              ? "reconnect the dongle and close any other process using its HID endpoint"
              : null
    steps = [
      {
        key: "runtime",
        title: "Runtime",
        done: u.installed,
        pending: "Install the pinned Ultimate runtime on this host",
        body: (
          <StepBody>
            <div className="flex flex-wrap gap-1.5">
              <ReadinessBadge tone={u.pinnedPyvut ? "ready" : "error"}>
                {u.pinnedPyvut
                  ? `Pinned ${u.pinnedRef.slice(0, 12)}`
                  : "pyvut revision unsupported"}
              </ReadinessBadge>
              <ReadinessBadge tone={u.nativeDependencies && u.pythonHid ? "ready" : "error"}>
                {u.nativeDependencies && u.pythonHid
                  ? "HID libraries ready"
                  : "HID libraries missing"}
              </ReadinessBadge>
              <ReadinessBadge tone={u.udevReady ? "ready" : "error"}>
                {u.udevReady ? "USB rule ready" : "USB rule missing"}
              </ReadinessBadge>
            </div>
            {u.issues.length > 0 && (
              <p className="text-[11px] leading-relaxed text-red-300/75">
                {u.issues.slice(0, 3).join(" · ")}
              </p>
            )}
            {!u.installed && (
              <Button
                variant="outline"
                size="sm"
                className="self-start"
                onClick={() => launch("tracker.ultimate.install")}
                disabled={!canAct}
              >
                {busy ? <Loader2 className="animate-spin" /> : <Download />}
                {partiallyPresent ? "Repair Ultimate support" : "Install Ultimate support"}
              </Button>
            )}
            {progressFor("runtime")}
          </StepBody>
        ),
      },
      {
        key: "dongle",
        title: "Dongle",
        done: dongleDone,
        pending: !u.dongleConnected
          ? "Connect the paired HTC wireless dongle"
          : !u.operatorAccess
            ? "Grant this operator dongle access, then re-login"
            : "The dongle's HID endpoint must be accessible",
        body: (
          <StepBody>
            <p>
              On a Windows PC with SteamVR (null driver), VIVE Streaming Hub, and the VIVE Ultimate
              Tracker service, pair both trackers to the wireless dongle, then create a map of the
              whole operating area from low and standing heights and confirm both trackers
              relocalize. The map stays on the trackers. Then connect that dongle to this host.
            </p>
            <div className="flex flex-wrap gap-3">
              <ExternalSetupLink href="https://business.vive.com/us/support/ultimate-tracker/category_howto/pairing-ultimate-tracker-with-the-dongle.html">
                HTC pairing guide
              </ExternalSetupLink>
              <ExternalSetupLink href="https://business.vive.com/eu/support/ultimate-tracker/category_howto/creating-a-tracking-map.html">
                HTC tracking-map guide
              </ExternalSetupLink>
            </div>
            <div className="flex flex-wrap gap-1.5">
              <ReadinessBadge tone={u.dongleConnected ? "ready" : "warning"}>
                {u.dongleConnected ? "Dongle detected" : "Dongle not detected"}
              </ReadinessBadge>
              <ReadinessBadge tone={u.operatorAccess ? "ready" : "error"}>
                {u.operatorAccess ? "Operator USB access ready" : "Operator USB access missing"}
              </ReadinessBadge>
              <ReadinessBadge
                tone={
                  u.endpointStatus === "accessible"
                    ? "ready"
                    : u.dongleConnected
                      ? "error"
                      : "neutral"
                }
              >
                {u.endpointStatus === "accessible"
                  ? "HID endpoint accessible"
                  : u.endpointStatus === "permission-denied"
                    ? "HID endpoint permission denied"
                    : u.endpointStatus === "missing"
                      ? "HID endpoint missing"
                      : "HID endpoint unavailable"}
              </ReadinessBadge>
            </div>
            {!u.operatorAccess && (
              <div className="flex flex-col gap-2 rounded-md border border-amber-400/20 bg-amber-400/[0.04] p-2.5">
                <p className="text-[11px] leading-relaxed text-amber-200/75">
                  Linux needs the login running Axol in the <code>dialout</code> group before it can
                  open the dongle; this is the one step that has to be done on the host itself. Run
                  the command there, log out and back in (or reboot), and reconnect the dongle. This
                  step turns green on its own afterwards.
                </p>
                <CopyableCommand command={'sudo usermod -aG dialout "$USER"'} />
              </div>
            )}
          </StepBody>
        ),
      },
      {
        key: "wifi",
        title: "Wi-Fi",
        done: u.wifiConfig === "valid",
        pending:
          u.wifiConfig === "permissions-warning"
            ? "Re-save the Wi-Fi configuration to fix its file permissions"
            : "Save the trackers' private shared-map Wi-Fi configuration",
        body: (
          <StepBody>
            <p>
              The trackers share their map over a private access point. Enter that AP&apos;s SSID,
              password, country, and frequency (not the robot LAN or router Wi-Fi).
            </p>
            <UltimateWifiPanel />
          </StepBody>
        ),
      },
      identifyStep("ultimate", identifyBlocker, u.installed),
    ]
  } else {
    const l = readiness.lighthouse
    const survey = l.baseStations ?? null
    const surveySupported = l.baseStations !== undefined
    const expectedStations = expectedStationCount
    const stationsMissing =
      survey !== null &&
      survey.clashingChannels.length === 0 &&
      survey.baseStationCount > 0 &&
      survey.baseStationCount < expectedStations
    const stationsDone = surveySupported
      ? survey !== null &&
        survey.clashingChannels.length === 0 &&
        survey.baseStationCount >= expectedStations &&
        survey.trackers.length >= 2
      : true
    const stationChannels = survey ? Object.keys(survey.channels) : []
    steps = [
      {
        key: "runtime",
        title: "Runtime",
        done: l.installed,
        pending: "Install Lighthouse support on this host",
        body: (
          <StepBody>
            <div className="flex flex-wrap gap-1.5">
              <ReadinessBadge tone={l.pinnedBuild ? "ready" : "error"}>
                {l.pinnedBuild
                  ? `Pinned ${l.pinnedRef.slice(0, 12)} · ${l.buildRevision}`
                  : "Pinned libsurvive build missing or stale"}
              </ReadinessBadge>
              <ReadinessBadge tone={l.udevReady ? "ready" : "error"}>
                {l.udevReady ? "Vive USB permissions ready" : "Vive USB rule missing"}
              </ReadinessBadge>
            </div>
            {l.issues.length > 0 && (
              <p className="text-[11px] leading-relaxed text-red-300/75">
                {l.issues.slice(0, 3).join(" · ")}
              </p>
            )}
            {l.installed ? (
              <p className="flex items-center gap-2 text-white/75">
                <Check className="size-4 shrink-0 text-emerald-400" /> The pinned libsurvive runtime
                and Vive USB rule are installed.
              </p>
            ) : (
              <>
                <p>
                  Installs the pinned libsurvive build and Vive USB rule on this host. An existing
                  pinned build is re-attested without rebuilding.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="self-start"
                  onClick={() => launch("tracker.install")}
                  disabled={!canAct}
                >
                  {busy ? <Loader2 className="animate-spin" /> : <Download />}
                  Install Lighthouse support
                </Button>
              </>
            )}
            {progressFor("runtime")}
          </StepBody>
        ),
      },
      {
        key: "stations",
        title: "Trackers",
        done: stationsDone,
        pending: !surveySupported
          ? ""
          : survey === null
            ? "Run Check base stations"
            : survey.clashingChannels.length > 0
              ? `Move one base station off channel ${survey.clashingChannels.join(", ")} and check again`
              : survey.baseStationCount === 0
                ? "No base station was seen; power them and check again"
                : stationsMissing
                  ? `Only ${survey.baseStationCount} of ${expectedStations} base stations seen; check power, visibility and distinct channels`
                  : "Both trackers need fresh poses; check power, dongles and visibility. Recalibrate if tracking is unstable.",
        body: (
          <StepBody>
            <ol className="flex flex-col gap-2">
              <SetupStep number={1}>
                Power all expected base stations and give each Base Station 2.0 a different channel.
                The numbers shown here come from the tracking software. Missing stations can mean
                blocked visibility, power or channel issues; the check will report a detected
                channel conflict explicitly.
              </SetupStep>
              <SetupStep number={2}>
                Pair each Tracker 3.0 to its own Watchman dongle: connect only that dongle, unplug
                the tracker&apos;s USB cable, power it on, hold its button until the LED blinks
                blue, then use Pair tracker. Repeat for the other side. Skip if both are already
                paired.
              </SetupStep>
              <SetupStep number={3}>
                Connect both dongles, power both trackers in view of the base stations, and use
                Check base stations. It listens to libsurvive for 20 seconds, or up to 150 seconds
                while libsurvive calibrates newly added or moved base stations (keep the trackers
                still), and must see every base station on its own channel.
              </SetupStep>
            </ol>
            <ExternalSetupLink href="https://www.vive.com/eu/support/vive-pro/category_howto/configuring-the-base-station-channels.html">
              HTC instructions for configuring Base Station 2.0 channels
            </ExternalSetupLink>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => launch("tracker.pair")}
                disabled={!canAct || !sourceSaved}
              >
                {busy ? <Loader2 className="animate-spin" /> : <Bluetooth />}
                Pair tracker
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  launch("tracker.lighthouse.check", { expected_stations: expectedStationCount })
                }
                disabled={!canAct}
              >
                {busy ? <Loader2 className="animate-spin" /> : <ListChecks />}
                {survey ? "Check base stations again" : "Check base stations"}
              </Button>
            </div>
            {surveySupported && survey && (
              <div className="flex flex-col gap-1.5">
                <div className="flex flex-wrap gap-1.5">
                  <ReadinessBadge
                    tone={
                      survey.clashingChannels.length > 0
                        ? "error"
                        : survey.baseStationCount === 0 || stationsMissing
                          ? "warning"
                          : "ready"
                    }
                  >
                    {survey.clashingChannels.length > 0
                      ? `Base stations share channel ${survey.clashingChannels.join(", ")}`
                      : survey.baseStationCount === 0
                        ? "No base station seen"
                        : stationsMissing
                          ? `Only ${survey.baseStationCount} of ${expectedStations} base stations seen (channel ${stationChannels.join(", ")})`
                          : `${survey.baseStationCount} base station${survey.baseStationCount === 1 ? "" : "s"} on channel${stationChannels.length === 1 ? "" : "s"} ${stationChannels.join(", ")}`}
                  </ReadinessBadge>
                  <ReadinessBadge tone={survey.trackers.length >= 2 ? "ready" : "warning"}>
                    {survey.trackers.length >= 2
                      ? `Fresh poses from ${survey.trackers.join(", ")}`
                      : survey.trackers.length === 1
                        ? `Only ${survey.trackers[0]} has a fresh pose`
                        : "No fresh tracker pose"}
                  </ReadinessBadge>
                  <ReadinessBadge tone="neutral">Checked {ago(survey.checkedAt)}</ReadinessBadge>
                </div>
                {survey.problems.length > 0 && (
                  <ul className="flex flex-col gap-1 text-[11px] leading-relaxed text-red-300/75">
                    {survey.problems.map((problem) => (
                      <li key={problem}>{problem}.</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {progressFor("stations")}
          </StepBody>
        ),
      },
      identifyStep("survive", null, l.installed),
    ]
  }

  const notices: ReactNode[] = []
  if (blockedReason && !running) {
    notices.push(
      <Notice key="blocked">Setup actions are unavailable while {blockedReason}.</Notice>
    )
  }
  if (running && sessionBackend !== backend) {
    notices.push(
      <Notice key="other">
        {sessionBackend === "ultimate" ? "Ultimate" : "Lighthouse"} setup is still running. Finish
        or stop it before working on this source.
      </Notice>
    )
  }

  return (
    <>
      {source === "lighthouse" && (
        <FlowCard title="Lighthouse tracking quality">
          {(() => {
            const lighthouse = readiness.lighthouse
            const validation = lighthouse.validation
            const expectedStations = expectedStationCount
            const expired = validation?.expiresAt != null && validation.expiresAt <= checkedAt
            const countChanged =
              validation?.expectedBaseStations != null &&
              validation.expectedBaseStations !== expectedStations
            const passed =
              !bindingsError &&
              validation?.valid &&
              validation.status === "passed" &&
              !expired &&
              !countChanged
            const stale = validation?.status === "stale" || expired || countChanged
            const supported = lighthouse.validation !== undefined
            return (
              <StepBody>
                <p>
                  Start with both rigs still on the table and their trackers visible. No fixed
                  reference fixture is needed. Follow the prompts for a 5-second hold, a 12-second
                  slow workspace sweep (at least 10 cm and 15° per rig), and a final 5-second hold.
                  Recalibration first adds three preparation holds at different working positions,
                  then restarts tracking to test the saved candidate using the same station geometry
                  settings as collection.
                </p>
                <div role="status" aria-live="polite" className="flex flex-col gap-2">
                  <ReadinessBadge
                    tone={passed ? "ready" : validation?.status === "failed" ? "error" : "warning"}
                  >
                    {bindingsError
                      ? "Validation status unavailable — reconnect to the host"
                      : passed
                        ? "Tracking check passed"
                        : validation?.status === "needs_identification"
                          ? "Calibration saved — identify sides, then validate tracking"
                          : validation?.status === "failed"
                            ? "Tracking check failed"
                            : stale
                              ? "Validation expired or setup changed — check again"
                              : "Tracking quality not checked"}
                  </ReadinessBadge>
                  {validation?.checkedAt != null && (
                    <p className="text-xs text-white/45">Checked {ago(validation.checkedAt)}</p>
                  )}
                  {validation?.problems.map((problem, index) => (
                    <p key={`${index}:${problem}`} className="text-xs text-amber-200/85">
                      {problem}
                    </p>
                  ))}
                </div>
                <label className="flex items-center gap-3 text-sm">
                  Expected base stations
                  <select
                    aria-label="Expected base stations"
                    className="rounded-md border border-white/20 bg-[#141414] px-3 py-1.5 text-white"
                    value={expectedStations}
                    onChange={(event) => {
                      const count = Number(event.target.value)
                      if (Number.isInteger(count) && count >= 1 && count <= 4)
                        setExpectedStationChoice(count)
                    }}
                    disabled={!canAct}
                  >
                    {[1, 2, 3, 4].map((count) => (
                      <option key={count} value={count}>
                        {count}
                      </option>
                    ))}
                  </select>
                </label>
                {qualityMode !== undefined && (
                  <label className="flex items-center gap-3 text-sm">
                    Tracking policy
                    <select
                      aria-label="Tracking policy"
                      className="rounded-md border border-white/20 bg-[#141414] px-3 py-1.5 text-white"
                      value={qualityMode}
                      onChange={(event) =>
                        void chooseQualityMode(event.target.value as LighthouseQualityMode)
                      }
                      disabled={!canAct}
                    >
                      <option value="standard">Standard</option>
                      <option value="strict">Strict (fixed installation)</option>
                    </select>
                  </label>
                )}
                <p className="text-xs text-white/45">
                  Select the number installed, including all three stations in a three-station
                  setup.{" "}
                  {qualityMode === "strict"
                    ? "Strict keeps the station geometry fixed between recalibrations, requires a passed validation before collecting (reports expire after 12 hours), and marks any take with a tracking fault for retake."
                    : "Standard lets libsurvive keep refining the calibration in the background; validation is optional and recording is judged by the episode quality gate."}{" "}
                  The check measures tracking consistency, not absolute accuracy. Recalibrate after
                  a station or tripod moves; the existing calibration is replaced only if the new
                  check passes.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={
                      !canAct ||
                      !sourceSaved ||
                      !lighthouse.installed ||
                      !lighthouse.binding.complete ||
                      !supported ||
                      bindingsError
                    }
                    onClick={() =>
                      launch("tracker.lighthouse.validate", {
                        expected_stations: expectedStations,
                        web_prompts: true,
                      })
                    }
                  >
                    <ListChecks /> Validate tracking
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={
                      !canAct ||
                      !sourceSaved ||
                      !lighthouse.installed ||
                      !supported ||
                      bindingsError
                    }
                    onClick={() =>
                      launch("tracker.lighthouse.calibrate", {
                        expected_stations: expectedStations,
                        web_prompts: true,
                      })
                    }
                  >
                    <RadioTower /> Recalibrate Lighthouse
                  </Button>
                </div>
                {!supported && (
                  <p className="text-xs text-amber-200/85">
                    Update Axol on the host to enable guided validation and recalibration.
                  </p>
                )}
                {!lighthouse.binding.complete && (
                  <p className="text-xs text-amber-200/85">
                    If Identify fails because tracking is unstable, run Recalibrate Lighthouse
                    first. It can recover station geometry using two detected trackers without
                    assigning sides. Then identify both rigs below and run Validate tracking before
                    recording.
                  </p>
                )}
                {blockedReason && (
                  <p className="text-xs text-amber-200/85">
                    Stop {blockedReason} before validating or recalibrating.
                  </p>
                )}
                {progressFor("validation")}
              </StepBody>
            )
          })()}
        </FlowCard>
      )}
      <SetupFlow
        key={source}
        title={`${label} setup`}
        steps={steps}
        heldStep={running && sessionBackend === backend ? sessionStep : null}
        notices={notices}
      />
    </>
  )
}

/** Header, one visible step, and Back / Continue navigation gated on `done`. */
function SetupFlow({
  title,
  steps,
  heldStep,
  notices,
}: {
  title: string
  steps: FlowStep[]
  /** A setup run owned by this step is in progress; keep the flow on it. */
  heldStep: StepKey | null
  notices: ReactNode[]
}) {
  const firstUnresolved = steps.findIndex((step) => !step.done)
  // The furthest step the operator may open: the first unresolved one, or the
  // last step once everything is confirmed. The flow ends there — once the
  // sides are identified, setup is complete.
  const reach = firstUnresolved === -1 ? steps.length - 1 : firstUnresolved
  const [chosen, setChosen] = useState(reach)
  const heldIndex = heldStep ? steps.findIndex((step) => step.key === heldStep) : -1
  const index = heldIndex >= 0 ? heldIndex : Math.min(chosen, reach)
  const step = steps[index]
  const last = index === steps.length - 1
  const complete = firstUnresolved === -1
  const resolvedCount = steps.filter((s) => s.done).length

  return (
    <FlowCard
      title={title}
      aside={
        <span className={cn("text-[11px]", complete ? "text-emerald-300/85" : "text-white/40")}>
          {complete ? "Setup complete" : `${resolvedCount}/${steps.length} steps resolved`}
        </span>
      }
    >
      <ol className="flex flex-wrap gap-1.5" aria-label="Setup steps">
        {steps.map((s, i) => {
          const active = i === index
          const reachable = i <= reach && heldIndex < 0
          return (
            <li key={s.key}>
              <button
                type="button"
                onClick={() => reachable && setChosen(i)}
                disabled={!reachable}
                aria-current={active ? "step" : undefined}
                className={cn(
                  "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] transition-colors",
                  active
                    ? "border-[#eff483]/40 bg-[#eff483]/10 text-[#eff483]"
                    : s.done
                      ? "border-emerald-400/20 bg-emerald-400/[0.06] text-emerald-300/85 hover:bg-emerald-400/10"
                      : reachable
                        ? "border-white/15 text-white/60 hover:bg-white/[0.05]"
                        : "border-white/[0.07] text-white/30"
                )}
              >
                <span
                  className={cn(
                    "flex size-4 items-center justify-center rounded-full font-mono text-[10px]",
                    s.done ? "bg-emerald-400/15" : active ? "bg-[#eff483]/15" : "bg-white/[0.07]"
                  )}
                >
                  {s.done ? <Check className="size-3" /> : i + 1}
                </span>
                {s.title}
              </button>
            </li>
          )
        })}
      </ol>

      {notices}

      <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-black/10 p-4">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-widest text-white/35 uppercase">
            Step {index + 1} of {steps.length}
          </span>
          <span className="text-sm font-medium text-white/85">{step.title}</span>
          {step.done && (
            <span className="ml-auto flex items-center gap-1 text-[11px] text-emerald-300/85">
              <Check className="size-3" /> Resolved
            </span>
          )}
        </div>
        {step.body}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setChosen(index - 1)}
          disabled={index === 0 || heldIndex >= 0}
        >
          <ChevronLeft /> Back
        </Button>
        {!step.done && step.pending && (
          <span className="ml-auto text-[11px] text-amber-300/80">{step.pending}</span>
        )}
        {last ? (
          step.done && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-300/85">
              <Check className="size-3.5" /> {title} is complete on this host.
            </span>
          )
        ) : (
          <Button
            size="sm"
            className={step.done ? "ml-auto" : ""}
            onClick={() => setChosen(index + 1)}
            disabled={!step.done || heldIndex >= 0}
          >
            Continue <ChevronRight />
          </Button>
        )}
      </div>
    </FlowCard>
  )
}

function FlowCard({
  title,
  aside,
  children,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-white/80">{title}</span>
        <span className="ml-auto">{aside}</span>
      </div>
      {children}
    </div>
  )
}

function StepBody({ children }: { children: ReactNode }) {
  return <div className="flex flex-col gap-3 text-xs leading-relaxed text-white/50">{children}</div>
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs leading-relaxed text-amber-200/80">
      {children}
    </p>
  )
}

function ReadinessBadge({
  tone,
  children,
}: {
  tone: "ready" | "warning" | "error" | "neutral"
  children: ReactNode
}) {
  const toneClass =
    tone === "ready"
      ? "bg-emerald-400/10 text-emerald-300"
      : tone === "warning"
        ? "bg-amber-400/10 text-amber-300"
        : tone === "error"
          ? "bg-red-400/10 text-red-300"
          : "bg-white/[0.06] text-white/50"
  return <span className={`rounded-full px-2 py-0.5 text-[11px] ${toneClass}`}>{children}</span>
}

function SetupStep({ number, children }: { number: number; children: ReactNode }) {
  return (
    <li className="flex items-start gap-3 text-xs leading-relaxed text-white/50">
      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-white/[0.07] font-mono text-[10px] text-white/60">
        {number}
      </span>
      <div className="min-w-0 pt-0.5">{children}</div>
    </li>
  )
}

function ExternalSetupLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 text-[#eff483]/70 underline decoration-[#eff483]/25 underline-offset-2 hover:text-[#eff483]"
    >
      {children} <ExternalLink className="size-3" />
    </a>
  )
}

function CopyableCommand({ command }: { command: string }) {
  const toast = useToast()

  async function copy() {
    try {
      await navigator.clipboard.writeText(command)
      toast.success("Copied.")
    } catch {
      toast.error("Could not copy automatically; select the text instead.")
    }
  }

  return (
    <div className="flex max-w-md items-center gap-2 rounded-md bg-black/25 px-2.5 py-1.5">
      <code className="min-w-0 flex-1 select-all overflow-x-auto text-[11px] whitespace-nowrap text-white/65">
        {command}
      </code>
      <button
        type="button"
        onClick={copy}
        className="flex shrink-0 items-center gap-1 text-[11px] text-white/40 hover:text-white/75"
        aria-label={`Copy ${command}`}
      >
        <Clipboard className="size-3" /> Copy
      </button>
    </div>
  )
}
