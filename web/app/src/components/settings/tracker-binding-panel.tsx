import { useEffect, useRef, useState, type ReactNode } from "react"
import {
  Bluetooth,
  Check,
  Clipboard,
  Download,
  ExternalLink,
  ListChecks,
  Loader2,
  RadioTower,
  Square,
  Wrench,
} from "lucide-react"
import {
  fetchTrackerBindings,
  sendSessionInput,
  stopSession,
  useSessionLogs,
  type MantisTrackerSource,
  type SessionInfo,
  type TrackerBackend,
  type TrackerBinding,
  type TrackerSourceReadiness,
} from "@/lib/supervisor"
import { startDiagnosticsRun } from "@/lib/telemetry"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/toast"
import { TrackerCalibrationPanel } from "./tracker-calibration-panel"
import { UltimateWifiPanel } from "./ultimate-wifi-panel"

function backendFor(source: string): TrackerBackend | null {
  if (source === "lighthouse") return "survive"
  if (source === "ultimate") return "ultimate"
  return null
}

function trackerSessionBackend(session: SessionInfo): TrackerBackend | null {
  if (session.command === "tracker.install" || session.command === "tracker.pair") return "survive"
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

function latestPrompt(lines: string[]): { index: number; text: string } | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]
    if (line.startsWith("[prompt] ")) {
      return { index: i, text: line.slice("[prompt] ".length).trim() }
    }
  }
  return null
}

/** Backend installation and guided left/right binding in Settings → Mantis. */
export function TrackerBindingPanel({
  source,
  sourceSaved,
  calibrationContextSaved,
  onQuestKeySelect,
  blockedReason,
  hostSession,
  onHostSessionChange,
}: {
  source: string
  /** False when the source selector is only a local, unsaved draft. */
  sourceSaved: boolean
  /** Source plus profile-scoped Quest datum match the persisted host settings. */
  calibrationContextSaved: boolean
  /** Fill the shared Quest datum field; the enclosing Settings save persists it. */
  onQuestKeySelect?: (key: string) => void
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
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [busy, setBusy] = useState(false)
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

  useEffect(() => {
    fetchTrackerBindings()
      .then(({ bindings: found, sources }) => {
        setBindings(found)
        setReadiness(sources ?? null)
        setBindingsLoaded(true)
        setBindingsError(false)
      })
      .catch(() => {
        setBindingsLoaded(true)
        setBindingsError(true)
      })
  }, [])

  // Keep runtime, binding, transform-file, and (for Quest) live WebXR datum
  // readiness current while Settings is open. Install/Identify runs and edits
  // to the calibration file happen outside React state, so a one-shot fetch
  // otherwise leaves an already-fixed setup looking blocked until reload.
  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      fetchTrackerBindings()
        .then(({ bindings: found, sources }) => {
          if (cancelled) return
          setBindings(found)
          setReadiness(sources ?? null)
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

  // useSessionLogs clears asynchronously when its session id changes. Hide
  // the previous run's lines during that one render rather than presenting an
  // old prompt under the newly selected tracker source.
  const currentLines = !status || status.id === activeSession?.id ? lines : []
  const prompt = latestPrompt(currentLines)
  const pendingPrompt =
    prompt &&
    !(dismissed && dismissed.id === activeSession?.id && dismissed.promptIndex === prompt.index)
      ? prompt.text
      : null
  const transcriptLimit = activeSession?.command === "tracker.ultimate.check" ? 24 : 6
  const transcript = currentLines
    .filter((line) => line.trim() && !line.startsWith("[serve]") && !line.startsWith("[prompt] "))
    .slice(-transcriptLimit)
  const activeLine = transcript.at(-1) ?? null
  // useSessionLogs can also retain the previous session's terminal status for
  // one render. Never let it complete a new run before that run has started.
  const current = status?.id === activeSession?.id ? status : activeSession
  const running = current?.status === "starting" || current?.status === "running"
  const terminal = current?.status === "exited" || current?.status === "error"
  const pairing = activeSession?.command === "tracker.pair"
  const installing =
    activeSession?.command === "tracker.install" ||
    activeSession?.command === "tracker.ultimate.install"
  const checkingUltimate = activeSession?.command === "tracker.ultimate.check"

  useEffect(() => {
    if (!terminal || !activeSession || completionRef.current === activeSession.id) return
    completionRef.current = activeSession.id
    const paired = activeSession.command === "tracker.pair"
    const installed =
      activeSession.command === "tracker.install" ||
      activeSession.command === "tracker.ultimate.install"
    const checkedUltimate = activeSession.command === "tracker.ultimate.check"
    fetchTrackerBindings()
      .then(({ bindings: found, sources }) => {
        setBindings(found)
        setReadiness(sources ?? null)
        setBindingsLoaded(true)
        setBindingsError(false)
      })
      .catch(() => setBindingsError(true))
    if (current?.status === "exited" && (current.exitCode ?? 0) === 0) {
      toast.success(
        checkedUltimate
          ? "Ultimate host setup is ready."
          : installed
            ? displayedBackend === "ultimate"
              ? "Ultimate Linux runtime installed. Configure shared-map Wi-Fi and connect the paired dongle next."
              : "Lighthouse tracking support installed. Pair and identify both trackers next."
            : paired
              ? "Lighthouse tracker paired."
              : "Mantis tracker binding saved."
      )
    } else {
      toast.error(
        checkedUltimate
          ? "Ultimate readiness check found setup issues. Review the results below."
          : installed
            ? "Tracker runtime installation failed. See the status below."
            : paired
              ? "Tracker pairing failed. See the status below."
              : "Tracker identification failed. See the status below."
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

  async function pair() {
    if (!sourceSaved) return
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const { session: started } = await startDiagnosticsRun("tracker.pair", {})
      setSession(started)
      onHostSessionChange(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function install() {
    if (backend === null) return
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const command = backend === "ultimate" ? "tracker.ultimate.install" : "tracker.install"
      const { session: started } = await startDiagnosticsRun(command, {})
      setSession(started)
      onHostSessionChange(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function identify() {
    if (!sourceSaved || backend === null) return
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const { session: started } = await startDiagnosticsRun("tracker.identify", {
        backend,
        web_prompts: true,
      })
      setSession(started)
      onHostSessionChange(started)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setBusy(false)
    }
  }

  async function checkUltimate() {
    setBusy(true)
    setDismissed(null)
    completionRef.current = null
    try {
      const { session: started } = await startDiagnosticsRun("tracker.ultimate.check", {})
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

  const binding = displayedBackend === null ? null : bindings[displayedBackend]
  const label = displayedBackend === "ultimate" ? "Ultimate" : "Lighthouse"
  const runtimeMissing =
    readiness !== null &&
    displayedBackend !== null &&
    !(displayedBackend === "ultimate"
      ? readiness.ultimate.installed
      : readiness.lighthouse.installed)
  const runtimeChecking = readiness === null
  const ultimateStatus = readiness?.ultimate ?? null
  const ultimateRuntimePartiallyPresent = Boolean(
    ultimateStatus &&
    (ultimateStatus.pythonHid ||
      ultimateStatus.apiCompatible ||
      ultimateStatus.pinnedPyvut ||
      ultimateStatus.udevReady)
  )
  const ultimateIdentifyBlocker =
    backend !== "ultimate" || !ultimateStatus
      ? null
      : !ultimateStatus.installed
        ? "install or repair the Linux runtime"
        : ultimateStatus.wifiConfig !== "valid"
          ? "save a valid private shared-map Wi-Fi configuration"
          : !ultimateStatus.dongleConnected
            ? "connect the paired HTC wireless dongle"
            : !ultimateStatus.operatorAccess
              ? "grant this operator durable dongle access"
              : ultimateStatus.endpointStatus !== "accessible"
                ? "reconnect the dongle and close any other process using its HID endpoint"
                : null
  const bindingStatus = binding?.complete
    ? "Left + right bound"
    : !bindingsLoaded
      ? "Checking binding…"
      : bindingsError
        ? "Status unavailable"
        : "Not configured"

  return (
    <div className="flex flex-col gap-3">
      <SourceChecklist source={source} readiness={readiness} onQuestKeySelect={onQuestKeySelect} />

      {source === "ultimate" && <UltimateWifiPanel />}

      {!sourceSaved && (
        <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs leading-relaxed text-amber-200/80">
          Use Save at the bottom of Settings before{" "}
          {backend === null ? "starting a Quest Mantis run" : "running Pair or Identify"}. This
          keeps the saved source used by a run in sync with the setup shown here.
        </p>
      )}

      {blockedReason && !running && (
        <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs leading-relaxed text-amber-200/80">
          Setup actions are unavailable while {blockedReason}.
        </p>
      )}

      {running && sessionBackend !== backend && (
        <p className="rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs leading-relaxed text-amber-200/80">
          {label} setup is still running. The edited source applies to the next run; finish or Stop
          this setup session first.
        </p>
      )}

      {displayedBackend === null ? (
        <div className="rounded-lg border border-white/10 bg-white/[0.02] p-4">
          <div className="flex items-center gap-2 text-sm text-white/75">
            <Check className="size-4 text-emerald-400" />
            No manual left/right binding is needed; WebXR supplies controller handedness.
          </div>
          <p className="mt-2 max-w-prose text-xs leading-relaxed text-white/40">
            This confirms the setup method only. Headset connection and live controller poses are
            established after a Quest Mantis run starts.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4">
          <div className="flex flex-wrap items-center gap-2">
            <RadioTower className="size-4 text-white/50" />
            <span className="text-sm font-medium">{label} tracker binding</span>
            <span
              className={
                binding?.complete
                  ? "rounded-full bg-emerald-400/10 px-2 py-0.5 text-[11px] text-emerald-300"
                  : bindingsError
                    ? "rounded-full bg-white/[0.06] px-2 py-0.5 text-[11px] text-white/45"
                    : "rounded-full bg-amber-400/10 px-2 py-0.5 text-[11px] text-amber-300"
              }
            >
              {bindingStatus}
            </span>
            {!running && (sessionBackend === null || sessionBackend === backend) && (
              <div className="ml-auto flex flex-wrap gap-2">
                {runtimeChecking ? (
                  <Button variant="outline" size="sm" disabled>
                    <Loader2 className="animate-spin" />
                    Checking runtime…
                  </Button>
                ) : runtimeMissing ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={install}
                    disabled={busy || blockedReason !== null}
                  >
                    {busy ? <Loader2 className="animate-spin" /> : <Download />}
                    {displayedBackend === "ultimate" && ultimateRuntimePartiallyPresent
                      ? "Repair Ultimate support"
                      : `Install ${label} support`}
                  </Button>
                ) : (
                  backend === "survive" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={pair}
                      disabled={busy || !sourceSaved || blockedReason !== null}
                    >
                      {busy ? <Loader2 className="animate-spin" /> : <Bluetooth />}
                      Pair tracker
                    </Button>
                  )
                )}
                {!runtimeChecking && !runtimeMissing && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={identify}
                    disabled={
                      busy ||
                      !sourceSaved ||
                      blockedReason !== null ||
                      ultimateIdentifyBlocker !== null
                    }
                  >
                    {busy ? <Loader2 className="animate-spin" /> : <RadioTower />}
                    {binding?.complete ? "Identify again" : "Identify trackers"}
                  </Button>
                )}
                {backend === "ultimate" && !runtimeChecking && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={checkUltimate}
                    disabled={busy || blockedReason !== null}
                  >
                    {busy ? <Loader2 className="animate-spin" /> : <ListChecks />}
                    Run readiness check
                  </Button>
                )}
              </div>
            )}
          </div>
          <p className="max-w-prose text-xs leading-relaxed text-white/40">
            {backend === "survive"
              ? "Pair each tracker with its intended Watchman dongle, then identify which Mantis it is mounted to. These steps run on this host without SteamVR."
              : "The tracker addresses do not identify a rig side. This guided check watches which tracker moves for each side and saves that mapping on this host."}
          </p>
          {backend === "ultimate" && ultimateIdentifyBlocker && !runtimeChecking && !running && (
            <p className="text-xs leading-relaxed text-amber-300/80">
              Identify is waiting: {ultimateIdentifyBlocker}. The readiness check reports every
              remaining host prerequisite.
            </p>
          )}
          {backend === "ultimate" && ultimateStatus && !ultimateStatus.operatorAccess && (
            <div className="flex flex-col gap-2 rounded-md border border-amber-400/20 bg-amber-400/[0.04] p-2.5">
              <p className="text-[11px] leading-relaxed text-amber-200/75">
                Grant the login running Axol durable HID access on the host, then log out and back
                in (or reboot) and reconnect the dongle. Runtime Repair does not change group
                membership.
              </p>
              <CopyableCommand command={'sudo usermod -aG dialout "$USER"'} />
            </div>
          )}
          {binding?.complete && !running && !terminal && (
            <div className="flex flex-col gap-1">
              <p className="font-mono text-[11px] text-white/35">
                Left {binding.left} · Right {binding.right}
              </p>
              <p className="text-[11px] text-white/30">
                Saved binding only — use the test step above to verify both trackers are live.
              </p>
            </div>
          )}
          {(running || (terminal && checkingUltimate)) && (
            <div className="flex flex-col gap-2 rounded-md border border-white/10 bg-black/20 p-3">
              {transcript.length > 0 && (
                <div
                  className="flex max-h-32 flex-col gap-1 overflow-y-auto font-mono text-[11px] leading-relaxed text-white/45"
                  aria-live="polite"
                >
                  {transcript.map((line, index) => (
                    <p key={`${index}:${line}`}>{line}</p>
                  ))}
                </div>
              )}
              {running && (
                <>
                  {pendingPrompt && !pairing && !installing && !checkingUltimate ? (
                    <>
                      <p className="text-sm text-amber-100/85">{pendingPrompt}</p>
                      <Button size="sm" className="self-start" onClick={capture}>
                        Start 3-second capture
                      </Button>
                    </>
                  ) : dismissed?.id === activeSession?.id ? (
                    <p className="flex items-center gap-2 text-sm text-white/60">
                      <Loader2 className="size-4 animate-spin" /> Capturing motion…
                    </p>
                  ) : (
                    <p className="flex items-center gap-2 text-sm text-white/60">
                      <Loader2 className="size-4 animate-spin" />
                      {installing
                        ? `Installing ${label} tracking support…`
                        : checkingUltimate
                          ? "Running the non-invasive Ultimate readiness check…"
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
                    disabled={busy}
                  >
                    <Square /> Stop
                  </Button>
                </>
              )}
            </div>
          )}
          {terminal && !running && (
            <p
              className={
                current?.status === "exited" && (current.exitCode ?? 0) === 0
                  ? "text-xs text-emerald-300/80"
                  : "text-xs text-red-300/80"
              }
            >
              {current?.status === "exited" && (current.exitCode ?? 0) === 0
                ? checkingUltimate
                  ? "Ultimate host setup passed. Identify is the separate live-pose test."
                  : installing
                    ? displayedBackend === "ultimate"
                      ? "Ultimate Linux support installed. Save shared-map Wi-Fi, connect the paired dongle, then identify both sides."
                      : "Lighthouse support installed. Pair and identify both trackers next."
                    : pairing
                      ? "Tracker paired. Repeat for the other tracker if needed, then identify both sides."
                      : "Binding saved. Complete the test step above before collecting data."
                : (activeLine ??
                  current?.error ??
                  (checkingUltimate
                    ? "Ultimate readiness check found setup issues."
                    : installing
                      ? "Tracker runtime installation failed."
                      : pairing
                        ? "Tracker pairing failed."
                        : "Tracker identification failed."))}
            </p>
          )}
        </div>
      )}

      {(source === "quest" || source === "lighthouse" || source === "ultimate") && (
        <TrackerCalibrationPanel
          key={`${source}:${calibrationContextSaved ? "saved" : "draft"}:${binding?.left ?? ""}:${binding?.right ?? ""}:${readiness?.quest.calibrationKey ?? ""}`}
          source={source as MantisTrackerSource}
          contextSaved={calibrationContextSaved}
        />
      )}

      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4">
        <Wrench className="size-4 text-white/50" />
        <div className="min-w-48 flex-1">
          <p className="text-sm font-medium text-white/75">Test triggers and grippers first</p>
          <p className="mt-1 text-xs leading-relaxed text-white/40">
            This diagnostic needs neither cameras nor a working tracker setup.
          </p>
        </div>
        <a
          href="/diagnostics"
          className="inline-flex h-8 items-center justify-center gap-2 rounded-lg border border-white/10 px-3 text-[0.8rem] font-medium text-white/80 transition-colors hover:bg-white/[0.06]"
        >
          Open Diagnostics <ExternalLink className="size-3.5" />
        </a>
      </div>
    </div>
  )
}

function SourceChecklist({
  source,
  readiness,
  onQuestKeySelect,
}: {
  source: string
  readiness: TrackerSourceReadiness | null
  onQuestKeySelect?: (key: string) => void
}) {
  const toast = useToast()
  if (source === "quest") {
    return (
      <Checklist
        title="Quest / WebXR setup"
        status={<SourceReadinessBadges source="quest" readiness={readiness} />}
      >
        <SetupStep number={1}>
          Put the Quest and this host on the same LAN. For lower-latency controller poses, enable
          Quest Developer Mode, attach USB-C, accept the USB-debugging prompt in-headset, then use
          the Quest USB tab. USB carries poses only; the page and camera video still use the LAN.
        </SetupStep>
        <SetupStep number={2}>
          Save this source, connect the Mantis hardware tile, and start a Mantis teleop or data
          collection run.
        </SetupStep>
        <SetupStep number={3}>
          In the Quest browser open <span className="text-white/70">axol.almond.bot</span>, enter
          this host, and connect. If prompted, choose Authorize certificate, approve the local
          certificate page, then return and reconnect.
        </SetupStep>
        <SetupStep number={4}>
          Choose Enter VR and allow the immersive session. Hold both Touch controllers; WebXR
          supplies left/right handedness automatically, so no Identify step is needed. The current
          client also reports each controller&apos;s WebXR profile and whether its pose came from
          gripSpace or the incompatible target-ray fallback. At the start pose, release both side
          grip buttons and then press them together to align and engage. Repeat that complete
          release→together-press after Reset, an occlusion, or SLAM recovery.
          {readiness?.quest.liveDatum && (
            <div className="mt-2 flex flex-col gap-2 rounded-md border border-white/10 bg-black/20 p-2">
              <span className="text-[11px] text-white/55">
                {readiness.quest.liveDatum.live ? "Live" : "Last reported"} left:{" "}
                {readiness.quest.liveDatum.left.profile ?? "profile missing"} ·{" "}
                {readiness.quest.liveDatum.left.poseSpace ?? "space missing"}; right:{" "}
                {readiness.quest.liveDatum.right.profile ?? "profile missing"} ·{" "}
                {readiness.quest.liveDatum.right.poseSpace ?? "space missing"}
              </span>
              {readiness.quest.liveDatum.live && readiness.quest.liveDatum.commonKey ? (
                <div className="flex flex-wrap items-center gap-2">
                  <CopyableCommand command={readiness.quest.liveDatum.commonKey} />
                  {onQuestKeySelect && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        onQuestKeySelect(readiness.quest.liveDatum!.commonKey!)
                        toast.success("Quest calibration key filled. Save settings to apply it.")
                      }}
                    >
                      <Check /> Use this key
                    </Button>
                  )}
                </div>
              ) : !readiness.quest.liveDatum.live ? (
                <span className="text-[11px] text-amber-300/80">
                  This report is {readiness.quest.liveDatum.ageSeconds.toFixed(1)} s old. Reconnect
                  or resume the headset before copying a calibration key.
                </span>
              ) : readiness.quest.liveDatum.left.poseSpace === "target-ray" ||
                readiness.quest.liveDatum.right.poseSpace === "target-ray" ? (
                <span className="text-[11px] text-red-300/85">
                  This WebXR runtime only supplied target-ray poses. Mantis requires gripSpace;
                  update/restart the Quest browser and controllers before calibrating or collecting.
                </span>
              ) : (
                <span className="text-[11px] text-amber-300/80">
                  The two sides do not report one matching calibration datum; do not collect.
                </span>
              )}
              {readiness.quest.liveDatum.live &&
                readiness.quest.liveDatum.commonKey &&
                readiness.quest.calibrationKey &&
                readiness.quest.liveDatum.commonKey !== readiness.quest.calibrationKey && (
                  <span className="text-[11px] text-red-300/85">
                    The live controller datum does not match the saved Quest calibration key.
                  </span>
                )}
            </div>
          )}
        </SetupStep>
        <SetupStep number={5}>
          For production data, enter both bench measurements in the tracker → gripper calibration
          editor below. They are saved under the exact reported
          <code className="text-white/65"> quest:&lt;profile&gt;:grip</code> datum. The host refuses
          a different Touch generation, missing profile, or target-ray datum instead of silently
          applying the wrong offset. Teleop bring-up can run without this calibration, but
          collection remains blocked unless explicitly marked uncalibrated.
        </SetupStep>
      </Checklist>
    )
  }

  if (source === "ultimate") {
    return (
      <Checklist
        title="VIVE Ultimate / Windows SLAM setup"
        status={<SourceReadinessBadges source="ultimate" readiness={readiness} />}
      >
        <SetupStep number={1}>
          On Windows, first install SteamVR, enable its null/virtual-headset driver, install VIVE
          Streaming Hub, and install the VIVE Ultimate Tracker service from VIVE Hub. Those
          prerequisites are required to create the tracker map even though this Mantis flow uses no
          headset. Connect the wireless dongle and open Settings → VIVE Ultimate Tracker → Trackers
          → Pair new. After powering on each tracker, wait through its green/blue startup flashes
          until the LED is solid blue; only then hold Power for about two seconds until blue flashes
          again and pair it. Pair both trackers to this dongle.
          <div className="mt-2">
            <ExternalSetupLink href="https://business.vive.com/us/support/ultimate-tracker/category_howto/pairing-ultimate-tracker-with-the-dongle.html">
              HTC pairing guide
            </ExternalSetupLink>
          </div>
        </SetupStep>
        <SetupStep number={2}>
          In VIVE Hub choose Start setup → Create map. Establish the center, then scan the complete
          operating area from low/kneeling and standing heights while facing all four directions.
          Let the map auto-save, refine/save it if offered, and confirm both trackers relocalize.
          The map remains on the trackers; only the final physical SteamVR-headset connection step
          is skipped—not the Windows software/service prerequisites above.
          <div className="mt-2 flex flex-wrap gap-3">
            <ExternalSetupLink href="https://business.vive.com/eu/support/ultimate-tracker/category_howto/creating-a-tracking-map.html">
              HTC tracking-map guide
            </ExternalSetupLink>
            <ExternalSetupLink href="https://github.com/nijkah/pyvut/blob/fcfcd33f4c1f16b0d84f5f741dc1319abdc7942a/README.md">
              Linux no-headset notes
            </ExternalSetupLink>
          </div>
        </SetupStep>
        <SetupStep number={3}>
          On this Linux/Jetson host, use Install or Repair below for the pinned runtime and USB
          rule, then fill in pyvut&apos;s protected shared-map AP configuration/fallback below. It
          is not router Wi-Fi, and dongle firmware may supply the active host credentials. The UI
          writes the file with mode 0600, never reads the saved password back, and preserves it when
          the password field is blank.
        </SetupStep>
        <SetupStep number={4}>
          Connect the HTC wireless dongle, power both trackers in the mapped area, and let their
          inside-out tracking converge. Use Identify trackers below as the live pose test, moving
          only the requested rig during each capture.
        </SetupStep>
        <SetupStep number={5}>
          Run the final readiness check after binding, then test with a short Mantis teleop run.
          Hold both rigs at the configured rest pose; when both trackers and trigger channels are
          live, release both triggers, then squeeze them together to align and engage. Before
          re-engaging after tracking loss or Reset, restore both inputs and repeat that full
          release→together-squeeze gesture. Before production collection, enter each bench-verified
          mount transform in the calibration editor below; it is attached to the Ultimate MAC
          identified for that side. A connected Quest can still show cameras and the recording HUD,
          but its SLAM world is unrelated to the Ultimate map. The 3D robot overlay stays hidden
          unless those origins and yaw have been explicitly co-registered; do not use an
          unregistered overlay to approve a mount transform.
        </SetupStep>
      </Checklist>
    )
  }

  return (
    <Checklist
      title="Lighthouse / Tracker 3.0 setup"
      status={<SourceReadinessBadges source="lighthouse" readiness={readiness} />}
    >
      <SetupStep number={1}>
        Provision this Linux host with pinned libsurvive and Vive USB permissions.
        <div className="mt-2">
          <CopyableCommand command="axol provision" />
        </div>
      </SetupStep>
      <SetupStep number={2}>
        Power the Lighthouse base stations, connect the Watchman dongles, and place both Tracker 3.0
        units where they can see the base stations.
      </SetupStep>
      <SetupStep number={3}>
        Pair each tracker with its intended dongle: connect only that dongle, unplug the tracker USB
        cable, power it on, hold its button until the LED blinks blue, then use Pair tracker below.
        Repeat for the other side.
      </SetupStep>
      <SetupStep number={4}>
        Power both paired trackers and use Identify trackers. Move only the requested Mantis during
        each three-second capture.
      </SetupStep>
      <SetupStep number={5}>
        Test with a short Mantis teleop run: leave both trackers visible and hold both rigs at the
        configured rest pose. Once both trackers and trigger channels are live, release both
        triggers, then squeeze them together to confirm the start-pose alignment and engage. Before
        re-engaging after occlusion or Reset, restore both trackers and repeat the full
        release→together-squeeze gesture. Before production collection, enter each bench-verified
        mount transform in the calibration editor below; it is attached to the libsurvive tracker ID
        identified for that side. A connected Quest can still show cameras and the recording HUD,
        but its local-floor world is unrelated to Lighthouse. The 3D robot overlay stays hidden
        unless the two worlds have been explicitly co-registered; do not use an unregistered overlay
        to approve a transform.
      </SetupStep>
    </Checklist>
  )
}

function SourceReadinessBadges({
  source,
  readiness,
}: {
  source: "quest" | "lighthouse" | "ultimate"
  readiness: TrackerSourceReadiness | null
}) {
  if (!readiness) return null
  const transforms = readiness[source].transforms
  const runtimeIssues = source === "quest" ? [] : readiness[source].issues
  const missingTransforms = (["left", "right"] as const).filter(
    (side) =>
      transforms[side] === "missing" ||
      transforms[side] === "candidate" ||
      transforms[side] === "stale"
  )

  let sourceBadges: ReactNode
  if (source === "quest") {
    const status = readiness.quest
    sourceBadges = (
      <>
        <ReadinessBadge tone="ready">WebXR built in</ReadinessBadge>
        <ReadinessBadge
          tone={
            status.datumStatus === "configured"
              ? status.poseSpace === "grip"
                ? "ready"
                : "warning"
              : status.datumStatus === "ambiguous"
                ? "error"
                : "warning"
          }
        >
          {status.datumStatus === "configured"
            ? `${status.controllerProfile} · ${status.poseSpace}`
            : status.datumStatus === "ambiguous"
              ? "Multiple Quest calibrations — enter one above"
              : status.datumStatus === "invalid"
                ? "Quest calibration key is invalid"
                : "Quest controller datum missing"}
        </ReadinessBadge>
      </>
    )
  } else if (source === "lighthouse") {
    const status = readiness.lighthouse
    sourceBadges = (
      <>
        <ReadinessBadge tone={status.installed ? "ready" : "error"}>
          {status.installed ? "Pinned libsurvive ready" : "Lighthouse runtime incomplete"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.pinnedBuild ? "ready" : "error"}>
          {status.pinnedBuild
            ? `Pinned ${status.pinnedRef.slice(0, 12)} · ${status.buildRevision}`
            : "Build stamp stale"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.udevReady ? "ready" : "error"}>
          {status.udevReady ? "Vive USB permissions ready" : "Vive USB rule missing"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.binding.complete ? "ready" : "warning"}>
          {status.binding.complete ? "Left + right bound" : "Binding incomplete"}
        </ReadinessBadge>
      </>
    )
  } else {
    const status = readiness.ultimate
    const wifiBadge = {
      valid: { tone: "ready" as const, label: "Wi-Fi config protected" },
      missing: { tone: "warning" as const, label: "Wi-Fi config missing" },
      invalid: { tone: "error" as const, label: "Wi-Fi config invalid" },
      "permissions-warning": {
        tone: "warning" as const,
        label: "Wi-Fi config needs chmod 600",
      },
    }[status.wifiConfig]
    sourceBadges = (
      <>
        <ReadinessBadge tone={status.installed ? "ready" : "error"}>
          {status.installed ? "Pinned Linux runtime ready" : "Linux runtime incomplete"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.pinnedPyvut ? "ready" : "error"}>
          {status.pinnedPyvut
            ? `Pinned ${status.pinnedRef.slice(0, 12)}`
            : "pyvut revision unsupported"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.udevReady ? "ready" : "error"}>
          {status.udevReady ? "USB rule ready" : "USB rule missing"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.operatorAccess ? "ready" : "error"}>
          {status.operatorAccess ? "Operator USB access ready" : "Operator USB access missing"}
        </ReadinessBadge>
        <ReadinessBadge tone={status.binding.complete ? "ready" : "warning"}>
          {status.binding.complete ? "Left + right bound" : "Binding incomplete"}
        </ReadinessBadge>
        <ReadinessBadge
          tone={
            status.dongleConnected && status.endpointStatus === "accessible" ? "ready" : "warning"
          }
        >
          {status.dongleConnected
            ? status.endpointStatus === "accessible"
              ? "Dongle endpoint accessible"
              : "Dongle endpoint unavailable"
            : "Dongle not detected"}
        </ReadinessBadge>
        <ReadinessBadge tone={wifiBadge.tone}>{wifiBadge.label}</ReadinessBadge>
        <ReadinessBadge tone="neutral">
          {status.quatOrder} quaternion · {status.upAxis}-up
        </ReadinessBadge>
      </>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-1.5">
        {sourceBadges}
        {(["left", "right"] as const).map((side) => {
          const transform = transforms[side]
          return (
            <ReadinessBadge
              key={side}
              tone={
                transform === "missing"
                  ? "error"
                  : transform === "candidate" || transform === "stale"
                    ? "warning"
                    : transform === "measured"
                      ? "ready"
                      : "neutral"
              }
            >
              {side === "left" ? "L" : "R"} mount {transform}
            </ReadinessBadge>
          )
        })}
      </div>
      {runtimeIssues.length > 0 && (
        <p className="text-[11px] leading-relaxed text-red-300/75">
          {runtimeIssues.slice(0, 2).join(" · ")}
        </p>
      )}
      {missingTransforms.length > 0 && (
        <p className="text-[11px] leading-relaxed text-red-300/75">
          {missingTransforms.map((side) => side[0].toUpperCase()).join(" + ")} mount transform
          {missingTransforms.length === 1 ? " is" : "s are"}{" "}
          {missingTransforms.some((side) => transforms[side] === "stale")
            ? "stale for the active pose convention"
            : missingTransforms.some((side) => transforms[side] === "candidate")
              ? "not bench-verified"
              : "missing"}
          . Teleop bring-up can warn and continue, but production data collection requires a
          measured or verified factory transform for both sides.
        </p>
      )}
    </div>
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

function Checklist({
  title,
  status,
  children,
}: {
  title: string
  status?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4">
      <span className="text-sm font-medium text-white/80">{title}</span>
      {status}
      <ol className="flex flex-col gap-3">{children}</ol>
    </div>
  )
}

function SetupStep({ number, children }: { number: number; children: ReactNode }) {
  return (
    <li className="flex items-start gap-3 text-xs leading-relaxed text-white/45">
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
      toast.success("Command copied.")
    } catch {
      toast.error("Could not copy automatically; select the command text instead.")
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
