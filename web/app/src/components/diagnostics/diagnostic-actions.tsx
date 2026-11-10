import { useEffect, useMemo, useRef, useState } from "react"
import { ArrowRight, Hand, Loader2, Play, Square, Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { CuratedForm } from "@/components/config-form"
import type { FieldSuggestion } from "@/components/suggest-input"
import { cn } from "@/lib/utils"
import {
  computeArgs,
  fetchFieldSuggestions,
  flattenFields,
  missingRequired,
  uploadFirmware,
  type CommandSpec,
  type FirmwareBoard,
  type FirmwareUpload,
  type FormValue,
} from "@/lib/supervisor"
import { JOINTS, JOINT_COLORS, jointLabel, type JointName } from "@/lib/telemetry"
import { parsePrompt } from "@/lib/prompt"

// Flags handled by the dashboard itself, never shown in a dialog:
// - web_prompts: forced on so hands-on steps drive a Continue button.
// - no_capture: dashboard runs always keep the telemetry capture for history.
const HIDDEN_FLAGS = new Set(["web_prompts", "no_capture"])

/**
 * Diagnostics as app actions: a card per test — click it, set parameters in a
 * dialog, hit Run. The card shows the running state; hands-on prompts and the
 * Stop control live in the page-level `ActiveRunPanel`; results land in run
 * history.
 */
export function DiagnosticActions({
  commands,
  activeCommand,
  activeSince,
  busy,
  disabled,
  hiddenKeys,
  pickerJoints,
  onLaunch,
  onStop,
}: {
  commands: CommandSpec[]
  /** Command id of the run in flight (any diagnostics launch), if one is. */
  activeCommand: string | null
  /** Epoch seconds the active run started, for the elapsed readout. */
  activeSince: number | null
  busy: boolean
  disabled: boolean
  /**
   * Field keys the page decides for every dialog (e.g. the CAN channel
   * arguments, injected from the configured adapter mapping at launch).
   */
  hiddenKeys?: string[]
  /** The `--joints` picker's choices (gripperless SKUs drop GRIPPER). */
  pickerJoints?: readonly JointName[]
  onLaunch: (command: string, args: Record<string, FormValue>) => void
  onStop: () => void
}) {
  const [openId, setOpenId] = useState<string | null>(null)
  const open = useMemo(() => commands.find((c) => c.id === openId) ?? null, [commands, openId])

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {commands.map((cmd) => {
        const running = activeCommand === cmd.id
        return (
          <button
            key={cmd.id}
            type="button"
            onClick={() => setOpenId(cmd.id)}
            className={cn(
              "flex flex-col gap-2 rounded-xl border p-4 text-left transition-all",
              running
                ? "border-emerald-400/40 bg-emerald-400/[0.06]"
                : "border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.05]"
            )}
          >
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white/90">{cmd.label}</span>
              {running && (
                <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-300">
                  <span className="size-2 animate-pulse rounded-full bg-emerald-400" />
                  <Elapsed since={activeSince} />
                </span>
              )}
            </div>
            <span className="line-clamp-2 text-xs leading-relaxed text-white/40">
              {cmd.description}
            </span>
          </button>
        )
      })}
      {open && (
        <ActionDialog
          spec={open}
          hideKeys={hiddenKeys}
          pickerJoints={pickerJoints}
          running={activeCommand === open.id}
          blocked={activeCommand != null && activeCommand !== open.id}
          busy={busy}
          disabled={disabled}
          onLaunch={(args) => {
            onLaunch(open.id, args)
            setOpenId(null)
          }}
          onStop={onStop}
          onClose={() => setOpenId(null)}
        />
      )}
    </div>
  )
}

function Elapsed({ since }: { since: number | null }) {
  const [now, setNow] = useState<number | null>(null)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 250)
    return () => clearInterval(t)
  }, [])
  if (since == null || now == null) return null
  const s = Math.max(0, Math.floor(now - since))
  const m = Math.floor(s / 60)
  return <span className="font-mono tabular-nums">{m > 0 ? `${m}m ${s % 60}s` : `${s}s`}</span>
}

/**
 * Floating status panel for the run in flight — one canonical home for the
 * live output line, hands-on prompts (Continue button), and Stop, wherever
 * the run was launched from (action cards or the Motors-section buttons).
 */
export function ActiveRunPanel({
  label,
  since,
  line,
  prompt,
  busy,
  onContinue,
  onStop,
}: {
  label: string
  since: number | null
  line: string | null
  prompt: string | null
  busy: boolean
  onContinue: () => void
  onStop: () => void
}) {
  const parsed = prompt ? parsePrompt(prompt) : null
  return (
    <Card
      className={cn(
        "fixed right-[max(1rem,env(safe-area-inset-right))] bottom-[max(1rem,env(safe-area-inset-bottom))] left-[max(1rem,env(safe-area-inset-left))] z-40 gap-2.5 bg-[#1a1a1a] p-4 shadow-2xl sm:left-auto sm:w-80",
        prompt ? "border-amber-400/50" : "border-emerald-400/30"
      )}
    >
      <div className="flex items-center gap-2">
        <span className="size-2 animate-pulse rounded-full bg-emerald-400" />
        <span className="text-sm font-semibold text-white/90">{label}</span>
        <span className="ml-auto text-xs text-emerald-300">
          <Elapsed since={since} />
        </span>
      </div>
      {parsed ? (
        <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-2.5">
          <span className="flex items-start gap-1.5 text-xs leading-relaxed text-amber-100/90">
            <Hand className="mt-0.5 size-3.5 shrink-0 text-amber-300" />
            {parsed.text}
          </span>
          <Button
            size="sm"
            className="self-start bg-amber-400 text-black hover:bg-amber-300"
            onClick={onContinue}
          >
            {parsed.button} <ArrowRight />
          </Button>
        </div>
      ) : (
        line && <span className="truncate font-mono text-[0.7rem] text-white/45">{line}</span>
      )}
      <Button
        variant="ghost"
        size="sm"
        className="self-start text-white/50"
        onClick={onStop}
        disabled={busy}
      >
        <Square /> Stop
      </Button>
    </Card>
  )
}

/**
 * Checkbox picker for a command's `--joints` subset (ROM tests, guided
 * zeroing). All checked (the default) omits the flag so the command runs its
 * own "all joints" default; any subset is sent as the comma-separated joint
 * list. `joints` restricts the choices (e.g. guided zeroing has no gripper —
 * it self-calibrates and has no zero to set).
 */
function JointPicker({
  value,
  disabled,
  onChange,
  joints = JOINTS,
}: {
  value: FormValue | undefined
  disabled: boolean
  onChange: (value: string | null) => void
  joints?: readonly JointName[]
}) {
  const selected = useMemo(() => {
    const text = typeof value === "string" ? value.trim() : ""
    if (!text) return new Set(joints)
    const names = new Set(text.split(",").map((s) => s.trim().toUpperCase()))
    return new Set(joints.filter((j) => names.has(j)))
  }, [value, joints])

  function toggle(joint: JointName) {
    const next = new Set(selected)
    if (next.has(joint)) next.delete(joint)
    else next.add(joint)
    if (next.size === joints.length)
      onChange(null) // all = command default
    else
      onChange(
        joints
          .filter((j) => next.has(j))
          .map((j) => j.toLowerCase())
          .join(",")
      )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm capitalize">Joints</span>
        <span className="text-xs text-white/35">
          {selected.size === joints.length ? "all" : `${selected.size} of ${joints.length}`}
        </span>
        {selected.size < joints.length && (
          <button
            type="button"
            onClick={() => onChange(null)}
            disabled={disabled}
            className="text-xs text-[#eff483]/80 hover:text-[#eff483]"
          >
            Select all
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {joints.map((joint) => {
          const on = selected.has(joint)
          return (
            <label
              key={joint}
              className={cn(
                "flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5 text-xs capitalize transition-colors",
                on
                  ? "border-white/20 bg-white/[0.05] text-white/85"
                  : "border-white/10 text-white/35 hover:border-white/20",
                disabled && "pointer-events-none opacity-50"
              )}
            >
              <input
                type="checkbox"
                checked={on}
                disabled={disabled}
                onChange={() => toggle(joint)}
                className="sr-only"
              />
              <span
                className={cn("inline-block size-2 rounded-full", !on && "opacity-30")}
                style={{ background: JOINT_COLORS[joint] }}
              />
              {jointLabel(joint)}
            </label>
          )
        })}
      </div>
    </div>
  )
}

/** One way of running a command inside its dialog (a tab). */
export interface ActionMode {
  key: string
  label: string
  description?: string
  presetArgs?: Record<string, FormValue>
  hideKeys?: string[]
  /** This mode works on a whole arm, so the bus picker offers only the arms. */
  armOnly?: boolean
}

/**
 * One choice in a per-motor tool's bus picker: an arm (sent as `--l`/`--r`,
 * its interface injected from the adapter mapping) or any CAN interface on
 * the host (sent as `--channel`), for a motor on a test stand.
 */
export interface BusOption {
  /** "left" / "right" for an arm, or "iface:<name>" for an interface. */
  value: string
  label: string
}

const IFACE_PREFIX = "iface:"

/**
 * Parameter dialog for a catalog command. `presetArgs` are merged into the
 * launch args without appearing in the form, and `hideKeys` removes fields
 * the preset already decides. `modes` renders those as selectable tabs, so
 * one command backs several entry points (e.g. single-motor vs guided
 * zeroing).
 */
export function ActionDialog({
  spec,
  title,
  description,
  presetArgs,
  hideKeys,
  modes,
  busOptions,
  pickerJoints,
  running,
  blocked,
  busy,
  disabled,
  onLaunch,
  onStop,
  onClose,
}: {
  spec: CommandSpec
  title?: string
  description?: string
  presetArgs?: Record<string, FormValue>
  hideKeys?: string[]
  modes?: ActionMode[]
  /**
   * Replace the Arm field with a Bus picker: the arms plus other CAN
   * interfaces. A preset `arm` becomes the picker's default.
   */
  busOptions?: BusOption[]
  /** Restrict the `--joints` picker's choices (default: every joint). */
  pickerJoints?: readonly JointName[]
  running: boolean
  blocked: boolean
  busy: boolean
  disabled: boolean
  onLaunch: (args: Record<string, FormValue>) => void
  onStop: () => void
  onClose: () => void
}) {
  const [overrides, setOverrides] = useState<Record<string, FormValue>>({})
  const [missing, setMissing] = useState<string[]>([])
  const [modeKey, setModeKey] = useState(modes?.[0]?.key ?? null)
  const mode = modes?.find((m) => m.key === modeKey) ?? null

  const allFields = useMemo(() => flattenFields(spec.schema), [spec])
  const busMode = busOptions != null && !mode?.armOnly && allFields.some((f) => f.key === "arm")
  const [bus, setBus] = useState("")
  const presetArm = presetArgs?.arm != null ? String(presetArgs.arm) : ""
  const busValue = bus || presetArm

  const effectivePresets = useMemo(() => {
    const merged: Record<string, FormValue> = {
      ...(presetArgs ?? {}),
      ...(mode?.presetArgs ?? {}),
    }
    // The bus picker decides the arm (or replaces it with an interface).
    if (busMode) delete merged.arm
    return merged
  }, [presetArgs, mode, busMode])
  const hidden = useMemo(
    () =>
      new Set([
        ...HIDDEN_FLAGS,
        ...(hideKeys ?? []),
        ...(mode?.hideKeys ?? []),
        ...(busMode ? ["arm", "channel"] : []),
        ...Object.keys(effectivePresets),
      ]),
    [hideKeys, mode, busMode, effectivePresets]
  )
  const jointsField = useMemo(
    () => allFields.find((f) => f.key === "joints" && !hidden.has("joints")),
    [allFields, hidden]
  )
  // `--firmware` (lift.update, safety.update, mantis.update) is a file the
  // operator picks on this machine: it is uploaded to the host first and the
  // stored path becomes the arg.
  const firmwareField = useMemo(
    () => allFields.find((f) => f.key === "firmware" && !hidden.has("firmware")),
    [allFields, hidden]
  )
  const fields = useMemo(
    () => allFields.filter((f) => !hidden.has(f.key) && f.key !== "joints" && f.key !== "firmware"),
    [allFields, hidden]
  )
  // Server-side pick lists for this command's fields (motor.burn-in's CAN
  // interfaces, say), fetched when the dialog opens. Typing stays free-form.
  const [suggestions, setSuggestions] = useState<Record<string, FieldSuggestion[]>>({})
  useEffect(() => {
    let active = true
    for (const key of spec.suggestedFields ?? []) {
      fetchFieldSuggestions(spec.id, key)
        .then(({ suggestions: rows }) => {
          if (!active) return
          const list = rows.map((r) => ({ value: r.value, label: r.label ?? undefined }))
          setSuggestions((prev) => ({ ...prev, [key]: list }))
        })
        .catch(() => {})
    }
    return () => {
      active = false
    }
  }, [spec.id, spec.suggestedFields])
  const [uploading, setUploading] = useState(false)
  const hasWebPrompts = allFields.some((f) => f.key === "web_prompts")

  function setOverride(key: string, value: FormValue | null) {
    setOverrides((prev) => {
      const next = { ...prev }
      if (value == null) delete next[key]
      else next[key] = value
      return next
    })
  }

  function handleRun() {
    const formFields = [
      ...fields,
      ...(jointsField ? [jointsField] : []),
      ...(firmwareField ? [firmwareField] : []),
    ]
    const miss = missingRequired(formFields, overrides)
    if (busMode && !busValue) miss.unshift("Bus")
    setMissing(miss)
    if (miss.length > 0) return
    const args = computeArgs(formFields, overrides)
    // A restricted picker (e.g. no gripper) must send its joint list even
    // when everything is selected: omitting --joints would fall through to
    // the command's own default, which includes the excluded joints.
    if (jointsField && pickerJoints && pickerJoints.length < JOINTS.length && args.joints == null) {
      args.joints = pickerJoints.map((j) => j.toLowerCase()).join(",")
    }
    if (hasWebPrompts) args.web_prompts = true
    if (busMode) {
      if (busValue.startsWith(IFACE_PREFIX)) args.channel = busValue.slice(IFACE_PREFIX.length)
      else args.arm = busValue
    }
    Object.assign(args, effectivePresets)
    onLaunch(args)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <Card
        className="max-h-[85vh] w-full max-w-md gap-4 overflow-auto bg-[#1a1a1a] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-2">
          <div className="flex flex-col gap-1">
            <h3 className="font-heading text-base font-semibold">{title ?? spec.label}</h3>
            <p className="text-sm leading-relaxed text-white/45">
              {mode?.description ?? description ?? spec.description}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto size-7 shrink-0"
            onClick={onClose}
            aria-label="Close"
          >
            <X />
          </Button>
        </div>

        {modes && modes.length > 1 && (
          <div className="flex self-start overflow-hidden rounded-md border border-white/10">
            {modes.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => setModeKey(m.key)}
                className={cn(
                  "px-3 py-1.5 text-xs transition-colors",
                  modeKey === m.key
                    ? "bg-[#eff483]/15 text-[#eff483]"
                    : "text-white/50 hover:bg-white/[0.05]"
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        )}

        {!spec.available && <p className="text-xs text-red-300">Unavailable: {spec.error}</p>}
        {disabled && !running && (
          <p className="text-xs text-amber-200/70">Connect to the robot host first.</p>
        )}
        {blocked && (
          <p className="text-xs text-amber-200/70">
            Another diagnostic is running. Stop it or wait for it to finish.
          </p>
        )}
        {missing.length > 0 && (
          <p className="text-xs text-red-300">Required: {missing.join(", ")}</p>
        )}

        {busMode && (
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Label htmlFor="cfg-bus">Bus</Label>
              <span className="text-xs text-[#eff483]">*</span>
            </div>
            <select
              id="cfg-bus"
              value={busValue}
              disabled={running || busy}
              onChange={(e) => setBus(e.target.value)}
              className="h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50"
            >
              <option value="">Select…</option>
              {(busOptions ?? []).map((o) => (
                <option key={o.value} value={o.value} className="bg-[#1a1a1a]">
                  {o.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-white/40">
              Pick an arm, or another CAN interface for a motor on a test stand.
            </p>
          </div>
        )}

        {jointsField && (
          <JointPicker
            value={overrides.joints}
            disabled={running || busy}
            onChange={(v) => setOverride("joints", v)}
            joints={pickerJoints}
          />
        )}

        {firmwareField && (
          <FirmwareUploadField
            board={FIRMWARE_BOARDS[spec.id] ?? "lift"}
            help={firmwareField.help}
            disabled={running || busy}
            onUploading={setUploading}
            onChange={(path) => setOverride("firmware", path)}
          />
        )}

        {fields.length > 0 && (
          <CuratedForm
            fields={fields}
            suggestions={suggestions}
            overrides={overrides}
            disabled={running || busy}
            onChange={(key, value) => setOverride(key, value)}
            onReset={(key) => setOverride(key, null)}
          />
        )}

        {hasWebPrompts && (
          <p className="flex items-start gap-2 rounded-md border border-white/10 bg-white/[0.02] p-2.5 text-xs leading-relaxed text-white/45">
            <Hand className="mt-0.5 size-3.5 shrink-0 text-white/35" />
            Hands-on steps pause the run until you press Continue.
          </p>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          {running ? (
            <Button variant="destructive" size="sm" onClick={onStop} disabled={busy}>
              {busy ? <Loader2 className="animate-spin" /> : <Square />} Stop
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={handleRun}
              disabled={busy || uploading || disabled || blocked || !spec.available}
            >
              {busy ? <Loader2 className="animate-spin" /> : <Play />} Run
            </Button>
          )}
        </div>
      </Card>
    </div>
  )
}

/** The board each firmware-update command's `--firmware` file is for. */
const FIRMWARE_BOARDS: Record<string, FirmwareBoard> = {
  "lift.update": "lift",
  "safety.update": "safety",
  "mantis.update": "mantis",
}

/**
 * File picker for a command's `--firmware` argument: uploads the chosen file
 * to the serve host, which validates it (that board's firmware.bin, so the
 * lift's file is refused for the safety board or Mantis, and the other way
 * round) and stores it; the stored path is what the run receives. Clearing or
 * a failed upload leaves the argument unset (the command then only reports).
 */
export function FirmwareUploadField({
  board = "lift",
  help,
  disabled,
  onUploading,
  onChange,
}: {
  board?: FirmwareBoard
  help?: string | null
  disabled: boolean
  onUploading: (uploading: boolean) => void
  onChange: (path: string | null) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [uploaded, setUploaded] = useState<FirmwareUpload | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  async function handleFile(file: File | undefined) {
    if (!file) return
    setFileName(file.name)
    setUploaded(null)
    setError(null)
    onChange(null)
    setUploading(true)
    onUploading(true)
    try {
      const result = await uploadFirmware(board, file)
      setUploaded(result)
      onChange(result.path)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setUploading(false)
      onUploading(false)
      if (input.current) input.current.value = ""
    }
  }

  function clear() {
    setUploaded(null)
    setFileName(null)
    setError(null)
    onChange(null)
  }

  return (
    <div className="flex flex-col gap-2 rounded-md border border-white/10 bg-white/[0.02] p-3">
      <span className="text-xs font-medium text-white/70">Firmware file</span>
      {help && <span className="text-xs leading-relaxed text-white/40">{help}</span>}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={disabled || uploading}
          onClick={() => input.current?.click()}
        >
          {uploading ? <Loader2 className="animate-spin" /> : <Upload />}
          {fileName ? "Choose another" : "Choose firmware.bin"}
        </Button>
        {(uploaded || error) && !uploading && (
          <Button variant="ghost" size="sm" disabled={disabled} onClick={clear}>
            Clear
          </Button>
        )}
        <input
          ref={input}
          type="file"
          accept=".bin,application/octet-stream"
          className="hidden"
          aria-label="Firmware file"
          onChange={(e) => void handleFile(e.target.files?.[0])}
        />
      </div>
      {uploading && <p className="text-xs text-white/45">Uploading {fileName}…</p>}
      {uploaded && (
        <p className="text-xs text-emerald-300/80">
          {fileName}: firmware {uploaded.version}, built {uploaded.built} (build {uploaded.buildId},{" "}
          {Math.round(uploaded.size / 1024)} KB)
        </p>
      )}
      {error && <p className="text-xs text-red-300">{error}</p>}
    </div>
  )
}
