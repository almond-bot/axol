import { useEffect, useMemo, useState } from "react"
import { ArrowRight, Hand, Loader2, Play, Square, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { CuratedForm } from "@/components/config-form"
import { cn } from "@/lib/utils"
import {
  computeArgs,
  flattenFields,
  missingRequired,
  type CommandSpec,
  type FormValue,
} from "@/lib/supervisor"
import { JOINTS, JOINT_COLORS, jointLabel, type JointName } from "@/lib/telemetry"

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
  return (
    <Card
      className={cn(
        "fixed right-4 bottom-4 z-40 w-80 gap-2.5 bg-[#1a1a1a] p-4 shadow-2xl",
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
      {prompt ? (
        <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-2.5">
          <span className="flex items-start gap-1.5 text-xs leading-relaxed text-amber-100/90">
            <Hand className="mt-0.5 size-3.5 shrink-0 text-amber-300" />
            {prompt}
          </span>
          <Button
            size="sm"
            className="self-start bg-amber-400 text-black hover:bg-amber-300"
            onClick={onContinue}
          >
            Continue <ArrowRight />
          </Button>
        </div>
      ) : (
        line && (
          <span className="truncate font-mono text-[0.7rem] text-white/45">{line}</span>
        )
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
 * Checkbox picker for a command's `--joints` subset (ROM tests). All checked
 * (the default) omits the flag so the command runs its own "all joints"
 * default; any subset is sent as the comma-separated joint list.
 */
function JointPicker({
  value,
  disabled,
  onChange,
}: {
  value: FormValue | undefined
  disabled: boolean
  onChange: (value: string | null) => void
}) {
  const selected = useMemo(() => {
    const text = typeof value === "string" ? value.trim() : ""
    if (!text) return new Set(JOINTS)
    const names = new Set(text.split(",").map((s) => s.trim().toUpperCase()))
    return new Set(JOINTS.filter((j) => names.has(j)))
  }, [value])

  function toggle(joint: JointName) {
    const next = new Set(selected)
    if (next.has(joint)) next.delete(joint)
    else next.add(joint)
    if (next.size === JOINTS.length) onChange(null) // all = command default
    else onChange(JOINTS.filter((j) => next.has(j)).map((j) => j.toLowerCase()).join(","))
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm capitalize">Joints</span>
        <span className="text-xs text-white/35">
          {selected.size === JOINTS.length ? "all" : `${selected.size} of ${JOINTS.length}`}
        </span>
        {selected.size < JOINTS.length && (
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
        {JOINTS.map((joint) => {
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
}

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

  const effectivePresets = useMemo(
    () => ({ ...(presetArgs ?? {}), ...(mode?.presetArgs ?? {}) }),
    [presetArgs, mode]
  )
  const hidden = useMemo(
    () =>
      new Set([
        ...HIDDEN_FLAGS,
        ...(hideKeys ?? []),
        ...(mode?.hideKeys ?? []),
        ...Object.keys(effectivePresets),
      ]),
    [hideKeys, mode, effectivePresets]
  )
  const allFields = useMemo(() => flattenFields(spec.schema), [spec])
  const jointsField = useMemo(
    () => allFields.find((f) => f.key === "joints" && !hidden.has("joints")),
    [allFields, hidden]
  )
  const fields = useMemo(
    () => allFields.filter((f) => !hidden.has(f.key) && f.key !== "joints"),
    [allFields, hidden]
  )
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
    const formFields = jointsField ? [...fields, jointsField] : fields
    const miss = missingRequired(formFields, overrides)
    setMissing(miss)
    if (miss.length > 0) return
    const args = computeArgs(formFields, overrides)
    if (hasWebPrompts) args.web_prompts = true
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
            Another diagnostic is running — stop it or wait for it to finish.
          </p>
        )}
        {missing.length > 0 && (
          <p className="text-xs text-red-300">Missing required: {missing.join(", ")}</p>
        )}

        {jointsField && (
          <JointPicker
            value={overrides.joints}
            disabled={running || busy}
            onChange={(v) => setOverride("joints", v)}
          />
        )}

        {fields.length > 0 && (
          <CuratedForm
            fields={fields}
            overrides={overrides}
            disabled={running || busy}
            onChange={(key, value) => setOverride(key, value)}
            onReset={(key) => setOverride(key, null)}
          />
        )}

        {hasWebPrompts && (
          <p className="flex items-start gap-2 rounded-md border border-white/10 bg-white/[0.02] p-2.5 text-xs leading-relaxed text-white/45">
            <Hand className="mt-0.5 size-3.5 shrink-0 text-white/35" />
            Hands-on steps pause the run and show a Continue button — the run waits for
            you, then proceeds when you click it.
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
              disabled={busy || disabled || blocked || !spec.available}
            >
              {busy ? <Loader2 className="animate-spin" /> : <Play />} Run
            </Button>
          )}
        </div>
      </Card>
    </div>
  )
}
