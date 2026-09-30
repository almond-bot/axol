import { useCallback, useEffect, useState, type ReactNode } from "react"
import { Check, Loader2, Lock, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectOption } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import {
  formatParamValue,
  groupParams,
  paramUi,
  parseParamInput,
  unitLabel,
} from "@/lib/motor-params"
import {
  fetchMotorConfig,
  writeMotorConfig,
  type MotorConfig,
  type MotorConfigParam,
} from "@/lib/telemetry"

function errorText(e: unknown): string {
  return String(e).replace(/^Error:\s*/, "")
}

/**
 * The MyActuator parameter editor: every configuration parameter of one motor,
 * laid out like the setup software's panels. Each Set writes one parameter
 * and persists it to the motor's flash, then re-reads the table (a write can
 * change a related parameter, e.g. the 2nd encoder's resolution).
 */
export function MotorParamsPanel({ arm, joint }: { arm: string; joint: string }) {
  const [config, setConfig] = useState<MotorConfig | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)
  const [rowStatus, setRowStatus] = useState<Record<string, { ok: boolean; text: string }>>({})
  const [unlocked, setUnlocked] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const next = await fetchMotorConfig(arm, joint)
      setConfig(next)
      setDrafts(Object.fromEntries(next.params.map((p) => [p.name, formatParamValue(p.value)])))
    } catch (e) {
      setLoadError(errorText(e))
    } finally {
      setLoading(false)
    }
  }, [arm, joint])

  useEffect(() => {
    void load()
  }, [load])

  const save = async (param: MotorConfigParam) => {
    const ui = paramUi(param.name)
    const value = parseParamInput(drafts[param.name] ?? "")
    if (value == null) return
    if (
      param.access === "protected" &&
      !window.confirm(`Write ${ui.label} = ${value} to this motor's flash?`)
    ) {
      return
    }
    setSaving(param.name)
    setRowStatus((s) => {
      const rest = { ...s }
      delete rest[param.name]
      return rest
    })
    try {
      const result = await writeMotorConfig(
        arm,
        joint,
        param.name,
        value,
        param.access === "protected"
      )
      const clamped = result.value != null && result.value !== value
      setRowStatus((s) => ({
        ...s,
        [param.name]: clamped
          ? { ok: false, text: `Motor stored ${formatParamValue(result.value)}` }
          : { ok: true, text: "Saved" },
      }))
      await load()
    } catch (e) {
      setRowStatus((s) => ({ ...s, [param.name]: { ok: false, text: errorText(e) } }))
    } finally {
      setSaving(null)
    }
  }

  if (loading && !config) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-white/40">
        <Loader2 className="size-4 animate-spin" /> Reading parameters…
      </div>
    )
  }
  if (loadError && !config) {
    return (
      <div className="flex flex-col items-start gap-3 py-2">
        <p className="text-sm text-red-300">{loadError}</p>
        <Button variant="outline" size="sm" onClick={() => void load()}>
          <RefreshCw /> Retry
        </Button>
      </div>
    )
  }
  if (!config) return null

  const hasProtected = config.params.some((p) => p.access === "protected")

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="text-xs text-white/45">
          Firmware <span className="font-mono text-white/80">{config.firmware ?? "–"}</span>
        </span>
        {hasProtected && (
          <label className="flex items-center gap-2 text-xs text-white/60">
            <Switch
              checked={unlocked}
              onChange={setUnlocked}
              aria-label="Allow editing protected parameters"
            />
            Edit protected parameters
          </label>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto"
          disabled={loading || saving != null}
          onClick={() => void load()}
        >
          {loading ? <Loader2 className="animate-spin" /> : <RefreshCw />} Reload
        </Button>
      </div>
      {unlocked && (
        <p className="rounded-md border border-amber-300/20 bg-amber-300/[0.06] px-3 py-2 text-xs text-amber-200/90">
          Protected parameters (marked <Lock className="inline size-3" />) are factory, calibration
          and bus-identity settings. A wrong value can leave the motor unable to commutate or
          unreachable on the bus.
        </p>
      )}
      {loadError && <p className="text-xs text-red-300">Reload failed: {loadError}</p>}

      <div className="grid gap-4 lg:grid-cols-2">
        {groupParams(config.params).map((section) => (
          <section
            key={section.title}
            className="flex flex-col rounded-xl border border-white/10 bg-white/[0.02] p-3"
          >
            <h4 className="mb-1 text-xs font-semibold tracking-wide text-white/70 uppercase">
              {section.title}
            </h4>
            <div className="flex flex-col divide-y divide-white/[0.06]">
              {section.params.map((param) => (
                <ParamRow
                  key={param.name}
                  param={param}
                  draft={drafts[param.name] ?? ""}
                  onDraft={(text) => setDrafts((d) => ({ ...d, [param.name]: text }))}
                  onSave={() => void save(param)}
                  locked={param.access === "protected" && !unlocked}
                  busy={saving != null}
                  saving={saving === param.name}
                  status={rowStatus[param.name]}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
      <p className="text-[0.7rem] text-white/35">
        Each Set writes one parameter to the motor and persists it to flash.
      </p>
    </div>
  )
}

function ParamRow({
  param,
  draft,
  onDraft,
  onSave,
  locked,
  busy,
  saving,
  status,
}: {
  param: MotorConfigParam
  draft: string
  onDraft: (text: string) => void
  onSave: () => void
  locked: boolean
  busy: boolean
  saving: boolean
  status?: { ok: boolean; text: string }
}) {
  const ui = paramUi(param.name)
  const unit = unitLabel(param.unit)
  const current = formatParamValue(param.value)
  const parsed = parseParamInput(draft)
  const readOnly = param.access === "read_only" || ui.kind === "display"
  const editable = param.supported && param.value != null && !readOnly
  const dirty = draft !== current
  const disabled = !editable || locked || busy

  let control: ReactNode
  if (!param.supported) {
    control = <span className="text-xs text-white/35">Not on this firmware</span>
  } else if (param.value == null) {
    control = <span className="text-xs text-red-300">{param.error ?? "Unreadable"}</span>
  } else if (readOnly) {
    control = (
      <span className="font-mono text-sm text-white/85 tabular-nums">
        {current}
        {unit && <span className="ml-1 text-white/40">{unit}</span>}
      </span>
    )
  } else if (ui.kind === "toggle") {
    control = (
      <Switch
        checked={parsed != null && parsed !== 0}
        disabled={disabled}
        onChange={(on) => onDraft(on ? "1" : "0")}
        aria-label={ui.label}
      />
    )
  } else if (ui.kind === "choice" && ui.choices) {
    const known = ui.choices.some((c) => String(c.value) === draft)
    control = (
      <Select
        value={draft}
        disabled={disabled}
        onChange={(e) => onDraft(e.target.value)}
        aria-label={ui.label}
        className="h-8 w-48"
      >
        {!known && <SelectOption value={draft} label={`Unknown (${draft})`} />}
        {ui.choices.map((c) => (
          <SelectOption key={c.value} value={String(c.value)} label={c.label} />
        ))}
      </Select>
    )
  } else {
    control = (
      <div className="flex items-center gap-1.5">
        <Input
          value={draft}
          inputMode="decimal"
          disabled={disabled}
          onChange={(e) => onDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && dirty && parsed != null) onSave()
          }}
          aria-label={ui.label}
          aria-invalid={parsed == null}
          className={cn("h-8 w-32 font-mono tabular-nums", parsed == null && "border-red-400/60")}
        />
        {unit && <span className="w-10 text-xs text-white/40">{unit}</span>}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 py-2">
      <div className="flex items-center gap-3">
        <span className="flex min-w-0 flex-1 items-center gap-1.5 text-xs text-white/70">
          {param.access === "protected" && (
            <Lock className="size-3 shrink-0 text-white/35" aria-label="Protected" />
          )}
          <span
            className="truncate"
            title={`${param.name} (0x${param.index.toString(16).padStart(2, "0")})`}
          >
            {ui.label}
          </span>
        </span>
        {control}
        {editable && (
          <Button
            size="sm"
            variant={dirty ? "default" : "outline"}
            className="h-8 w-14"
            disabled={disabled || !dirty || parsed == null}
            onClick={onSave}
          >
            {saving ? <Loader2 className="animate-spin" /> : "Set"}
          </Button>
        )}
      </div>
      {(status || ui.hint) && (
        <div className="flex items-center gap-2 text-[0.7rem]">
          {status && (
            <span
              className={cn(
                "flex items-center gap-1",
                status.ok ? "text-emerald-300" : "text-amber-300"
              )}
            >
              {status.ok && <Check className="size-3" />}
              {status.text}
            </span>
          )}
          {ui.hint && <span className="text-white/35">{ui.hint}</span>}
        </div>
      )}
    </div>
  )
}
