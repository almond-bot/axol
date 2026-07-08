import { useEffect, useMemo, useState } from "react"
import { Loader2, Play, Square, X } from "lucide-react"
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

/**
 * Diagnostics as app actions: a card per test — click it, set parameters in a
 * dialog, hit Run. Progress feedback is the card's running state + toasts;
 * results (with telemetry and the full log) land in the run history below.
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

function ActionDialog({
  spec,
  running,
  blocked,
  busy,
  disabled,
  onLaunch,
  onStop,
  onClose,
}: {
  spec: CommandSpec
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

  const allFields = useMemo(() => flattenFields(spec.schema), [spec])
  // Prompts can't be answered in the browser (sessions have no stdin), so the
  // no-prompt countdown mode is forced on and hidden from the form.
  const fields = useMemo(() => allFields.filter((f) => f.key !== "no_prompt"), [allFields])
  const hasNoPrompt = allFields.length !== fields.length

  function handleRun() {
    const miss = missingRequired(fields, overrides)
    setMissing(miss)
    if (miss.length > 0) return
    const args = computeArgs(fields, overrides)
    if (hasNoPrompt) args.no_prompt = true
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
            <h3 className="font-heading text-base font-semibold">{spec.label}</h3>
            <p className="text-sm leading-relaxed text-white/45">{spec.description}</p>
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

        {fields.length > 0 && (
          <CuratedForm
            fields={fields}
            overrides={overrides}
            disabled={running || busy}
            onChange={(key, value) => setOverrides((prev) => ({ ...prev, [key]: value }))}
            onReset={(key) =>
              setOverrides((prev) => {
                const next = { ...prev }
                delete next[key]
                return next
              })
            }
          />
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
