import { useEffect, useState } from "react"
import { Check, Loader2, RefreshCw, Ruler, Save } from "lucide-react"
import {
  fetchTrackerCalibration,
  saveTrackerCalibration,
  type MantisTrackerSource,
  type TrackerCalibrationSide,
  type TrackerCalibrationSnapshot,
  type TrackerCalibrationValue,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useToast } from "@/components/ui/toast"

type Side = "left" | "right"

interface TransformDraft {
  pos: [string, string, string]
  quat: [string, string, string, string]
}

const EMPTY_DRAFT: TransformDraft = {
  pos: ["", "", ""],
  quat: ["", "", "", ""],
}

function message(error: unknown): string {
  return String(error).replace(/^Error:\s*/, "")
}

function draftFor(entry: TrackerCalibrationSide): TransformDraft {
  if (!entry.pos || !entry.quat || entry.status !== "measured") {
    return { pos: [...EMPTY_DRAFT.pos], quat: [...EMPTY_DRAFT.quat] }
  }
  return {
    pos: entry.pos.map(String) as TransformDraft["pos"],
    quat: entry.quat.map(String) as TransformDraft["quat"],
  }
}

function parseDraft(draft: TransformDraft): Omit<TrackerCalibrationValue, "key"> | string {
  const raw = [...draft.pos, ...draft.quat]
  if (raw.some((value) => value.trim() === "")) return "Enter all seven measured values."
  const values = raw.map(Number)
  if (values.some((value) => !Number.isFinite(value))) return "Every value must be a finite number."
  const quatNorm = Math.hypot(...values.slice(3))
  if (quatNorm <= 1e-12) return "The quaternion must have a non-zero norm."
  return {
    pos: values.slice(0, 3) as TrackerCalibrationValue["pos"],
    quat: values.slice(3) as TrackerCalibrationValue["quat"],
  }
}

function sourceLabel(source: MantisTrackerSource): string {
  if (source === "quest") return "Quest / WebXR"
  if (source === "ultimate") return "VIVE Ultimate"
  return "Lighthouse / Tracker 3.0"
}

/** Writes only explicitly confirmed, active-datum tracker→TCP measurements. */
export function TrackerCalibrationPanel({
  source,
  contextSaved,
}: {
  source: MantisTrackerSource
  contextSaved: boolean
}) {
  const toast = useToast()
  const [snapshot, setSnapshot] = useState<TrackerCalibrationSnapshot | null>(null)
  const [drafts, setDrafts] = useState<Record<Side, TransformDraft>>({
    left: { pos: [...EMPTY_DRAFT.pos], quat: [...EMPTY_DRAFT.quat] },
    right: { pos: [...EMPTY_DRAFT.pos], quat: [...EMPTY_DRAFT.quat] },
  })
  const [confirmed, setConfirmed] = useState<Record<Side, boolean>>({
    left: false,
    right: false,
  })
  const [loading, setLoading] = useState(true)
  const [savingSide, setSavingSide] = useState<Side | null>(null)
  const [error, setError] = useState<string | null>(null)

  function load() {
    setLoading(true)
    setError(null)
    fetchTrackerCalibration(source)
      .then((found) => {
        setSnapshot(found)
        setDrafts({ left: draftFor(found.left), right: draftFor(found.right) })
        setConfirmed({ left: false, right: false })
      })
      .catch((reason) => setError(message(reason)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    let cancelled = false
    if (!contextSaved) return
    fetchTrackerCalibration(source)
      .then((found) => {
        if (cancelled) return
        setSnapshot(found)
        setDrafts({ left: draftFor(found.left), right: draftFor(found.right) })
        setConfirmed({ left: false, right: false })
      })
      .catch((reason) => {
        if (!cancelled) setError(message(reason))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [source, contextSaved])

  function setPart(side: Side, part: "pos" | "quat", index: number, value: string) {
    setDrafts((current) => {
      const nextPart = [...current[side][part]]
      nextPart[index] = value
      return {
        ...current,
        [side]: { ...current[side], [part]: nextPart },
      } as Record<Side, TransformDraft>
    })
    setConfirmed((current) => ({ ...current, [side]: false }))
  }

  async function save(side: Side) {
    const trackerKey = snapshot?.[side].key
    if (!trackerKey || !confirmed[side] || !contextSaved) return
    const parsed = parseDraft(drafts[side])
    if (typeof parsed === "string") {
      setError(`${side === "left" ? "Left" : "Right"}: ${parsed}`)
      return
    }
    setSavingSide(side)
    setError(null)
    try {
      const saved = await saveTrackerCalibration(source, {
        [side]: { key: trackerKey, ...parsed },
      })
      setSnapshot(saved)
      setDrafts((current) => ({ ...current, [side]: draftFor(saved[side]) }))
      setConfirmed((current) => ({ ...current, [side]: false }))
      toast.success(`${side === "left" ? "Left" : "Right"} tracker mount saved as measured.`)
    } catch (reason) {
      setError(message(reason))
      toast.error(`Could not save tracker calibration: ${message(reason)}`)
    } finally {
      setSavingSide(null)
    }
  }

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-white/10 bg-white/[0.02] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Ruler className="size-4 text-white/50" />
        <span className="text-sm font-medium text-white/80">
          {sourceLabel(source)} tracker → gripper calibration
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="ml-auto"
          onClick={load}
          disabled={loading || savingSide !== null}
        >
          <RefreshCw className={loading ? "animate-spin" : ""} /> Refresh
        </Button>
      </div>
      <div className="max-w-prose space-y-1 text-xs leading-relaxed text-white/45">
        <p>
          Enter the gripper TCP expressed in each tracker&apos;s local frame: position in metres and
          quaternion in <span className="font-mono text-white/60">qx, qy, qz, qw</span> order.
        </p>
        <p className="text-amber-200/75">
          Uncalibrated sides stay blank. The UI never inserts an identity placeholder or promotes an
          unverified CAD candidate; Save requires you to confirm a physical bench measurement.
        </p>
      </div>

      {!contextSaved && (
        <p className="rounded-md border border-amber-400/25 bg-amber-400/[0.05] p-2 text-xs text-amber-200/80">
          Save Settings first so this measurement is attached to the selected source and, for Quest,
          its exact reported controller datum.
        </p>
      )}

      {!contextSaved ? null : loading ? (
        <p className="flex items-center gap-2 text-xs text-white/45">
          <Loader2 className="size-4 animate-spin" /> Loading active tracker keys…
        </p>
      ) : snapshot ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {(["left", "right"] as const).map((side) => (
            <CalibrationSideEditor
              key={`${source}:${side}:${snapshot[side].key ?? "unbound"}`}
              side={side}
              entry={snapshot[side]}
              draft={drafts[side]}
              confirmed={confirmed[side]}
              saving={savingSide === side}
              disabled={!contextSaved || savingSide !== null}
              onPartChange={(part, index, value) => setPart(side, part, index, value)}
              onConfirmedChange={(value) =>
                setConfirmed((current) => ({ ...current, [side]: value }))
              }
              onSave={() => save(side)}
            />
          ))}
        </div>
      ) : null}

      {error && <p className="text-xs leading-relaxed text-red-300/80">{error}</p>}
    </section>
  )
}

function CalibrationSideEditor({
  side,
  entry,
  draft,
  confirmed,
  saving,
  disabled,
  onPartChange,
  onConfirmedChange,
  onSave,
}: {
  side: Side
  entry: TrackerCalibrationSide
  draft: TransformDraft
  confirmed: boolean
  saving: boolean
  disabled: boolean
  onPartChange: (part: "pos" | "quat", index: number, value: string) => void
  onConfirmedChange: (value: boolean) => void
  onSave: () => void
}) {
  const label = side === "left" ? "Left Mantis" : "Right Mantis"
  const parsed = parseDraft(draft)
  const complete = typeof parsed !== "string"
  const statusCopy = {
    measured: "Measured override saved",
    factory: "Verified factory constant active",
    candidate: "CAD candidate only — not calibrated",
    missing: "Measurement required",
    unbound: "Active tracker key unavailable",
  }[entry.status]
  const statusClass =
    entry.status === "measured" || entry.status === "factory"
      ? "bg-emerald-400/10 text-emerald-300"
      : entry.status === "candidate"
        ? "bg-amber-400/10 text-amber-300"
        : "bg-red-400/10 text-red-300"

  return (
    <fieldset className="flex min-w-0 flex-col gap-3 rounded-md border border-white/10 bg-black/15 p-3">
      <legend className="sr-only">{label} tracker calibration</legend>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-white/70">{label}</span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] ${statusClass}`}>{statusCopy}</span>
      </div>
      {entry.key ? (
        <code className="break-all text-[10px] text-white/35">{entry.key}</code>
      ) : (
        <p className="text-[11px] leading-relaxed text-red-300/75">
          {entry.status === "unbound"
            ? "Identify this tracker first. For Quest, connect WebXR, use its reported grip key, and save Settings."
            : "No active tracker datum is available for this side."}
        </p>
      )}

      <TransformInputs
        side={side}
        part="pos"
        labels={["x", "y", "z"]}
        values={draft.pos}
        disabled={!entry.key || disabled}
        onChange={(index, value) => onPartChange("pos", index, value)}
      />
      <TransformInputs
        side={side}
        part="quat"
        labels={["qx", "qy", "qz", "qw"]}
        values={draft.quat}
        disabled={!entry.key || disabled}
        onChange={(index, value) => onPartChange("quat", index, value)}
      />

      <label className="flex items-start gap-2 text-[11px] leading-relaxed text-white/50">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(event) => onConfirmedChange(event.target.checked)}
          disabled={!entry.key || disabled || !complete}
          className="mt-0.5 size-3.5 accent-[#eff483]"
        />
        <span>I bench-measured these values for this physical tracker mount.</span>
      </label>
      {!complete && entry.key && (
        <p className="text-[10px] text-white/35">
          {typeof parsed === "string" ? parsed : "Enter all seven values."}
        </p>
      )}
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="self-start"
        onClick={onSave}
        disabled={!entry.key || disabled || !complete || !confirmed || saving}
      >
        {saving ? (
          <Loader2 className="animate-spin" />
        ) : entry.status === "measured" ? (
          <Save />
        ) : (
          <Check />
        )}
        Save {side} as measured
      </Button>
    </fieldset>
  )
}

function TransformInputs({
  side,
  part,
  labels,
  values,
  disabled,
  onChange,
}: {
  side: Side
  part: "pos" | "quat"
  labels: string[]
  values: readonly string[]
  disabled: boolean
  onChange: (index: number, value: string) => void
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] text-white/45">
        {part === "pos" ? "Position (metres)" : "Quaternion (xyzw)"}
      </span>
      <div
        className={`grid gap-2 ${part === "pos" ? "grid-cols-3" : "grid-cols-2 sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4"}`}
      >
        {labels.map((component, index) => (
          <label key={component} className="flex min-w-0 flex-col gap-1">
            <span className="font-mono text-[9px] text-white/30">{component}</span>
            <Input
              type="number"
              inputMode="decimal"
              step="any"
              value={values[index] ?? ""}
              onChange={(event) => onChange(index, event.target.value)}
              disabled={disabled}
              placeholder={component}
              aria-label={`${side} ${part === "pos" ? "position" : "quaternion"} ${component}`}
              className="h-8 px-2 font-mono text-xs"
            />
          </label>
        ))}
      </div>
    </div>
  )
}
