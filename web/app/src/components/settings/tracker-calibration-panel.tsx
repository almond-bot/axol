import { useEffect, useState } from "react"
import { Check, Loader2, RefreshCw, Ruler, Save, Trash2 } from "lucide-react"
import {
  fetchTrackerCalibration,
  removeTrackerCalibration,
  saveTrackerCalibration,
  type MantisTrackerSource,
  type TrackerCalibrationSide,
  type TrackerCalibrationSnapshot,
  type TrackerCalibrationValue,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useToast } from "@/components/ui/toast"
import { hasApprovedTrackerFactoryTransform } from "@/lib/tracker-calibration"

type Side = "left" | "right"

interface RemovalTarget {
  side: Side
  key: string
}

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
  if (!entry.pos || !entry.quat || !["measured", "stale"].includes(entry.status)) {
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
  if (Math.hypot(...values.slice(0, 3)) > 1) {
    return "Position must be within 1 metre of the tracker. Enter metres, not millimetres."
  }
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
  const [removingSide, setRemovingSide] = useState<Side | null>(null)
  const [removalTarget, setRemovalTarget] = useState<RemovalTarget | null>(null)
  const [error, setError] = useState<string | null>(null)

  function load() {
    setLoading(true)
    setError(null)
    fetchTrackerCalibration(source)
      .then((found) => {
        setSnapshot(found)
        setDrafts({ left: draftFor(found.left), right: draftFor(found.right) })
        setConfirmed({ left: false, right: false })
        setRemovalTarget(null)
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
        setRemovalTarget(null)
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
    setRemovalTarget(null)
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
      setRemovalTarget(null)
      toast.success(`${side === "left" ? "Left" : "Right"} tracker mount saved as measured.`)
    } catch (reason) {
      setError(message(reason))
      toast.error(`Could not save tracker calibration: ${message(reason)}`)
    } finally {
      setSavingSide(null)
    }
  }

  async function remove(side: Side, overrideKey: string) {
    const activeKey = snapshot?.[side].key
    const revision = snapshot?.[side].overrideRevisions?.[overrideKey]
    if (
      !activeKey ||
      !revision ||
      removalTarget?.side !== side ||
      removalTarget.key !== overrideKey ||
      !contextSaved
    )
      return
    setRemovingSide(side)
    setError(null)
    try {
      const saved = await removeTrackerCalibration(source, side, overrideKey, activeKey, revision)
      setSnapshot(saved)
      setDrafts((current) => ({ ...current, [side]: draftFor(saved[side]) }))
      setConfirmed((current) => ({ ...current, [side]: false }))
      setRemovalTarget(null)
      if (saved[side].status === "factory") {
        toast.success(
          `${side === "left" ? "Left" : "Right"} override removed; factory transform restored.`
        )
      } else if ((saved[side].overrideKeys?.length ?? 0) > 0) {
        toast.success(
          `${side === "left" ? "Left" : "Right"} override removed; review the remaining saved override.`
        )
      } else {
        toast.success(
          `${side === "left" ? "Left" : "Right"} override removed; this side is now uncalibrated.`
        )
      }
    } catch (reason) {
      setError(message(reason))
      toast.error(`Could not remove tracker calibration: ${message(reason)}`)
    } finally {
      setRemovingSide(null)
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
          disabled={loading || savingSide !== null || removingSide !== null}
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
          Factory-backed VIVE sides may stay blank; saved values are optional overrides. Other blank
          sides are uncalibrated. The UI never inserts an identity placeholder, and Save requires
          you to confirm a physical bench measurement.
        </p>
        {snapshot?.activePoseConvention && (
          <p>
            Active Ultimate pose convention: {snapshot.activePoseConvention.quatOrder} quaternion ·{" "}
            {snapshot.activePoseConvention.upAxis}-up.
          </p>
        )}
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
              factoryBacked={hasApprovedTrackerFactoryTransform(
                source,
                snapshot.activePoseConvention
              )}
              entry={snapshot[side]}
              draft={drafts[side]}
              confirmed={confirmed[side]}
              saving={savingSide === side}
              removing={removingSide === side}
              removeKey={removalTarget?.side === side ? removalTarget.key : null}
              disabled={!contextSaved || savingSide !== null || removingSide !== null}
              onPartChange={(part, index, value) => setPart(side, part, index, value)}
              onConfirmedChange={(value) =>
                setConfirmed((current) => ({ ...current, [side]: value }))
              }
              onSave={() => save(side)}
              onRequestRemove={(key) => setRemovalTarget({ side, key })}
              onCancelRemove={() => setRemovalTarget(null)}
              onRemove={(key) => remove(side, key)}
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
  factoryBacked,
  entry,
  draft,
  confirmed,
  saving,
  removing,
  removeKey,
  disabled,
  onPartChange,
  onConfirmedChange,
  onSave,
  onRequestRemove,
  onCancelRemove,
  onRemove,
}: {
  side: Side
  factoryBacked: boolean
  entry: TrackerCalibrationSide
  draft: TransformDraft
  confirmed: boolean
  saving: boolean
  removing: boolean
  removeKey: string | null
  disabled: boolean
  onPartChange: (part: "pos" | "quat", index: number, value: string) => void
  onConfirmedChange: (value: boolean) => void
  onSave: () => void
  onRequestRemove: (key: string) => void
  onCancelRemove: () => void
  onRemove: (key: string) => void
}) {
  const label = side === "left" ? "Left Mantis" : "Right Mantis"
  const parsed = parseDraft(draft)
  const complete = typeof parsed !== "string"
  // Older serve hosts do not expose the revision-guarded DELETE contract.
  // Only offer removal when the snapshot explicitly advertises removable
  // keys; synthesizing one from a measured entry would lead to a guaranteed
  // 404 against those hosts.
  const overrideKeys = entry.overrideKeys ?? []
  const overrideNeedsAttention = entry.status === "missing" && overrideKeys.length > 0
  const statusCopy = {
    measured: "Measured override saved",
    stale: "Saved convention is stale — re-check required",
    factory: "Verified factory constant active",
    candidate: "CAD candidate only — not calibrated",
    missing: overrideNeedsAttention ? "Saved override needs attention" : "Measurement required",
    unbound: "Active tracker key unavailable",
  }[entry.status]
  const statusClass =
    entry.status === "measured" || entry.status === "factory"
      ? "bg-emerald-400/10 text-emerald-300"
      : entry.status === "candidate" || entry.status === "stale"
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

      {entry.status === "stale" && (
        <p className="text-[11px] leading-relaxed text-amber-200/75">
          This saved value has no Ultimate pose-convention proof or was measured under a different
          convention
          {entry.poseConvention
            ? ` (${entry.poseConvention.quatOrder} quaternion · ${entry.poseConvention.upAxis}-up)`
            : ""}
          . Re-check all axes and the physical overlay under the active convention before confirming
          and resaving it. It cannot authorize production collection as-is.
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
        <span>
          I bench-measured these values for this physical tracker mount under the active pose
          convention.
        </span>
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

      {overrideKeys.length > 0 && removeKey === null && (
        <div className="space-y-2">
          {overrideNeedsAttention && (
            <p className="text-[11px] leading-relaxed text-red-200/75">
              {factoryBacked
                ? "A malformed, legacy, or different-device override is preventing the approved factory transform from becoming active."
                : "The active tracker override is malformed and cannot authorize collection."}{" "}
              Verify the physical mount before removing a saved entry.
            </p>
          )}
          {overrideKeys.map((key) => {
            const ordinaryActiveOverride =
              key === entry.key && (entry.status === "measured" || entry.status === "stale")
            const description =
              key === entry.key
                ? ordinaryActiveOverride
                  ? "Active tracker override"
                  : "Malformed active tracker override"
                : key === "legacy"
                  ? "Legacy unkeyed override"
                  : "Different-device override blocking factory fallback"
            return (
              <div
                key={key}
                className={
                  ordinaryActiveOverride
                    ? "flex flex-col items-start gap-1.5"
                    : "space-y-1.5 rounded-md border border-white/10 bg-black/15 p-2"
                }
              >
                {!ordinaryActiveOverride && (
                  <>
                    <p className="text-[10px] text-white/45">{description}</p>
                    <code className="block break-all text-[10px] text-white/35">{key}</code>
                  </>
                )}
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  onClick={() => onRequestRemove(key)}
                  disabled={disabled || removing}
                >
                  <Trash2 />
                  {ordinaryActiveOverride ? "Remove saved override" : "Remove this override"}
                </Button>
              </div>
            )
          })}
        </div>
      )}
      {removeKey !== null && (
        <div className="space-y-2 rounded-md border border-red-400/25 bg-red-400/[0.05] p-3">
          <p className="text-[11px] leading-relaxed text-red-200/80">
            Remove exactly this saved override? The active tracker identity will be checked again
            before deletion.{" "}
            {factoryBacked
              ? "The approved factory transform will become active only after every blocking entry is gone."
              : "This side will become uncalibrated until a new measured override is saved."}{" "}
            Other tracker families and the other side will not change.
          </p>
          <code className="block break-all text-[10px] text-red-100/65">{removeKey}</code>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onCancelRemove}
              disabled={disabled || removing}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={() => onRemove(removeKey)}
              disabled={disabled || removing}
            >
              {removing ? <Loader2 className="animate-spin" /> : <Trash2 />}
              Yes, remove override
            </Button>
          </div>
        </div>
      )}
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
