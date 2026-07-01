import { useCallback, useEffect, useState } from "react"
import { AlertTriangle, Camera, Loader2, RotateCw, X } from "lucide-react"
import {
  detectCameras,
  restartCameraDaemon,
  RESOLUTION_OFF,
  type BranchSel,
  type CameraDevice,
  type CameraSlot,
  type CameraSpec,
  type StereoEyes,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type Serials = CameraSpec["serials"]
type BranchMap = Partial<Record<CameraSlot, BranchSel>>

const EMPTY_SERIALS: Serials = { overhead: "", left_arm: "", right_arm: "" }

const CAMERA_SLOTS: { key: CameraSlot; label: string }[] = [
  { key: "overhead", label: "Overhead" },
  { key: "left_arm", label: "Left arm" },
  { key: "right_arm", label: "Right arm" },
]

const RESOLUTIONS: { value: string; label: string }[] = [
  { value: "SVGA", label: "SVGA (960×600)" },
  { value: "HD1080", label: "HD1080 (1920×1080)" },
  { value: "HD1200", label: "HD1200 (1920×1200)" },
  { value: RESOLUTION_OFF, label: "Off" },
]

const DEFAULT_RESOLUTION = "SVGA"

// Default eye selection for a stereo slot, matching the backend's head/wrist
// policy: the overhead uses both eyes (true stereo / depth), a wrist a single
// (left) eye so it costs like a mono camera.
const defaultEyes = (slot: CameraSlot): StereoEyes => (slot === "overhead" ? "both" : "left")

const eyesLeft = (e: StereoEyes) => e !== "right"
const eyesRight = (e: StereoEyes) => e !== "left"

// A per-branch value is "enabled" unless explicitly false; its eye selection
// (stereo only) is the stored eye name, else the slot default.
const selEnabled = (v: BranchSel | undefined) => v === undefined || v !== false
const selEyes = (v: BranchSel | undefined, slot: CameraSlot): StereoEyes =>
  v === "both" || v === "left" || v === "right" ? v : defaultEyes(slot)

/**
 * Local ZED camera setup dialog. The cameras are attached to the machine
 * running `axol serve`. Streaming (the headset feed) and recording (the
 * dataset) are configured independently: each has its own resolution (or Off to
 * disable the whole branch), and each camera can be individually included in or
 * excluded from streaming and recording — for a stereo camera, down to which
 * eye(s). The spec is stored client-side and sent with each op start.
 */
export function CamerasDialog({
  open,
  onClose,
  initial,
  onSave,
}: {
  open: boolean
  onClose: () => void
  /** Persisted camera spec to prefill. */
  initial: CameraSpec
  onSave: (spec: CameraSpec) => void
}) {
  const [serials, setSerials] = useState<Serials>(initial.serials ?? EMPTY_SERIALS)
  const [streamResolution, setStreamResolution] = useState(
    initial.stream_resolution || initial.resolution || DEFAULT_RESOLUTION
  )
  const [recordResolution, setRecordResolution] = useState(
    initial.record_resolution || DEFAULT_RESOLUTION
  )
  const [stream, setStream] = useState<BranchMap>(initial.stream ?? {})
  const [record, setRecord] = useState<BranchMap>(initial.record ?? {})

  const [devices, setDevices] = useState<CameraDevice[] | null>(null)
  const [detectError, setDetectError] = useState<string | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [restarting, setRestarting] = useState(false)

  const refresh = useCallback(async () => {
    setDetecting(true)
    try {
      const result = await detectCameras()
      setDevices(result.devices)
      setDetectError(result.error)
    } catch (e) {
      setDevices(null)
      setDetectError(String(e).replace(/^Error:\s*/, ""))
    } finally {
      setDetecting(false)
    }
  }, [])

  // Detect once when the dialog opens; after that it's manual (Refresh /
  // Restart daemon). Keyed on `open` only — `onClose` is an inline prop that
  // changes identity on every parent render, so it must not retrigger this.
  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  async function restartDaemon() {
    setRestarting(true)
    try {
      const result = await restartCameraDaemon()
      if (result.error) setDetectError(result.error)
      else await refresh()
    } catch (e) {
      setDetectError(String(e).replace(/^Error:\s*/, ""))
    } finally {
      setRestarting(false)
    }
  }

  // Stereo vs mono is determined from the detected device, so the operator
  // never flags it: surface the detected kind next to each assigned slot and
  // shape the per-camera controls (mono = a single toggle; stereo = L/R).
  const kindBySerial = new Map((devices ?? []).map((d) => [String(d.serial), d.kind]))

  const streamingOn = streamResolution !== RESOLUTION_OFF
  const recordingOn = recordResolution !== RESOLUTION_OFF

  // Toggle one camera's participation in a branch. Mono flips on/off; stereo
  // flips the given eye, dropping to "off" when neither eye is left selected.
  function toggle(
    map: BranchMap,
    set: (m: BranchMap) => void,
    slot: CameraSlot,
    stereo: boolean,
    side?: "left" | "right"
  ) {
    const cur = map[slot]
    if (!stereo) {
      set({ ...map, [slot]: !selEnabled(cur) })
      return
    }
    const enabled = selEnabled(cur)
    const eyes = selEyes(cur, slot)
    let left = enabled && eyesLeft(eyes)
    let right = enabled && eyesRight(eyes)
    if (side === "left") left = !left
    else right = !right
    const next: BranchSel = left && right ? "both" : left ? "left" : right ? "right" : false
    set({ ...map, [slot]: next })
  }

  function save() {
    // Persist exactly what each control displays. A slot the operator never
    // touched still shows a default (e.g. overhead streams "both"); writing that
    // default explicitly keeps the saved spec matching the dialog, so the
    // backend never falls back to a different value (streaming would otherwise
    // default to the recorded eyes). Materialized only for slots whose kind is
    // known, so mono saves a boolean and stereo an eye selection.
    const materialize = (map: BranchMap, slot: CameraSlot, stereo: boolean): BranchSel =>
      !selEnabled(map[slot]) ? false : stereo ? selEyes(map[slot], slot) : true
    const outStream: BranchMap = { ...stream }
    const outRecord: BranchMap = { ...record }
    for (const { key } of CAMERA_SLOTS) {
      const kind = kindBySerial.get(serials[key].trim())
      if (!kind) continue // unknown kind / unassigned: leave backend defaults
      const stereo = kind === "stereo"
      outStream[key] = materialize(stream, key, stereo)
      outRecord[key] = materialize(record, key, stereo)
    }
    onSave({
      serials: {
        overhead: serials.overhead.trim(),
        left_arm: serials.left_arm.trim(),
        right_arm: serials.right_arm.trim(),
      },
      stream_resolution: streamResolution,
      record_resolution: recordResolution,
      stream: outStream,
      record: outRecord,
    })
    onClose()
  }

  const assigned = new Set(
    Object.values(serials)
      .map((s) => s.trim())
      .filter(Boolean)
  )

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-lg rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Camera className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">Cameras</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-white/40 transition-colors hover:text-white/80"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="flex flex-col gap-5 p-5">
          <p className="text-xs text-white/45">
            Assign the ZED cameras connected to this machine to their slots, then choose what each
            one is used for. Collect data and run policy need at least one camera recording; teleop
            streams whichever are set to the headset.
          </p>

          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <Label>Detected cameras</Label>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={refresh}
                  disabled={detecting || restarting}
                  title="Re-query the ZED daemon's current device list. The daemon only scans the GMSL links at startup, so plugging/unplugging a camera won't show here until you restart the daemon."
                >
                  {detecting ? <Loader2 className="animate-spin" /> : <RotateCw />}
                  Refresh
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={restartDaemon}
                  disabled={detecting || restarting}
                  title="Restart the ZED X daemon so it re-scans the GMSL links — needed to pick up any camera plugged in or unplugged since boot"
                >
                  {restarting ? <Loader2 className="animate-spin" /> : null}
                  Restart daemon
                </Button>
              </div>
            </div>
            {detectError ? (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertTriangle className="size-3 shrink-0" />
                {detectError}
              </p>
            ) : devices == null ? (
              <p className="text-xs text-white/35">Detecting…</p>
            ) : devices.length === 0 ? (
              <p className="text-xs text-white/35">
                No cameras detected. Check the GMSL cables, or restart the daemon if a camera was
                plugged in after boot.
              </p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {devices.map((d) => (
                  <span
                    key={d.serial}
                    className={
                      "rounded-md border px-2 py-1 font-mono text-xs " +
                      (assigned.has(String(d.serial))
                        ? "border-[#eff483]/40 bg-[#eff483]/10 text-[#eff483]"
                        : "border-white/15 bg-white/[0.03] text-white/65")
                    }
                    title={`${d.model} (${d.kind})`}
                  >
                    {d.serial}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 border-t border-white/10 pt-4">
            <ResolutionSelect
              id="camera-stream-resolution"
              label="Streaming resolution"
              hint="Capture resolution sent to the headset (Off disables streaming)"
              value={streamResolution}
              onChange={setStreamResolution}
            />
            <ResolutionSelect
              id="camera-record-resolution"
              label="Recording resolution"
              hint="Dataset video is downscaled to this (Off disables recording)"
              value={recordResolution}
              onChange={setRecordResolution}
            />
          </div>

          <div className="flex flex-col gap-3 border-t border-white/10 pt-4">
            <Label>Cameras</Label>
            {CAMERA_SLOTS.map((slot) => {
              const kind = kindBySerial.get(serials[slot.key].trim())
              const stereo = kind === "stereo"
              const assignedSlot = serials[slot.key].trim() !== ""
              return (
                <div key={slot.key} className="flex flex-col gap-2">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                      <Label className="text-white/70">{slot.label}</Label>
                      {kind && (
                        <span
                          className="rounded border border-white/15 bg-white/[0.03] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white/45"
                          title={
                            stereo
                              ? "Stereo ZED X (both eyes from one grab) — detected automatically"
                              : "Mono ZED-X One — detected automatically"
                          }
                        >
                          {kind}
                        </span>
                      )}
                    </div>
                    <Input
                      value={serials[slot.key]}
                      inputMode="numeric"
                      spellCheck={false}
                      autoCapitalize="off"
                      autoCorrect="off"
                      onChange={(e) => setSerials((c) => ({ ...c, [slot.key]: e.target.value }))}
                      placeholder="serial"
                      className="max-w-[180px]"
                    />
                  </div>
                  {assignedSlot && kind && (
                    <div className="ml-1 flex flex-wrap items-center gap-x-5 gap-y-1.5 rounded-md border border-white/10 bg-white/[0.02] px-3 py-2">
                      <BranchControl
                        label="Stream"
                        stereo={stereo}
                        value={stream[slot.key]}
                        slot={slot.key}
                        disabled={!streamingOn}
                        onToggle={(side) => toggle(stream, setStream, slot.key, stereo, side)}
                      />
                      <BranchControl
                        label="Record"
                        stereo={stereo}
                        value={record[slot.key]}
                        slot={slot.key}
                        disabled={!recordingOn}
                        onToggle={(side) => toggle(record, setRecord, slot.key, stereo, side)}
                      />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <div className="flex justify-end border-t border-white/10 pt-4">
            <Button onClick={save}>Save</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ResolutionSelect({
  id,
  label,
  hint,
  value,
  onChange,
}: {
  id: string
  label: string
  hint: string
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor={id}>{label}</Label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        title={hint}
        className="h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70"
      >
        {RESOLUTIONS.map((r) => (
          <option key={r.value} value={r.value} className="bg-[#1a1a1a]">
            {r.label}
          </option>
        ))}
      </select>
      <span className="text-[10px] text-white/35">{hint}</span>
    </div>
  )
}

/** One branch (Stream/Record) toggle for a camera: mono = a single checkbox;
 * stereo = L/R checkboxes (none checked = the camera is off for this branch). */
function BranchControl({
  label,
  stereo,
  value,
  slot,
  disabled,
  onToggle,
}: {
  label: string
  stereo: boolean
  value: BranchSel | undefined
  slot: CameraSlot
  disabled: boolean
  onToggle: (side?: "left" | "right") => void
}) {
  const enabled = !disabled && selEnabled(value)
  const eyes = selEyes(value, slot)
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-medium uppercase tracking-wide text-white/45">{label}</span>
      {stereo ? (
        <>
          <EyeBox
            label="L"
            checked={enabled && eyesLeft(eyes)}
            disabled={disabled}
            onChange={() => onToggle("left")}
          />
          <EyeBox
            label="R"
            checked={enabled && eyesRight(eyes)}
            disabled={disabled}
            onChange={() => onToggle("right")}
          />
        </>
      ) : (
        <EyeBox label="On" checked={enabled} disabled={disabled} onChange={() => onToggle()} />
      )}
    </div>
  )
}

function EyeBox({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string
  checked: boolean
  disabled?: boolean
  onChange: () => void
}) {
  return (
    <label
      className={
        "flex items-center gap-1 text-xs select-none " +
        (disabled ? "cursor-not-allowed text-white/25" : "cursor-pointer text-white/70")
      }
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={onChange}
        className="size-3.5 accent-[#eff483]"
      />
      {label}
    </label>
  )
}
