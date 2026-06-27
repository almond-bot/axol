import { useCallback, useEffect, useState } from "react"
import { AlertTriangle, Camera, Loader2, RotateCw, X } from "lucide-react"
import {
  detectCameras,
  restartCameraDaemon,
  type CameraDevice,
  type CameraSlot,
  type CameraSpec,
  type StereoEyes,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type Serials = CameraSpec["serials"]
type EyeMap = Partial<Record<CameraSlot, StereoEyes>>

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
]

const DEFAULT_RESOLUTION = "SVGA"

// Default eye selection for a freshly-detected stereo slot, matching the
// backend's head/wrist policy: the overhead streams/records both eyes (true
// stereo / depth), a wrist a single (left) eye so it costs like a mono camera.
const defaultEyes = (slot: CameraSlot): StereoEyes => (slot === "overhead" ? "both" : "left")

const eyesLeft = (e: StereoEyes) => e !== "right"
const eyesRight = (e: StereoEyes) => e !== "left"
const eyesFromBools = (left: boolean, right: boolean): StereoEyes =>
  left && right ? "both" : right ? "right" : "left"

/**
 * Local ZED camera setup dialog. The cameras are attached to the machine
 * running `axol serve`, so this assigns each camera slot a serial number (with
 * a detector to list what's plugged in) and configures capture/record settings
 * — no network link to manage. Streaming (the headset feed) and recording (the
 * dataset) are configured separately: each gets its own resolution, and for a
 * stereo camera its own eye selection (e.g. stream both eyes for depth while
 * recording only one). The spec is stored client-side and sent with each op
 * start.
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
  const [streamEyes, setStreamEyes] = useState<EyeMap>(initial.stream_eyes ?? {})
  const [recordEyes, setRecordEyes] = useState<EyeMap>(initial.record_eyes ?? {})

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
  // gate the per-eye controls on it.
  const kindBySerial = new Map((devices ?? []).map((d) => [String(d.serial), d.kind]))
  const isStereo = (slot: CameraSlot) => kindBySerial.get(serials[slot].trim()) === "stereo"

  function toggleEye(
    map: EyeMap,
    set: (m: EyeMap) => void,
    slot: CameraSlot,
    side: "left" | "right"
  ) {
    const cur = map[slot] ?? defaultEyes(slot)
    let left = eyesLeft(cur)
    let right = eyesRight(cur)
    if (side === "left") left = !left
    else right = !right
    // A stereo camera must expose at least one eye; ignore a toggle that would
    // leave neither selected.
    if (!left && !right) return
    set({ ...map, [slot]: eyesFromBools(left, right) })
  }

  function save() {
    // Only persist eye selections for slots currently detected as stereo, so a
    // stale per-eye choice can't linger on a slot that's now mono / unassigned.
    const pickEyes = (map: EyeMap): EyeMap => {
      const out: EyeMap = {}
      for (const { key } of CAMERA_SLOTS) {
        if (isStereo(key) && map[key]) out[key] = map[key]
      }
      return out
    }
    onSave({
      serials: {
        overhead: serials.overhead.trim(),
        left_arm: serials.left_arm.trim(),
        right_arm: serials.right_arm.trim(),
      },
      stream_resolution: streamResolution,
      record_resolution: recordResolution,
      stream_eyes: pickEyes(streamEyes),
      record_eyes: pickEyes(recordEyes),
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
            Assign the ZED cameras connected to this machine to their slots. Collect data and run
            policy need at least one; teleop streams whichever are set to the headset.
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
                >
                  {detecting ? <Loader2 className="animate-spin" /> : <RotateCw />}
                  Refresh
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={restartDaemon}
                  disabled={detecting || restarting}
                  title="Restart the ZED X daemon so cameras plugged in after boot show up"
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
              hint="Capture resolution sent to the headset"
              value={streamResolution}
              onChange={setStreamResolution}
            />
            <ResolutionSelect
              id="camera-record-resolution"
              label="Recording resolution"
              hint="Dataset video is downscaled to this"
              value={recordResolution}
              onChange={setRecordResolution}
            />
          </div>

          <div className="flex flex-col gap-3 border-t border-white/10 pt-4">
            <Label>Camera serials</Label>
            {CAMERA_SLOTS.map((slot) => {
              const kind = kindBySerial.get(serials[slot.key].trim())
              const stereo = kind === "stereo"
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
                  {stereo && (
                    <div className="ml-1 flex flex-wrap items-center gap-x-5 gap-y-1.5 rounded-md border border-white/10 bg-white/[0.02] px-3 py-2">
                      <EyeChoice
                        label="Stream"
                        eyes={streamEyes[slot.key] ?? defaultEyes(slot.key)}
                        onToggle={(side) => toggleEye(streamEyes, setStreamEyes, slot.key, side)}
                      />
                      <EyeChoice
                        label="Record"
                        eyes={recordEyes[slot.key] ?? defaultEyes(slot.key)}
                        onToggle={(side) => toggleEye(recordEyes, setRecordEyes, slot.key, side)}
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

/** Left/right eye checkboxes for one branch (stream or record) of a stereo slot. */
function EyeChoice({
  label,
  eyes,
  onToggle,
}: {
  label: string
  eyes: StereoEyes
  onToggle: (side: "left" | "right") => void
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-medium uppercase tracking-wide text-white/45">{label}</span>
      <EyeBox label="L" checked={eyesLeft(eyes)} onChange={() => onToggle("left")} />
      <EyeBox label="R" checked={eyesRight(eyes)} onChange={() => onToggle("right")} />
    </div>
  )
}

function EyeBox({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: () => void
}) {
  return (
    <label className="flex cursor-pointer items-center gap-1 text-xs text-white/70 select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="size-3.5 accent-[#eff483]"
      />
      {label}
    </label>
  )
}
