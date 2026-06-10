import { useCallback, useEffect, useState } from "react"
import { AlertTriangle, Camera, Loader2, RotateCw, X } from "lucide-react"
import {
  detectCameras,
  restartCameraDaemon,
  type CameraDevice,
  type CameraSpec,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type Serials = CameraSpec["serials"]

const EMPTY_SERIALS: Serials = { overhead: "", left_arm: "", right_arm: "" }

const CAMERA_SLOTS: { key: keyof Serials; label: string }[] = [
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

/**
 * Local ZED camera setup dialog. The cameras are attached to the machine
 * running `axol serve`, so this only assigns each camera slot a serial number
 * (with a detector to list what's plugged in) — no network link to manage.
 * The spec is stored client-side and sent along with each op start.
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
  const [overheadStereo, setOverheadStereo] = useState(initial.overheadStereo ?? false)
  const [resolution, setResolution] = useState(initial.resolution || DEFAULT_RESOLUTION)

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

  function save() {
    onSave({
      serials: {
        overhead: serials.overhead.trim(),
        left_arm: serials.left_arm.trim(),
        right_arm: serials.right_arm.trim(),
      },
      overheadStereo,
      resolution,
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
            policy need all three; teleop streams whichever are set to the headset.
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

          <div className="flex flex-col gap-3 border-t border-white/10 pt-4">
            <div className="flex items-center justify-between gap-4">
              <Label>Camera Serials</Label>
              <select
                id="camera-resolution"
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                title="Capture resolution for all cameras"
                className="h-9 w-full max-w-[180px] shrink-0 rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70"
              >
                {RESOLUTIONS.map((r) => (
                  <option key={r.value} value={r.value} className="bg-[#1a1a1a]">
                    {r.label}
                  </option>
                ))}
              </select>
            </div>
            {CAMERA_SLOTS.map((slot) => (
              <div key={slot.key} className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <Label className="text-white/70">{slot.label}</Label>
                  {slot.key === "overhead" && (
                    <label
                      className="flex cursor-pointer items-center gap-1.5 text-white/55"
                      title="Stereo ZED X (both eyes from one grab)"
                    >
                      <input
                        type="checkbox"
                        checked={overheadStereo}
                        onChange={(e) => setOverheadStereo(e.target.checked)}
                        className="size-3.5 accent-[#eff483]"
                      />
                      <span className="text-xs">Stereo</span>
                    </label>
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
            ))}
          </div>

          <div className="flex justify-end border-t border-white/10 pt-4">
            <Button onClick={save}>Save</Button>
          </div>
        </div>
      </div>
    </div>
  )
}
