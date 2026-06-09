import { useEffect, useState } from "react"
import { AlertTriangle, Camera, Loader2, Plug, X } from "lucide-react"
import { zedConnect, type ZedLinkStatus, type ZedSpec } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type Cameras = ZedSpec["cameras"]

const EMPTY_CAMERAS: Cameras = { overhead: "", left_arm: "", right_arm: "" }

const CAMERA_SLOTS: { key: keyof Cameras; label: string }[] = [
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
 * Lightweight ZED box link dialog. Verifies the box's `axol serve` is reachable
 * and stores the box URL on the host. Connecting also starts PTP clock sync so
 * the clocks are locked before a task. Any camera serials entered here start
 * streaming once the clocks lock (a task then reuses the live feeds). The PTP
 * interfaces on both machines are derived automatically from the box address.
 */
export function ZedConnectDialog({
  open,
  onClose,
  initial,
  defaultUrl,
  defaultCameras,
  defaultOverheadStereo,
  defaultResolution,
  onConnected,
}: {
  open: boolean
  onClose: () => void
  initial: ZedLinkStatus | null
  /** Persisted box address to prefill when not currently connected. */
  defaultUrl?: string
  /** Persisted camera serials to prefill. */
  defaultCameras?: Cameras
  /** Persisted "overhead is stereo" flag to prefill. */
  defaultOverheadStereo?: boolean
  /** Persisted camera resolution to prefill. */
  defaultResolution?: string
  onConnected: (
    status: ZedLinkStatus,
    url: string,
    cameras: Cameras,
    overheadStereo: boolean,
    resolution: string
  ) => void
}) {
  const [url, setUrl] = useState(initial?.boxUrl || defaultUrl || "")
  const [cameras, setCameras] = useState<Cameras>(defaultCameras ?? EMPTY_CAMERAS)
  const [overheadStereo, setOverheadStereo] = useState(
    initial?.overheadStereo ?? defaultOverheadStereo ?? false
  )
  const [resolution, setResolution] = useState(
    initial?.resolution || defaultResolution || DEFAULT_RESOLUTION
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  async function connect() {
    if (!url.trim()) return
    setBusy(true)
    setError(null)
    const trimmed: Cameras = {
      overhead: cameras.overhead.trim(),
      left_arm: cameras.left_arm.trim(),
      right_arm: cameras.right_arm.trim(),
    }
    try {
      const status = await zedConnect(url.trim(), trimmed, overheadStereo, resolution)
      onConnected(status, url.trim(), trimmed, overheadStereo, resolution)
      onClose()
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-lg rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Camera className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">ZED Box</span>
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
            Syncs clocks with the box, then streams the cameras below once locked.
          </p>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="zed-box-url">ZED Box Address</Label>
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                connect()
              }}
            >
              <Input
                id="zed-box-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="192.168.1.50"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <Button
                type="submit"
                variant="outline"
                size="sm"
                className="shrink-0"
                disabled={busy || !url.trim()}
              >
                {busy ? <Loader2 className="animate-spin" /> : <Plug />}
                Connect
              </Button>
            </form>
            {error && (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertTriangle className="size-3 shrink-0" />
                {error}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-3 border-t border-white/10 pt-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex flex-col gap-0.5">
                <Label>Camera Serials (optional)</Label>
                <p className="text-xs text-white/35">Leave blank to skip camera streaming.</p>
              </div>
              <select
                id="zed-resolution"
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
                      title="Stereo ZED X (both eyes on one stream)"
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
                  value={cameras[slot.key]}
                  inputMode="numeric"
                  spellCheck={false}
                  autoCapitalize="off"
                  autoCorrect="off"
                  onChange={(e) => setCameras((c) => ({ ...c, [slot.key]: e.target.value }))}
                  placeholder="serial"
                  className="max-w-[180px]"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
