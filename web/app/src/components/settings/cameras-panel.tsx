import { useState } from "react"
import { CameraOff, Loader2, RotateCw } from "lucide-react"
import {
  apiUrl,
  restartCameraDaemon,
  RESOLUTION_OFF,
  type BranchSel,
  type CameraDevice,
  type CameraSlot,
  type CameraSpec,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectOption } from "@/components/ui/select"
import { useToast } from "@/components/ui/toast"
import { eyesLeft, eyesRight, selEnabled, selEyes } from "@/lib/camera-spec"
import { cn } from "@/lib/utils"

type BranchMap = Partial<Record<CameraSlot, BranchSel>>

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

/**
 * Local ZED camera setup (the Cameras tab of the Settings dialog). The cameras
 * are attached to the machine running `axol serve`. Streaming (the headset
 * feed) and recording (the dataset) are configured independently: each has its
 * own resolution (or Off to disable the whole branch), and each camera can be
 * individually included in or excluded from streaming and recording — for a
 * stereo camera, down to which eye(s).
 *
 * Controlled component: edits flow up through `onChange` and are persisted by
 * the dialog's Save.
 */
export function CamerasPanel({
  spec,
  onChange,
  devices,
  detecting,
  onRefresh,
}: {
  spec: CameraSpec
  onChange: (spec: CameraSpec) => void
  /** Detected ZED devices (shared with the badge; null until first detection). */
  devices: CameraDevice[] | null
  /** A shared detection is currently in flight. */
  detecting: boolean
  /** Re-run detection, updating the shared state (and the badge). */
  onRefresh: () => void
}) {
  const [restarting, setRestarting] = useState(false)
  // Cache-buster for the per-camera preview frames; bumped by Refresh so the
  // operator can grab fresh frames after moving a camera.
  const [previewNonce, setPreviewNonce] = useState(0)
  const toast = useToast()

  function refresh() {
    onRefresh()
    setPreviewNonce((n) => n + 1)
  }

  const serials = spec.serials
  const streamResolution = spec.stream_resolution || spec.resolution || "SVGA"
  const recordResolution = spec.record_resolution || "SVGA"
  const stream = spec.stream ?? {}
  const record = spec.record ?? {}

  async function restartDaemon() {
    setRestarting(true)
    try {
      const result = await restartCameraDaemon()
      if (result.error) toast.error(result.error)
      else refresh()
    } catch (e) {
      toast.error(String(e).replace(/^Error:\s*/, ""))
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
    key: "stream" | "record",
    slot: CameraSlot,
    stereo: boolean,
    side?: "left" | "right"
  ) {
    const cur = map[slot]
    let next: BranchSel
    if (!stereo) {
      next = !selEnabled(cur)
    } else {
      const enabled = selEnabled(cur)
      const eyes = selEyes(cur, slot)
      let left = enabled && eyesLeft(eyes)
      let right = enabled && eyesRight(eyes)
      if (side === "left") left = !left
      else right = !right
      next = left && right ? "both" : left ? "left" : right ? "right" : false
    }
    onChange({ ...spec, [key]: { ...map, [slot]: next } })
  }

  return (
    <div className="flex flex-col gap-5">
      <p className="text-xs text-white/45">
        Assign the ZED cameras connected to this machine to their slots, then choose what each one
        is used for. Collect data and run policy need at least one camera recording; teleop streams
        whichever are set to the headset.
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
              title="Re-query the ZED daemon's current device list and grab fresh preview frames. The daemon only scans the GMSL links at startup, so plugging/unplugging a camera won't show here until you restart the daemon."
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
        {devices == null ? (
          <p className="text-xs text-white/35">Detecting…</p>
        ) : devices.length === 0 ? (
          <p className="text-xs text-white/35">
            No cameras detected. Check the GMSL cables, or restart the daemon if a camera was
            plugged in after boot.
          </p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {devices.map((d) => {
              const slot = CAMERA_SLOTS.find(({ key }) => serials[key].trim() === String(d.serial))
              return (
                <div
                  key={d.serial}
                  className={cn(
                    "flex flex-col gap-2 rounded-lg border p-2",
                    slot
                      ? "border-[#eff483]/35 bg-[#eff483]/[0.05]"
                      : "border-white/10 bg-white/[0.02]"
                  )}
                >
                  <CameraPreview serial={d.serial} nonce={previewNonce} className="aspect-video" />
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-0.5">
                    <span className="font-mono text-xs text-white/85">{d.serial}</span>
                    <KindBadge kind={d.kind} />
                    <span className="text-[11px] text-white/45">{humanModel(d.model)}</span>
                    {slot && (
                      <span className="ml-auto text-[11px] text-[#eff483]">→ {slot.label}</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 border-t border-white/10 pt-4">
        <ResolutionSelect
          id="camera-stream-resolution"
          label="Streaming resolution"
          hint="Capture resolution sent to the headset (Off disables streaming)"
          value={streamResolution}
          onChange={(v) => onChange({ ...spec, stream_resolution: v })}
        />
        <ResolutionSelect
          id="camera-record-resolution"
          label="Recording resolution"
          hint="Dataset video is downscaled to this (Off disables recording)"
          value={recordResolution}
          onChange={(v) => onChange({ ...spec, record_resolution: v })}
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
                <div className="flex min-w-0 items-center gap-2.5">
                  {assignedSlot && kind && (
                    <CameraPreview
                      serial={Number(serials[slot.key].trim())}
                      nonce={previewNonce}
                      className="h-10 w-16 shrink-0"
                    />
                  )}
                  <Label className="text-white/70">{slot.label}</Label>
                  {kind && <KindBadge kind={kind} />}
                </div>
                <Input
                  value={serials[slot.key]}
                  inputMode="numeric"
                  spellCheck={false}
                  autoCapitalize="off"
                  autoCorrect="off"
                  onChange={(e) =>
                    onChange({ ...spec, serials: { ...serials, [slot.key]: e.target.value } })
                  }
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
                    onToggle={(side) => toggle(stream, "stream", slot.key, stereo, side)}
                  />
                  <BranchControl
                    label="Record"
                    stereo={stereo}
                    value={record[slot.key]}
                    slot={slot.key}
                    disabled={!recordingOn}
                    onToggle={(side) => toggle(record, "record", slot.key, stereo, side)}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** SDK model strings arrive like "MODEL.ZED_XONE_GS" — humanize for display. */
function humanModel(model: string): string {
  return (
    model
      .replace(/^MODEL\.?/i, "")
      .replace(/_/g, " ")
      .trim() || model
  )
}

function KindBadge({ kind }: { kind: string }) {
  return (
    <span
      className="rounded border border-white/15 bg-white/[0.03] px-1.5 py-0.5 text-[10px] tracking-wide text-white/55 uppercase"
      title={
        kind === "stereo"
          ? "Stereo ZED X (both eyes from one grab) — detected automatically"
          : "Mono ZED-X One — detected automatically"
      }
    >
      {kind}
    </span>
  )
}

/**
 * A live frame from one camera (GET /api/cameras/preview/<serial>), so the
 * operator can see which physical camera a serial belongs to. Best-effort: a
 * busy host (operation running) or capture failure shows a quiet placeholder.
 */
function CameraPreview({
  serial,
  nonce,
  className,
}: {
  serial: number
  nonce: number
  className?: string
}) {
  const [state, setState] = useState<"loading" | "ok" | "error">("loading")
  const [loadedKey, setLoadedKey] = useState("")
  const key = `${serial}:${nonce}`
  if (key !== loadedKey) {
    setLoadedKey(key)
    setState("loading")
  }
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md border border-white/10 bg-black/40",
        className
      )}
    >
      {state !== "error" && (
        <img
          src={apiUrl(`/api/cameras/preview/${serial}?v=${nonce}`)}
          alt={`Camera ${serial} preview`}
          className="size-full object-cover"
          onLoad={() => setState("ok")}
          onError={() => setState("error")}
        />
      )}
      {state !== "ok" && (
        <div className="absolute inset-0 flex items-center justify-center">
          {state === "loading" ? (
            <Loader2 className="size-4 animate-spin text-white/25" />
          ) : (
            <CameraOff className="size-4 text-white/20" />
          )}
        </div>
      )}
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
      <Select id={id} value={value} onChange={(e) => onChange(e.target.value)} title={hint}>
        {RESOLUTIONS.map((r) => (
          <SelectOption key={r.value} value={r.value} label={r.label} />
        ))}
      </Select>
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
