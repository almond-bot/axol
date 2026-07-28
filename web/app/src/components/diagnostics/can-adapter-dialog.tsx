import { useEffect, useMemo, useState } from "react"
import { Loader2, Plug, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"
import {
  fetchCanInterfaces,
  type CanInterface,
  type RobotChannels,
} from "@/lib/supervisor"

type ArmsMode = "both" | "left" | "right"

const ARMS_MODES: { key: ArmsMode; label: string }[] = [
  { key: "both", label: "Both arms" },
  { key: "left", label: "Left only" },
  { key: "right", label: "Right only" },
]

const DEFAULTS: Record<"left" | "right", string> = {
  left: "can_alm_axol_l",
  right: "can_alm_axol_r",
}

function initialMode(channels: RobotChannels | null | undefined): ArmsMode {
  if (channels && channels.left && !channels.right) return "left"
  if (channels && channels.right && !channels.left) return "right"
  return "both"
}

/**
 * The arm→CAN-adapter mapping for the robot link — opened from the CAN
 * enable button (and the robot-link banners) and changeable at any time by
 * reopening it. One interface is enough for a single arm; running both arms
 * asks for both. Save & connect persists the mapping on the host, brings the
 * interfaces up and (re)connects the link; every diagnostic, calibration
 * tool and operation then follows it — an arm without an adapter is skipped
 * automatically, and interface names never need to be retyped per run.
 */
export function CanAdapterDialog({
  channels,
  busy,
  onConnect,
  onClose,
}: {
  /** Currently configured interfaces (prefills the form); null when unknown. */
  channels: RobotChannels | null | undefined
  busy: boolean
  onConnect: (channels: RobotChannels) => void
  onClose: () => void
}) {
  const [mode, setMode] = useState<ArmsMode>(() => initialMode(channels))
  const [left, setLeft] = useState(channels?.left ?? DEFAULTS.left)
  const [right, setRight] = useState(channels?.right ?? DEFAULTS.right)
  const [detected, setDetected] = useState<CanInterface[] | null>(null)

  useEffect(() => {
    let active = true
    fetchCanInterfaces()
      .then(({ interfaces }) => {
        if (active) setDetected(interfaces)
      })
      .catch(() => {
        if (active) setDetected([])
      })
    return () => {
      active = false
    }
  }, [])

  const wantLeft = mode !== "right"
  const wantRight = mode !== "left"
  const valid = (!wantLeft || left.trim() !== "") && (!wantRight || right.trim() !== "")

  const selection: RobotChannels = useMemo(
    () => ({
      left: wantLeft ? left.trim() : null,
      right: wantRight ? right.trim() : null,
    }),
    [wantLeft, wantRight, left, right]
  )

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
            <h3 className="font-heading text-base font-semibold">CAN adapter</h3>
            <p className="text-sm leading-relaxed text-white/45">
              Pick the SocketCAN interface of the adapter driving each arm — one adapter
              is enough for a single arm (the other arm is then skipped everywhere). The
              mapping is saved and applied to every diagnostic, calibration tool and
              operation; reopen this dialog to change it. The Axol hub adapter&apos;s own
              interfaces are the prefilled defaults.
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

        <div className="flex self-start overflow-hidden rounded-md border border-white/10">
          {ARMS_MODES.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMode(m.key)}
              className={cn(
                "px-3 py-1.5 text-xs transition-colors",
                mode === m.key
                  ? "bg-[#eff483]/15 text-[#eff483]"
                  : "text-white/50 hover:bg-white/[0.05]"
              )}
            >
              {m.label}
            </button>
          ))}
        </div>

        {detected != null && detected.length === 0 && (
          <p className="rounded-md border border-amber-400/25 bg-amber-400/[0.05] p-2.5 text-xs leading-relaxed text-amber-100/90">
            No CAN interfaces were detected on the host — plug in a CAN adapter, or type
            the interface name it will have.
          </p>
        )}

        {wantLeft && (
          <ChannelField
            label="Left arm interface"
            value={left}
            detected={detected ?? []}
            disabled={busy}
            onChange={setLeft}
          />
        )}
        {wantRight && (
          <ChannelField
            label="Right arm interface"
            value={right}
            detected={detected ?? []}
            disabled={busy}
            onChange={setRight}
          />
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" onClick={() => onConnect(selection)} disabled={busy || !valid}>
            {busy ? <Loader2 className="animate-spin" /> : <Plug />} Save &amp; connect
          </Button>
        </div>
      </Card>
    </div>
  )
}

/** One arm's interface: free-text input plus the detected interfaces as chips. */
function ChannelField({
  label,
  value,
  detected,
  disabled,
  onChange,
}: {
  label: string
  value: string
  detected: CanInterface[]
  disabled: boolean
  onChange: (value: string) => void
}) {
  const fieldId = `can-iface-${label.toLowerCase().replace(/\s+/g, "-")}`
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={fieldId}>{label}</Label>
      <Input
        id={fieldId}
        value={value}
        placeholder="e.g. can0"
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        spellCheck={false}
      />
      {detected.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
          <span className="text-xs text-white/35">Detected:</span>
          {detected.map((iface) => (
            <button
              key={iface.name}
              type="button"
              disabled={disabled}
              onClick={() => onChange(iface.name)}
              title={iface.up ? "interface is up" : "interface is down"}
              className={cn(
                "rounded-md border px-2 py-0.5 font-mono text-xs transition-colors",
                value === iface.name
                  ? "border-[#eff483]/50 bg-[#eff483]/10 text-[#eff483]"
                  : "border-white/10 text-white/60 hover:border-white/25"
              )}
            >
              {iface.name}
              <span className={cn("ml-1.5", iface.up ? "text-emerald-300/80" : "text-white/30")}>
                {iface.up ? "up" : "down"}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
