import { useRef, useState, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { padCommand, type Vec3 } from "@/lib/jog"

/** A button that acts only while it is pressed, and always lets go. */
export function HoldButton({
  label,
  icon,
  disabled,
  onHold,
  className,
  "aria-label": ariaLabel,
}: {
  label: string
  icon?: ReactNode
  disabled?: boolean
  onHold: (held: boolean) => void
  className?: string
  "aria-label"?: string
}) {
  const [held, setHeld] = useState(false)
  const set = (value: boolean) => {
    setHeld(value)
    onHold(value)
  }
  return (
    <Button
      type="button"
      variant="outline"
      disabled={disabled}
      aria-pressed={held}
      aria-label={ariaLabel}
      className={cn(
        "h-11 flex-1 touch-none select-none",
        held && "border-[#eff483]/60 bg-[#eff483]/15",
        className
      )}
      onPointerDown={(event) => {
        // Only the primary button: a right-click's menu can swallow the release.
        if (event.button !== 0) return
        event.currentTarget.setPointerCapture(event.pointerId)
        set(true)
      }}
      onContextMenu={(event) => event.preventDefault()}
      onPointerUp={() => set(false)}
      onPointerCancel={() => set(false)}
      onLostPointerCapture={() => held && set(false)}
      onKeyDown={(event) => {
        if ((event.key === " " || event.key === "Enter") && !event.repeat) set(true)
      }}
      onKeyUp={(event) => {
        if (event.key === " " || event.key === "Enter") set(false)
      }}
      onBlur={() => held && set(false)}
    >
      {icon}
      {label}
    </Button>
  )
}

/** A round pad that drives the base while it is held: up is forward, left is left. */
export function DrivePad({
  disabled,
  speed,
  onDrive,
  className,
}: {
  disabled: boolean
  speed: number
  onDrive: (command: Vec3) => void
  /** Sizes the pad (`size-52` by default). */
  className?: string
}) {
  const pad = useRef<HTMLDivElement>(null)
  const [knob, setKnob] = useState<[number, number] | null>(null)
  const update = (clientX: number, clientY: number) => {
    const rect = pad.current?.getBoundingClientRect()
    if (!rect) return
    const radius = rect.width / 2
    const dx = clientX - (rect.left + radius)
    const dy = clientY - (rect.top + rect.height / 2)
    const length = Math.hypot(dx, dy)
    const clip = length > radius ? radius / length : 1
    setKnob([dx * clip, dy * clip])
    onDrive(padCommand(dx, dy, radius, speed))
  }
  const release = () => {
    setKnob(null)
    onDrive([0, 0, 0])
  }
  return (
    <div
      ref={pad}
      role="application"
      aria-label="Drive pad"
      aria-disabled={disabled}
      className={cn(
        "relative size-52 shrink-0 touch-none rounded-full border border-white/15 bg-white/[0.03] select-none",
        disabled ? "opacity-40" : "cursor-grab active:cursor-grabbing",
        className
      )}
      onPointerDown={(event) => {
        if (disabled || event.button !== 0) return
        event.currentTarget.setPointerCapture(event.pointerId)
        update(event.clientX, event.clientY)
      }}
      onContextMenu={(event) => event.preventDefault()}
      onPointerMove={(event) => {
        if (knob) update(event.clientX, event.clientY)
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={() => knob && release()}
    >
      <span className="absolute inset-x-6 top-1/2 h-px bg-white/10" />
      <span className="absolute inset-y-6 left-1/2 w-px bg-white/10" />
      <span className="absolute top-2 left-1/2 -translate-x-1/2 text-[10px] text-white/40">
        Forward
      </span>
      <span className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-white/40">
        Back
      </span>
      <span className="absolute top-1/2 left-2 -translate-y-1/2 text-[10px] text-white/40">
        Left
      </span>
      <span className="absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-white/40">
        Right
      </span>
      <span
        className={cn(
          "absolute top-1/2 left-1/2 size-12 rounded-full border transition-colors",
          knob ? "border-[#eff483]/70 bg-[#eff483]/25" : "border-white/20 bg-white/[0.06]"
        )}
        style={{
          transform: `translate(calc(-50% + ${knob?.[0] ?? 0}px), calc(-50% + ${knob?.[1] ?? 0}px))`,
        }}
      />
    </div>
  )
}
