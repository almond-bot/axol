import { cn } from "@/lib/utils"

/**
 * Range slider with a live value readout. Uses the native range input (styled
 * via the accent color) so it works with keyboard / touch out of the box.
 */
export function Slider({
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  format = (v) => String(v),
  className,
  "aria-label": ariaLabel,
}: {
  value: number
  min: number
  max: number
  step: number
  disabled?: boolean
  onChange: (v: number) => void
  /** Render the readout (e.g. add units / fix decimals). */
  format?: (v: number) => string
  className?: string
  "aria-label"?: string
}) {
  return (
    <div className={cn("flex w-full items-center gap-3", className)}>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#eff483] disabled:cursor-not-allowed disabled:opacity-50"
      />
      <span className="w-14 shrink-0 text-right font-mono text-xs text-white/70 tabular-nums">
        {format(value)}
      </span>
    </div>
  )
}
