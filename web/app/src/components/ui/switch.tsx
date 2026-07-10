import { cn } from "@/lib/utils"

export function Switch({
  checked,
  disabled,
  onChange,
  "aria-label": ariaLabel,
}: {
  checked: boolean
  disabled?: boolean
  onChange: (v: boolean) => void
  "aria-label"?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full border transition-colors disabled:opacity-50",
        checked ? "border-[#eff483]/50 bg-[#eff483]/80" : "border-white/15 bg-white/[0.06]"
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 left-0.5 size-4.5 rounded-full transition-transform",
          checked ? "translate-x-5 bg-[#121212]" : "translate-x-0 bg-white/80"
        )}
      />
    </button>
  )
}
