import { type SelectHTMLAttributes, forwardRef } from "react"
import { cn } from "@/lib/utils"

/** Styled native select matching the Input primitive. Pass <option>s as children. */
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        "h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground transition-colors outline-none focus-visible:border-ring/70 focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    >
      {children}
    </select>
  )
)
Select.displayName = "Select"

export function SelectOption({ value, label }: { value: string; label?: string }) {
  return (
    <option value={value} className="bg-[#1a1a1a]">
      {label ?? value}
    </option>
  )
}
