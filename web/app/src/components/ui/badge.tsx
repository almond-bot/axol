import { type HTMLAttributes } from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

export const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-xs tracking-wider whitespace-nowrap",
  {
    variants: {
      variant: {
        default: "border-[#eff483]/30 bg-[#eff483]/15 text-[#eff483]",
        neutral: "border-white/15 bg-white/[0.04] text-white/60",
        success: "border-emerald-400/30 bg-emerald-400/15 text-emerald-300",
        warning: "border-amber-400/30 bg-amber-400/15 text-amber-300",
        destructive: "border-red-400/30 bg-red-400/15 text-red-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
