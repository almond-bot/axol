import { ExternalLink } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * Shared top bar for both routes. ``current`` controls the page label and
 * which cross-link is shown (control panel <-> VR app). Uses plain anchors so
 * switching routes does a full navigation (each route lazy-loads its bundle).
 */
export function SiteNav({ current }: { current: "control" | "vr" }) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#121212]/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <img src="/almond.svg" alt="Almond" className="h-6 w-6" />
          <span className="font-heading text-base font-semibold tracking-tight">Almond Axol</span>
          <span className="hidden text-sm text-white/35 sm:inline">
            {current === "control" ? "Control Panel" : "VR"}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <a
            href="https://docs.almond.bot"
            target="_blank"
            rel="noreferrer"
            className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}
          >
            Docs
            <ExternalLink />
          </a>
          {current === "control" ? (
            <a href="/vr" className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}>
              VR App
            </a>
          ) : (
            <a href="/control" className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}>
              Control Panel
            </a>
          )}
        </div>
      </div>
    </header>
  )
}
