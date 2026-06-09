import type { ReactNode } from "react"
import { ExternalLink } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { QuickstartButton } from "@/components/quickstart-dialog"
import { cn } from "@/lib/utils"

/**
 * Shared top bar for both routes. ``current`` controls the page label and
 * which cross-link is shown (control panel <-> VR app). Uses plain anchors so
 * switching routes does a full navigation (each route lazy-loads its bundle).
 * ``right`` injects route-specific controls (e.g. the connection pill) just
 * before the Docs / cross-link buttons.
 */
const PAGE_LABEL: Record<string, string> = {
  control: "Control Panel",
  vr: "VR",
}

export function SiteNav({ current, right }: { current: "control" | "vr"; right?: ReactNode }) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#121212]/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <img src="/almond.svg" alt="Almond" className="h-6 w-6" />
          <span className="font-heading text-base font-semibold tracking-tight">Almond Axol</span>
          <span className="hidden text-sm text-white/35 sm:inline">{PAGE_LABEL[current]}</span>
        </div>
        <div className="flex items-center gap-2">
          {right}
          <a
            href="https://docs.almond.bot"
            target="_blank"
            rel="noreferrer"
            className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}
          >
            Docs
            <ExternalLink />
          </a>
          {current === "control" && <QuickstartButton />}
          {current !== "control" && (
            <a href="/control" className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}>
              Control Panel
            </a>
          )}
          {current !== "vr" && (
            <a href="/vr" className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}>
              VR App
            </a>
          )}
        </div>
      </div>
    </header>
  )
}
