import { useState, type ReactNode } from "react"
import { ExternalLink, Menu, X } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { QuickstartButton } from "@/components/quickstart-dialog"
import { cn } from "@/lib/utils"

/**
 * Shared top bar for both routes. ``current`` controls the page label and
 * which cross-link is shown (control panel <-> VR app). Uses plain anchors so
 * switching routes does a full navigation (each route lazy-loads its bundle).
 * ``right`` injects route-specific controls (e.g. the connection pill) just
 * before the Docs / cross-link buttons. On narrow screens the links collapse
 * into a hamburger menu so the bar never overflows the viewport.
 */
const PAGE_LABEL: Record<string, string> = {
  control: "Control Panel",
  vr: "VR",
  diagnostics: "Diagnostics",
}

function NavLinks({ current, itemClass }: { current: string; itemClass?: string }) {
  const item = cn(buttonVariants({ variant: "ghost", size: "sm" }), itemClass)
  return (
    <>
      <a href="https://docs.almond.bot" target="_blank" rel="noreferrer" className={item}>
        Docs
        <ExternalLink />
      </a>
      {current === "control" && <QuickstartButton className={itemClass} />}
      {current !== "control" && (
        <a href="/control" className={item}>
          Control Panel
        </a>
      )}
      {current !== "diagnostics" && (
        <a href="/diagnostics" className={item}>
          Diagnostics
        </a>
      )}
      {current !== "vr" && (
        <a href="/vr" className={item}>
          VR App
        </a>
      )}
    </>
  )
}

export function SiteNav({
  current,
  right,
}: {
  current: "control" | "vr" | "diagnostics"
  right?: ReactNode
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#121212]/85 backdrop-blur-md">
      <div className="relative mx-auto flex h-14 max-w-6xl items-center justify-between gap-2 px-4 sm:h-16 sm:px-6">
        <div className="flex min-w-0 items-center gap-2 sm:gap-3">
          <img src="/almond.svg" alt="Almond" className="h-6 w-6 shrink-0" />
          <span className="font-heading text-base font-semibold tracking-tight whitespace-nowrap">
            Almond Axol
          </span>
          <span className="hidden truncate text-sm text-white/35 min-[420px]:inline">
            {PAGE_LABEL[current]}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {right}
          <nav className="hidden items-center gap-2 md:flex">
            <NavLinks current={current} />
          </nav>
          <button
            type="button"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            className={cn(buttonVariants({ variant: "ghost", size: "icon" }), "size-8 md:hidden")}
          >
            {menuOpen ? <X /> : <Menu />}
          </button>
        </div>
        {menuOpen && (
          <nav
            className="absolute inset-x-0 top-full flex flex-col gap-1 border-b border-white/10 bg-[#121212]/95 p-3 shadow-xl backdrop-blur-md md:hidden"
            onClick={() => setMenuOpen(false)}
          >
            <NavLinks current={current} itemClass="w-full justify-start" />
          </nav>
        )}
      </div>
    </header>
  )
}
