import { useState, type ReactNode } from "react"
import { ExternalLink, Menu, Rocket, X } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { QuickstartDialog } from "@/components/quickstart-dialog"
import { type ServerPage } from "@/lib/server-pages"
import { serverPageHref } from "@/lib/supervisor"
import { cn } from "@/lib/utils"

/**
 * Shared top bar for both routes. ``current`` controls the page label and
 * which cross-link is shown (control panel <-> VR app). Uses plain anchors so
 * switching routes does a full navigation (each route lazy-loads its bundle).
 * ``right`` injects route-specific controls (e.g. the connection pill) just
 * before the Docs / cross-link buttons. ``pages`` are the backend-served
 * pages ``/api/info`` advertises (lib/server-pages.ts): they link against the
 * connected server's origin, so when the panel runs on another host they open
 * in a new tab and keep this one connected. On narrow screens the links
 * collapse into a hamburger menu so the bar never overflows the viewport.
 */
const PAGE_LABEL: Record<string, string> = {
  control: "Control Panel",
  vr: "VR",
  diagnostics: "Diagnostics",
}

function NavLinks({
  current,
  pages,
  onQuickstart,
  itemClass,
}: {
  current: string
  pages: ServerPage[]
  onQuickstart: () => void
  itemClass?: string
}) {
  const item = cn(buttonVariants({ variant: "ghost", size: "sm" }), itemClass)
  return (
    <>
      {pages.map((page) => {
        const href = serverPageHref(page.path)
        const external = href !== page.path
        return (
          <a
            key={page.path}
            href={href}
            title={page.description}
            target={external ? "_blank" : undefined}
            rel={external ? "noreferrer" : undefined}
            className={item}
          >
            {page.label}
            {external && <ExternalLink />}
          </a>
        )
      })}
      <a href="https://docs.almond.bot" target="_blank" rel="noreferrer" className={item}>
        Docs
        <ExternalLink />
      </a>
      {current === "control" && (
        <button type="button" onClick={onQuickstart} className={item}>
          <Rocket />
          Quickstart
        </button>
      )}
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
  pages = [],
}: {
  current: "control" | "vr" | "diagnostics"
  right?: ReactNode
  /** Backend-served pages to link (``ServerInfo.pages``); none by default. */
  pages?: ServerPage[]
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  // Owned here, not by the menu item: closing the menu unmounts its contents,
  // which must not take an opening Quickstart dialog down with it.
  const [quickstartOpen, setQuickstartOpen] = useState(false)
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#121212]/85 pt-[env(safe-area-inset-top)] backdrop-blur-md">
      <div className="safe-x relative mx-auto flex h-14 max-w-6xl items-center justify-between gap-2 sm:h-16">
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
            <NavLinks
              current={current}
              pages={pages}
              onQuickstart={() => setQuickstartOpen(true)}
            />
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
            className="safe-x absolute inset-x-0 top-full flex flex-col gap-1 border-b border-white/10 bg-[#121212]/95 py-3 shadow-xl backdrop-blur-md md:hidden"
            onClick={() => setMenuOpen(false)}
          >
            <NavLinks
              current={current}
              pages={pages}
              onQuickstart={() => setQuickstartOpen(true)}
              itemClass="w-full justify-start"
            />
          </nav>
        )}
      </div>
      <QuickstartDialog open={quickstartOpen} onClose={() => setQuickstartOpen(false)} />
    </header>
  )
}
