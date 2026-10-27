import { useEffect, useState, type ComponentType, type ReactNode } from "react"
import { Link, NavLink, useLocation } from "react-router-dom"
import {
  Activity,
  BookOpen,
  ChevronLeft,
  Cpu,
  Database,
  DatabaseZap,
  Headset,
  Glasses,
  LayoutDashboard,
  Menu,
  Play,
  Rocket,
  ScrollText,
  Settings2,
  X,
} from "lucide-react"
import { QuickstartDialog } from "@/components/quickstart-dialog"
import { STATUS_DOT_CLASS, type StatusDot } from "@/lib/jelly-view"
import { cn } from "@/lib/utils"

/**
 * The app frame shared by the control panel and the diagnostics dashboard:
 * a sidebar with the main sections (a slide-out drawer below `lg`), and a
 * page header with a title, actions and the connection status.
 *
 * Navigation between sections is client-side, so the control panel's state
 * (connections, the running operation, its logs) survives moving between
 * its pages. The VR app is a plain link because it loads its own 3D bundle.
 */

type NavItem = {
  to: string
  label: string
  icon: ComponentType<{ className?: string }>
  /** Matches only this exact path (Overview), not its children. */
  end?: boolean
}

const MAIN_NAV: NavItem[] = [
  { to: "/control", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/control/devices", label: "Devices", icon: Cpu },
  { to: "/control/operations", label: "Operations", icon: Play },
  { to: "/control/datasets", label: "Datasets", icon: Database },
  { to: "/control/request-data", label: "Request data", icon: DatabaseZap },
  { to: "/control/request-teleoperator", label: "Request teleoperator", icon: Headset },
  { to: "/control/settings", label: "Settings", icon: Settings2 },
]

const TOOLS_NAV: NavItem[] = [
  { to: "/diagnostics", label: "Diagnostics", icon: Activity },
  { to: "/control/logs", label: "Logs", icon: ScrollText },
]

/** Live state shown next to a nav item (e.g. a running operation). */
export type NavBadges = Partial<Record<string, "running" | "attention">>

export type HostSummary = {
  dot: StatusDot
  label: string
  /** What the robot side looks like, after the host ("Axol ready"). */
  detail?: string | null
  version?: string | null
  /** Opens the connection details; the pill is a plain status otherwise. */
  onClick?: () => void
}

/** One sub-page of a section, listed under it in the sidebar. */
export type SubNavItem = { to: string; label: string }

function SidebarLink({
  item,
  badge,
  onNavigate,
}: {
  item: NavItem
  badge?: "running" | "attention"
  onNavigate?: () => void
}) {
  const Icon = item.icon
  return (
    <NavLink
      to={item.to}
      end={item.end}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          "group flex h-9 items-center gap-3 rounded-lg px-3 text-sm transition-colors",
          isActive
            ? "bg-white/[0.07] text-white"
            : "text-white/55 hover:bg-white/[0.04] hover:text-white/90"
        )
      }
    >
      {({ isActive }) => (
        <>
          <Icon
            className={cn(
              "size-4 shrink-0",
              isActive ? "text-[#eff483]" : "text-white/40 group-hover:text-white/70"
            )}
          />
          <span className="min-w-0 flex-1 truncate">{item.label}</span>
          {badge === "running" && (
            <span
              className="size-2 shrink-0 animate-pulse rounded-full bg-emerald-400"
              title="Running"
            />
          )}
          {badge === "attention" && (
            <span className="size-2 shrink-0 rounded-full bg-amber-400" title="Needs attention" />
          )}
        </>
      )}
    </NavLink>
  )
}

function SubNav({ items, onNavigate }: { items: SubNavItem[]; onNavigate?: () => void }) {
  return (
    <div className="ml-[1.375rem] flex flex-col gap-0.5 border-l border-white/10 py-1 pl-3">
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "rounded-md px-2 py-1 text-[0.8rem] transition-colors",
              isActive ? "text-[#eff483]" : "text-white/45 hover:text-white/85"
            )
          }
        >
          {item.label}
        </NavLink>
      ))}
    </div>
  )
}

function SidebarContents({
  badges,
  subNav,
  onNavigate,
  onQuickstart,
}: {
  badges?: NavBadges
  subNav?: { under: string; items: SubNavItem[] }
  onNavigate?: () => void
  onQuickstart: () => void
}) {
  const footerLink =
    "flex h-8 items-center gap-3 rounded-lg px-3 text-[0.8rem] text-white/45 transition-colors hover:bg-white/[0.04] hover:text-white/85"
  const renderItem = (item: NavItem) => (
    <div key={item.to}>
      <SidebarLink item={item} badge={badges?.[item.to]} onNavigate={onNavigate} />
      {subNav?.under === item.to && <SubNav items={subNav.items} onNavigate={onNavigate} />}
    </div>
  )
  return (
    <div className="flex h-full flex-col">
      <Link
        to="/control"
        onClick={onNavigate}
        className="flex h-14 shrink-0 items-center gap-2.5 px-5 sm:h-16"
      >
        <img src="/almond.svg" alt="" className="size-6 shrink-0" />
        <span className="font-heading text-base font-semibold tracking-tight">Almond Axol</span>
      </Link>

      <nav aria-label="Main" className="flex flex-1 flex-col gap-6 overflow-y-auto px-3 py-2">
        <div className="flex flex-col gap-0.5">{MAIN_NAV.map(renderItem)}</div>
        <div className="flex flex-col gap-0.5">
          <span className="px-3 pb-1 font-mono text-[0.65rem] tracking-widest text-white/30 uppercase">
            Tools
          </span>
          {TOOLS_NAV.map(renderItem)}
        </div>
      </nav>

      <div className="flex shrink-0 flex-col gap-0.5 border-t border-white/10 px-3 py-3">
        <a href="/vr" className={footerLink}>
          <Glasses className="size-4 shrink-0" />
          VR app
        </a>
        <button
          type="button"
          onClick={() => {
            onNavigate?.()
            onQuickstart()
          }}
          className={footerLink}
        >
          <Rocket className="size-4 shrink-0" />
          Quickstart
        </button>
        <a href="https://docs.almond.bot" target="_blank" rel="noreferrer" className={footerLink}>
          <BookOpen className="size-4 shrink-0" />
          Docs
        </a>
      </div>
    </div>
  )
}

/** The connection at a glance, at the top right of every page: the robot's
 *  computer and the selected device. Opens the connection details. */
function HostPill({ host }: { host: HostSummary }) {
  const body = (
    <>
      <span className={cn("size-2 shrink-0 rounded-full", STATUS_DOT_CLASS[host.dot])} />
      <span className="max-w-32 truncate sm:max-w-48">{host.label}</span>
      {host.detail && (
        <span className="hidden max-w-40 truncate text-white/45 md:inline">· {host.detail}</span>
      )}
    </>
  )
  const className =
    "flex h-8 min-w-0 items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 text-xs text-white/80"
  return host.onClick ? (
    <button
      type="button"
      onClick={host.onClick}
      title={host.version ? `axol v${host.version}` : undefined}
      className={cn(className, "transition-colors hover:border-white/25 hover:bg-white/[0.06]")}
    >
      {body}
    </button>
  ) : (
    <div className={className}>{body}</div>
  )
}

export function AppShell({
  title,
  back,
  description,
  actions,
  notices,
  badges,
  host,
  subNav,
  wide = false,
  children,
}: {
  title: string
  /** A page inside a section: the header leads back to the section's page. */
  back?: string
  description?: ReactNode
  /** Page-level controls on the right of the header. */
  actions?: ReactNode
  /** Alerts shown above every page of a section (updates, lockouts). */
  notices?: ReactNode
  badges?: NavBadges
  host?: HostSummary
  /** Sub-pages of one section, listed under its sidebar item. */
  subNav?: { under: string; items: SubNavItem[] }
  /** Wider content column, for chart-heavy pages. */
  wide?: boolean
  children: ReactNode
}) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  // Owned here, not by the sidebar: closing the drawer unmounts its contents,
  // which must not take an opening Quickstart dialog down with it.
  const [quickstartOpen, setQuickstartOpen] = useState(false)
  const location = useLocation()

  // A page change scrolls back to the top, like any app screen.
  useEffect(() => {
    window.scrollTo({ top: 0 })
  }, [location.pathname])

  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [drawerOpen])

  const sidebarProps = {
    badges,
    subNav,
    onQuickstart: () => setQuickstartOpen(true),
  }

  return (
    <div className="min-h-screen lg:pl-60">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 border-r border-white/10 bg-[#0e0e0f] pt-[env(safe-area-inset-top)] lg:block">
        <SidebarContents {...sidebarProps} />
      </aside>

      {/* Phone / tablet drawer */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setDrawerOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 w-72 max-w-[85vw] border-r border-white/10 bg-[#0e0e0f] pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)] shadow-2xl">
            <button
              type="button"
              onClick={() => setDrawerOpen(false)}
              aria-label="Close menu"
              className="absolute top-3 right-3 flex size-8 items-center justify-center rounded-md text-white/50 hover:bg-white/[0.06] hover:text-white"
            >
              <X className="size-4" />
            </button>
            <SidebarContents {...sidebarProps} onNavigate={() => setDrawerOpen(false)} />
          </aside>
        </div>
      )}

      <header className="sticky top-0 z-20 border-b border-white/10 bg-[#121212]/85 pt-[env(safe-area-inset-top)] backdrop-blur-md">
        <div
          className={cn(
            "safe-x mx-auto flex h-14 items-center gap-3 sm:h-16",
            wide ? "max-w-6xl" : "max-w-5xl"
          )}
        >
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open menu"
            aria-expanded={drawerOpen}
            className="-ml-1.5 flex size-9 shrink-0 items-center justify-center rounded-md text-white/60 hover:bg-white/[0.06] hover:text-white lg:hidden"
          >
            <Menu className="size-5" />
          </button>
          {back ? (
            <Link
              to={back}
              className="-ml-2 flex min-w-0 flex-1 items-center gap-1 rounded-md py-1 pr-2 pl-1 text-white/85 transition-colors hover:text-white"
            >
              <ChevronLeft className="size-5 shrink-0 text-white/50" />
              <h1 className="truncate font-heading text-lg font-semibold tracking-tight">
                {title}
              </h1>
            </Link>
          ) : (
            <h1 className="min-w-0 flex-1 truncate font-heading text-lg font-semibold tracking-tight">
              {title}
            </h1>
          )}
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
          {host && <HostPill host={host} />}
        </div>
      </header>

      <main
        className={cn(
          "safe-x mx-auto flex flex-col gap-6 py-6 pb-[max(2rem,env(safe-area-inset-bottom))]",
          wide ? "max-w-6xl" : "max-w-5xl"
        )}
      >
        {description && <p className="-mt-1 max-w-prose text-sm text-white/50">{description}</p>}
        {notices}
        {children}
      </main>

      <QuickstartDialog open={quickstartOpen} onClose={() => setQuickstartOpen(false)} />
    </div>
  )
}

/** A titled group of content on a page. */
export function PageSection({
  title,
  description,
  actions,
  children,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-x-3 gap-y-1">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h2 className="font-heading text-base font-semibold">{title}</h2>
          {description && <p className="text-xs text-white/45">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  )
}
