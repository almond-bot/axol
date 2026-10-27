import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "./app-shell"

// Lets React's act() flush updates outside a test renderer.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
  window.scrollTo = vi.fn()
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

function render(path: string, props: Partial<Parameters<typeof AppShell>[0]> = {}) {
  act(() =>
    root.render(
      <MemoryRouter initialEntries={[path]}>
        <AppShell title="Page" {...props}>
          <p>Body</p>
        </AppShell>
      </MemoryRouter>
    )
  )
}

const desktopNav = () => container.querySelector("aside nav")!
const link = (label: string) =>
  [...desktopNav().querySelectorAll("a")].find((a) => a.textContent?.includes(label))!

describe("AppShell", () => {
  it("lists every section and marks the current one", () => {
    render("/control/settings/cameras")
    const labels = [...desktopNav().querySelectorAll("a")].map((a) => a.textContent)
    expect(labels).toEqual([
      "Overview",
      "Operations",
      "Datasets",
      "Request data",
      "Devices",
      "Settings",
      "Diagnostics",
      "Logs",
    ])
    expect(link("Settings").getAttribute("aria-current")).toBe("page")
    // Overview matches only /control itself, not every page under it.
    expect(link("Overview").getAttribute("aria-current")).toBeNull()
  })

  it("shows a running operation next to its nav item", () => {
    render("/control", { badges: { "/control/operations": "running" } })
    expect(link("Operations").querySelector('[title="Running"]')).not.toBeNull()
    expect(link("Overview").querySelector('[title="Running"]')).toBeNull()
  })

  it("shows the connection in the header, and opens its details", () => {
    const onClick = vi.fn()
    render("/control", {
      host: { dot: "ok", label: "robot-1", detail: "Axol ready", version: "1.2.0", onClick },
    })
    const pill = [...container.querySelectorAll("header button")].find((b) =>
      b.textContent?.includes("robot-1")
    )!
    expect(pill.textContent).toContain("Axol ready")
    expect(pill.getAttribute("title")).toBe("axol v1.2.0")
    act(() => (pill as HTMLButtonElement).click())
    expect(onClick).toHaveBeenCalledTimes(1)
    expect(container.querySelector("aside")?.textContent).not.toContain("robot-1")
  })

  it("lists a section's sub-pages under it", () => {
    render("/diagnostics/tuning", {
      subNav: {
        under: "/diagnostics",
        items: [
          { to: "/diagnostics", label: "Live" },
          { to: "/diagnostics/tuning", label: "Tuning" },
        ],
      },
    })
    expect(link("Tuning").getAttribute("aria-current")).toBe("page")
    expect(link("Live").getAttribute("aria-current")).toBeNull()
  })
})
