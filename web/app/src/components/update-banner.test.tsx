import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import type { UpdatePhase, UpdateStatus } from "@/lib/supervisor"
import { UpdateBanner } from "./update-banner"

// Lets React's act() flush updates outside a test renderer.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const status: UpdateStatus = {
  enabled: true,
  version: "0.2.10",
  remoteVersion: "0.2.11",
  updateAvailable: true,
  idle: true,
  state: "updating",
  phase: null,
  error: null,
}

const renderBanner = (updating: boolean, phase: UpdatePhase | null) =>
  act(() =>
    root.render(
      <UpdateBanner
        update={status}
        updating={updating}
        phase={phase}
        blocked={false}
        onUpdate={() => {}}
      />
    )
  )

const button = () => container.querySelector("button") as HTMLButtonElement

describe("UpdateBanner", () => {
  it("shows the version jump and an enabled Update button when idle", () => {
    renderBanner(false, null)
    expect(container.textContent).toContain("v0.2.10")
    expect(container.textContent).toContain("v0.2.11")
    expect(button().textContent).toBe("Update")
    expect(button().disabled).toBe(false)
  })

  it.each([
    ["upgrading", "Upgrading…"],
    ["provisioning", "Installing packages…"],
    ["restarting", "Restarting…"],
    ["rebooting", "Rebooting host…"],
  ] as const)("labels the %s phase while updating", (phase, label) => {
    renderBanner(true, phase)
    expect(button().textContent).toBe(label)
    expect(button().disabled).toBe(true)
  })

  it("shows an install from a file without a release link", () => {
    act(() =>
      root.render(
        <UpdateBanner
          update={{ ...status, updateAvailable: false, remoteVersion: null }}
          fromFileVersion="0.2.47"
          updating
          phase="provisioning"
          blocked={false}
          onUpdate={() => {}}
        />
      )
    )
    expect(container.textContent).toContain("Installing an update from a file")
    expect(container.textContent).toContain("v0.2.10 → v0.2.47")
    expect(container.querySelector("a")).toBeNull()
    expect(button().textContent).toBe("Installing packages…")
  })

  it("falls back to a generic label before the first phase arrives", () => {
    renderBanner(true, null)
    expect(button().textContent).toBe("Updating…")
  })

  it("shows the commit jump and a compare link on main", () => {
    act(() =>
      root.render(
        <UpdateBanner
          update={{
            ...status,
            state: "idle",
            channel: "main",
            commit: "1111111aaaa",
            remoteCommit: "2222222bbbb",
          }}
          updating={false}
          phase={null}
          blocked={false}
          onUpdate={() => {}}
        />
      )
    )
    expect(container.textContent).toContain("New changes are available on main")
    expect(container.textContent).toContain("main 1111111 \u2192 2222222")
    const link = container.querySelector("a") as HTMLAnchorElement
    expect(link.textContent).toBe("View changes")
    expect(link.href).toBe("https://github.com/almond-bot/axol/compare/1111111aaaa...2222222bbbb")
  })

  it("shows a switch to main while it applies", () => {
    act(() =>
      root.render(
        <UpdateBanner
          update={{
            ...status,
            version: "0.2.44",
            updateAvailable: false,
            channel: "release",
            remoteCommit: "2222222bbbb",
          }}
          switchTo="main"
          updating
          phase="provisioning"
          blocked={false}
          onUpdate={() => {}}
        />
      )
    )
    expect(container.textContent).toContain("Switching to main")
    expect(container.textContent).toContain("v0.2.44 \u2192 main 2222222")
    expect(button().textContent).toBe("Installing packages…")
  })

  it("shows a switch back to the latest release while it applies", () => {
    act(() =>
      root.render(
        <UpdateBanner
          update={{ ...status, updateAvailable: false, channel: "main", commit: "1111111aaaa" }}
          switchTo="release"
          updating
          phase={null}
          blocked={false}
          onUpdate={() => {}}
        />
      )
    )
    expect(container.textContent).toContain("Switching to the latest release")
    expect(container.textContent).toContain("main 1111111 \u2192 v0.2.11")
    expect((container.querySelector("a") as HTMLAnchorElement).textContent).toBe("Release notes")
  })
})
