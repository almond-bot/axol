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
    ["provisioning", "Installing deps…"],
    ["restarting", "Restarting…"],
    ["rebooting", "Rebooting host…"],
  ] as const)("labels the %s phase while updating", (phase, label) => {
    renderBanner(true, phase)
    expect(button().textContent).toBe(label)
    expect(button().disabled).toBe(true)
  })

  it("falls back to a generic label before the first phase arrives", () => {
    renderBanner(true, null)
    expect(button().textContent).toBe("Updating…")
  })
})
