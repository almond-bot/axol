import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { UpdateChannel, UpdateStatus } from "@/lib/supervisor"
import { SoftwareChannel } from "./software-channel"

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
  channel: "release",
  version: "0.2.41",
  commit: null,
  remoteVersion: "0.2.41",
  remoteCommit: "abcdef0123",
  updateAvailable: false,
  idle: true,
  state: "idle",
  phase: null,
  error: null,
}

function render(
  update: UpdateStatus,
  onSwitch: (channel: UpdateChannel) => void,
  { opRunning = false, updating = false } = {}
) {
  act(() =>
    root.render(
      <SoftwareChannel
        update={update}
        updating={updating}
        opRunning={opRunning}
        hostInstalling={false}
        onSwitch={onSwitch}
      />
    )
  )
}

const buttons = (label: string) =>
  [...container.querySelectorAll<HTMLButtonElement>("button")].filter(
    (b) => b.textContent === label
  )
/** The button in the open confirmation (the last match), if any. */
const confirm = (label: string) => buttons(label).at(-1)

describe("SoftwareChannel", () => {
  it("offers the other channel", () => {
    render(status, () => {})
    expect(buttons("Switch to main")).toHaveLength(1)
    render({ ...status, channel: "main" }, () => {})
    expect(buttons("Switch to release")).toHaveLength(1)
  })

  it("shows what the switch installs, asks, then switches", () => {
    const onSwitch = vi.fn()
    render({ ...status, version: "0.2.44", remoteCommit: "2222222bbbb" }, onSwitch)
    act(() => buttons("Switch to main")[0].click())
    expect(container.textContent).toContain("v0.2.44 \u2192 main 2222222")
    const link = container.querySelector("a") as HTMLAnchorElement
    expect(link.textContent).toBe("View changes")
    expect(link.href).toBe("https://github.com/almond-bot/axol/compare/v0.2.44...main")
    expect(onSwitch).not.toHaveBeenCalled()
    act(() => confirm("Switch to main")!.click())
    expect(onSwitch).toHaveBeenCalledWith("main")
    expect(buttons("Cancel")).toHaveLength(0)
  })

  it("names the release it switches back to", () => {
    const onSwitch = vi.fn()
    render({ ...status, channel: "main", commit: "1111111aaaa" }, onSwitch)
    act(() => buttons("Switch to release")[0].click())
    expect(container.textContent).toContain("main 1111111 \u2192 v0.2.41")
    expect((container.querySelector("a") as HTMLAnchorElement).textContent).toBe("Release notes")
    act(() => confirm("Switch to release")!.click())
    expect(onSwitch).toHaveBeenCalledWith("release")
  })

  it("can't switch to release before the latest release is known", () => {
    render({ ...status, channel: "main", remoteVersion: null }, () => {})
    act(() => buttons("Switch to release")[0].click())
    expect(container.textContent).toContain("The latest release is not known yet.")
    expect(confirm("Switch to release")!.disabled).toBe(true)
  })

  it("can't switch while an operation is running", () => {
    render(status, () => {}, { opRunning: true })
    act(() => buttons("Switch to main")[0].click())
    expect(container.textContent).toContain("An operation is running. Stop it first.")
    expect(confirm("Switch to main")!.disabled).toBe(true)
  })

  it("can't switch while an update applies", () => {
    render(status, () => {}, { updating: true })
    expect(buttons("Switch to main")[0].disabled).toBe(true)
  })
})
