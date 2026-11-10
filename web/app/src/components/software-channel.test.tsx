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
        phase={null}
        opRunning={opRunning}
        hostInstalling={false}
        onSwitch={onSwitch}
      />
    )
  )
}

const option = (label: string) =>
  [...container.querySelectorAll<HTMLButtonElement>("[role=radio]")].find(
    (b) => b.textContent === label
  ) as HTMLButtonElement
const button = (label: string) =>
  [...container.querySelectorAll<HTMLButtonElement>("button")].find(
    (b) => b.textContent === label
  ) as HTMLButtonElement | undefined

describe("SoftwareChannel", () => {
  it("marks the installed channel", () => {
    render(status, () => {})
    expect(option("Release").getAttribute("aria-checked")).toBe("true")
    expect(option("Main").getAttribute("aria-checked")).toBe("false")
  })

  it("asks before switching, then switches", () => {
    const onSwitch = vi.fn()
    render(status, onSwitch)
    act(() => option("Main").click())
    expect(container.textContent).toContain("Switch to main")
    expect(onSwitch).not.toHaveBeenCalled()
    act(() => button("Switch to main")!.click())
    expect(onSwitch).toHaveBeenCalledWith("main")
    expect(button("Switch to main")).toBeUndefined()
  })

  it("does nothing when the installed channel is clicked", () => {
    render(status, () => {})
    act(() => option("Release").click())
    expect(button("Cancel")).toBeUndefined()
  })

  it("names the release it switches back to", () => {
    const onSwitch = vi.fn()
    render({ ...status, channel: "main", commit: "abcdef0123" }, onSwitch)
    act(() => option("Release").click())
    expect(container.textContent).toContain("Install the latest release (v0.2.41).")
    act(() => button("Switch to release")!.click())
    expect(onSwitch).toHaveBeenCalledWith("release")
  })

  it("can't switch to release before the latest release is known", () => {
    render({ ...status, channel: "main", remoteVersion: null }, () => {})
    act(() => option("Release").click())
    expect(container.textContent).toContain("The latest release is not known yet.")
    expect(button("Switch to release")!.disabled).toBe(true)
  })

  it("can't switch while an operation is running", () => {
    render(status, () => {}, { opRunning: true })
    act(() => option("Main").click())
    expect(container.textContent).toContain("An operation is running. Stop it first.")
    expect(button("Switch to main")!.disabled).toBe(true)
  })

  it("locks the choice while an update applies", () => {
    render(status, () => {}, { updating: true })
    expect(option("Main").disabled).toBe(true)
    expect(container.textContent).toContain("Updating…")
  })
})
