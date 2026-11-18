import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { BundleStatus, UpdateStatus } from "@/lib/supervisor"
import { InstallFromFile } from "./install-from-file"

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
  vi.restoreAllMocks()
})

const idleBundle: BundleStatus = {
  state: "idle",
  uploadId: null,
  name: null,
  size: 0,
  received: 0,
  version: null,
  error: null,
  errorCode: null,
}

const status = (bundle: Partial<BundleStatus> = {}): UpdateStatus => ({
  enabled: true,
  channel: "release",
  version: "0.2.46",
  remoteVersion: null,
  updateAvailable: false,
  idle: true,
  state: "idle",
  phase: null,
  error: null,
  bundle: { ...idleBundle, ...bundle },
})

function render(
  update: UpdateStatus,
  {
    onInstall = () => {},
    onStatus = () => {},
    opRunning = false,
  }: {
    onInstall?: () => void
    onStatus?: (u: UpdateStatus) => void
    opRunning?: boolean
  } = {}
) {
  act(() =>
    root.render(
      <InstallFromFile
        update={update}
        updating={false}
        opRunning={opRunning}
        hostInstalling={false}
        onStatus={onStatus}
        onInstall={onInstall}
        chunkBytes={4}
        retryDelayMs={0}
      />
    )
  )
}

const button = (label: string) =>
  [...container.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.textContent === label)

function openDialog() {
  act(() => button("Install from file")!.click())
}

describe("InstallFromFile", () => {
  it("asks for the bundle file", () => {
    render(status())
    openDialog()
    expect(container.textContent).toContain("axol-offline-v<version>-linux-aarch64.tar")
    expect(button("Choose file")).toBeDefined()
  })

  it("installs a checked bundle", () => {
    const onInstall = vi.fn()
    render(status({ state: "ready", version: "0.2.47" }), { onInstall })
    openDialog()
    expect(container.textContent).toContain("v0.2.46 → v0.2.47")
    act(() => button("Install update")!.click())
    expect(onInstall).toHaveBeenCalledOnce()
  })

  it("won't install while an operation runs", () => {
    render(status({ state: "ready", version: "0.2.47" }), { opRunning: true })
    openDialog()
    expect(button("Install update")!.disabled).toBe(true)
    expect(container.textContent).toContain("An operation is running. Stop it first.")
  })

  it("offers an older version only after saying so", () => {
    render(
      status({
        state: "error",
        error: "This file has Axol v0.2.40, which is older than the installed v0.2.46.",
        errorCode: "older",
      })
    )
    openDialog()
    expect(container.textContent).toContain("older than the installed v0.2.46")
    expect(button("Use older version")).toBeDefined()
  })

  it("uploads in chunks and resumes after a dropped connection", async () => {
    const puts: number[] = []
    let stored = 0
    let dropped = false
    const reply = (body: unknown, init?: ResponseInit) =>
      new Response(JSON.stringify(body), {
        headers: { "Content-Type": "application/json" },
        ...init,
      })
    const uploading = (received: number): BundleStatus => ({
      ...idleBundle,
      state: "uploading",
      uploadId: "u1",
      name: "axol-offline-v0.2.47-linux-aarch64.tar",
      size: 10,
      received,
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      const method = init?.method ?? "GET"
      if (method === "POST" && url.endsWith("/api/update/bundle")) return reply(uploading(stored))
      if (method === "PUT") {
        const offset = Number(new URL(url, "http://host").searchParams.get("offset"))
        puts.push(offset)
        if (offset === 4 && !dropped) {
          dropped = true
          throw new TypeError("Failed to fetch")
        }
        stored = offset + (init!.body as Blob).size
        return reply(uploading(stored))
      }
      if (url.endsWith("/check")) return reply({ ...uploading(10), state: "checking" })
      return reply(status({ state: "ready", version: "0.2.47" }))
    })
    const onStatus = vi.fn()
    render(status(), { onStatus })
    openDialog()
    const input = container.querySelector<HTMLInputElement>("input[type=file]")!
    const file = new File(["0123456789"], "axol-offline-v0.2.47-linux-aarch64.tar")
    Object.defineProperty(input, "files", { value: [file] })
    await act(async () => {
      input.dispatchEvent(new Event("change", { bubbles: true }))
    })
    await vi.waitFor(() => expect(onStatus).toHaveBeenCalled())
    expect(puts).toEqual([0, 4, 4, 8])
    expect(stored).toBe(10)
    expect(onStatus.mock.calls.at(-1)![0].bundle.state).toBe("ready")
  })
})
