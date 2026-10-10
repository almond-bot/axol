import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { setServerBase } from "./supervisor"
import { useVrSocket } from "./vr-socket"

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  closed = false
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  close() {
    this.closed = true
    this.onclose?.()
  }
}

let root: Root
const probe: { latest: WebSocket | null } = { latest: null }

function Probe({
  enabled,
  onRender,
}: {
  enabled: boolean
  onRender: (ws: WebSocket | null) => void
}) {
  onRender(useVrSocket("robot.local", 8000, enabled))
  return null
}

const render = (enabled: boolean) =>
  act(async () =>
    root.render(
      <Probe
        enabled={enabled}
        onRender={(ws) => {
          probe.latest = ws
        }}
      />
    )
  )

describe("useVrSocket", () => {
  beforeEach(() => {
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.useFakeTimers()
    FakeWebSocket.instances = []
    probe.latest = null
    setServerBase("robot.local")
    vi.stubGlobal("WebSocket", FakeWebSocket)
    root = createRoot(document.createElement("div"))
  })
  afterEach(() => {
    act(() => root.unmount())
    vi.unstubAllGlobals()
    vi.useRealTimers()
    setServerBase("")
  })

  it("stays idle while disabled", async () => {
    await render(false)
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it("connects to the VR server, reconnects after a drop, closes on disable", async () => {
    await render(true)
    const first = FakeWebSocket.instances[0]
    expect(first.url).toBe("wss://robot.local:8000/ws")
    expect(probe.latest).toBeNull()
    await act(async () => first.onopen?.())
    expect(probe.latest).toBe(first)

    await act(async () => first.onclose?.())
    expect(probe.latest).toBeNull()
    await act(async () => vi.advanceTimersByTime(3000))
    const second = FakeWebSocket.instances[1]
    expect(second).toBeDefined()
    await act(async () => second.onopen?.())

    await render(false)
    expect(second.closed).toBe(true)
    expect(probe.latest).toBeNull()
  })
})
