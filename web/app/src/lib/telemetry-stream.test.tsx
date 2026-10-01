import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { setServerBase } from "./supervisor"
import { useTelemetryStream, type TelemetryStream } from "./telemetry"

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  url: string

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  close(): void {
    this.closed = true
    this.onclose?.()
  }

  send(message: object): void {
    this.onmessage?.({ data: JSON.stringify(message) })
  }
}

const history = {
  frames: [{ t: 1, m: { "left:ELBOW": [0.1, 0, 0] } }],
  slow: [
    { t: 1, m: { "left:ELBOW": { reachable: true, status: "ok", temperature: 30, voltage: 24 } } },
  ],
  timing: [{ t: 1, arms: { left: { sourceJoint: "ELBOW", targetHz: 100, deadlineMisses: 0 } } }],
}

const probe: { latest: TelemetryStream | null } = { latest: null }
let root: Root | null = null
let container: HTMLDivElement

function Probe({
  enabled,
  onRender,
}: {
  enabled: boolean
  onRender: (s: TelemetryStream) => void
}) {
  onRender(useTelemetryStream(enabled))
  return null
}

async function render(enabled: boolean) {
  await act(async () => {
    root!.render(
      <Probe
        enabled={enabled}
        onRender={(stream) => {
          probe.latest = stream
        }}
      />
    )
  })
}

const socket = () => FakeWebSocket.instances[FakeWebSocket.instances.length - 1]

describe("useTelemetryStream", () => {
  beforeEach(() => {
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.useFakeTimers()
    FakeWebSocket.instances = []
    probe.latest = null
    setServerBase("robot.local")
    vi.stubGlobal("WebSocket", FakeWebSocket)
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(history), { status: 200 }))
    )
    container = document.createElement("div")
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(async () => {
    await act(async () => root?.unmount())
    container.remove()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it("does nothing while disabled", async () => {
    await render(false)
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(probe.latest?.streaming).toBe(false)
    expect(probe.latest?.state).toBe("disconnected")
  })

  it("streams frames, backfills history, and tracks robot state", async () => {
    await render(true)
    expect(socket().url).toBe("wss://robot.local:8001/api/telemetry/ws")

    await act(async () => {
      socket().onopen?.()
    })
    expect(probe.latest?.streaming).toBe(true)
    expect(probe.latest?.frames.map((f) => f.t)).toEqual([1])
    expect(probe.latest?.slowFrames).toEqual([{ t: 1, m: { "left:ELBOW": [30, 24] } }])
    expect(probe.latest?.timingFrames.map((f) => f.t)).toEqual([1])
    expect(probe.latest?.timingLatest.left?.sourceJoint).toBe("ELBOW")

    const before = probe.latest!.version
    await act(async () => {
      socket().send({
        type: "hello",
        state: "connected",
        slow: { "left:ELBOW": { reachable: true } },
      })
      socket().send({ type: "frame", t: 2, m: { "left:ELBOW": [0.2, 0, 0] } })
      socket().send({
        type: "slow",
        t: 2,
        m: { "left:ELBOW": { reachable: true, status: "ok", temperature: 31, voltage: 23 } },
      })
      socket().send({ type: "timing", t: 2, arms: { right: { sourceJoint: "WRIST_1" } } })
      socket().send({ type: "state", state: "connecting" })
    })
    expect(probe.latest?.version).toBeGreaterThan(before)
    expect(probe.latest?.state).toBe("connecting")
    expect(probe.latest?.frames.map((f) => f.t)).toEqual([1, 2])
    expect(probe.latest?.slow["left:ELBOW"]?.temperature).toBe(31)
    expect(probe.latest?.slowFrames.map((f) => f.t)).toEqual([1, 2])
    expect(probe.latest?.timingLatest.right?.sourceJoint).toBe("WRIST_1")
    expect(probe.latest?.timingLatest.left?.sourceJoint).toBe("ELBOW")

    await act(async () => {
      socket().send({ type: "timing_reset" })
    })
    expect(probe.latest?.timingFrames).toEqual([])
    expect(probe.latest?.timingLatest).toEqual({})
  })

  it("keeps history older than streamed frames and reconnects after close", async () => {
    await render(true)
    const first = socket()
    // A frame arrives before the history response resolves.
    await act(async () => {
      first.send({ type: "frame", t: 5, m: {} })
      first.onopen?.()
    })
    expect(probe.latest?.frames.map((f) => f.t)).toEqual([1, 5])

    await act(async () => {
      first.onerror?.()
    })
    expect(first.closed).toBe(true)
    expect(probe.latest?.streaming).toBe(false)
    expect(FakeWebSocket.instances).toHaveLength(1)

    await act(async () => {
      vi.advanceTimersByTime(3000)
    })
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(socket()).not.toBe(first)
  })

  it("tolerates a failed history request", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 503 }))
    )
    await render(true)
    await act(async () => {
      socket().onopen?.()
    })
    expect(probe.latest?.streaming).toBe(true)
    expect(probe.latest?.frames).toEqual([])
  })
})
