import { act, useRef } from "react"
import { createRoot, type Root } from "react-dom/client"
import type { RefObject } from "react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { useAxolJoints, type AxolJointSample } from "./useAxolJoints"

class FakeSocket extends EventTarget {
  push(msg: object) {
    this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(msg) }))
  }
}

let root: Root
let ref: RefObject<AxolJointSample | null> | null = null

function Probe({ ws }: { ws: FakeSocket }) {
  const wsRef = useRef(ws as unknown as WebSocket)
  ref = useAxolJoints(wsRef, true)
  return null
}

describe("useAxolJoints", () => {
  beforeEach(() => {
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    root = createRoot(document.createElement("div"))
    ref = null
  })
  afterEach(() => act(() => root.unmount()))

  it("keeps the latest joints push with its pair status", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<Probe ws={ws} />))
    expect(ref!.current).toBeNull()

    ws.push({
      type: "joints",
      value: {
        q: { left_s1_0: 0.1 },
        l_grip: 0.5,
        engaged: true,
        pair: { aligned: true, width: 0.3, grasp: "flush", squeeze: 6 },
      },
    })
    const s = ref!.current!
    expect(s.q).toEqual({ left_s1_0: 0.1 })
    expect(s.l_grip).toBe(0.5)
    expect(s.r_grip).toBe(1)
    expect(s.engaged).toBe(true)
    expect(s.pair).toEqual({
      aligned: true,
      width: 0.3,
      grasp: "flush",
      elbow: null,
      squeeze: 6,
      trim: null,
    })
  })

  it("drops a pair without a width and ignores pushes without joints", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<Probe ws={ws} />))
    ws.push({ type: "joints", value: { q: {}, pair: { aligned: true } } })
    expect(ref!.current!.pair).toBeNull()
    ws.push({ type: "joints", value: { l_grip: 0 } })
    expect(ref!.current!.l_grip).toBe(1)
  })
})
