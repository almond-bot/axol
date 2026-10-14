import { act, useRef } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import type { AxolSettingDef } from "./types"
import { formatSettingValue, nextSettingValue, useAxolSettings } from "./useAxolSettings"

const num = (over: Partial<AxolSettingDef> = {}): AxolSettingDef => ({
  key: "position_multiplier",
  label: "Reach",
  type: "number",
  help: "",
  options: [],
  min: 0.5,
  max: 2,
  step: 0.1,
  unit: "x",
  ...over,
})
const bool: AxolSettingDef = { ...num(), key: "box_mode", type: "boolean", unit: "" }
const select: AxolSettingDef = {
  ...num(),
  key: "reengage",
  type: "select",
  options: ["clutch", "ramp"],
  unit: "",
}

describe("nextSettingValue", () => {
  it("toggles booleans and cycles selects", () => {
    expect(nextSettingValue(bool, true, 1)).toBe(false)
    expect(nextSettingValue(select, "clutch", 1)).toBe("ramp")
    expect(nextSettingValue(select, "clutch", -1)).toBe("ramp")
    expect(nextSettingValue(select, "ramp", 1)).toBe("clutch")
  })

  it("steps numbers on the grid and stops at the range ends", () => {
    expect(nextSettingValue(num(), 1, 1)).toBe(1.1)
    expect(nextSettingValue(num(), 0.7 + 0.1 + 0.1, 1)).toBe(1)
    expect(nextSettingValue(num(), 1.95, 1)).toBe(2)
    expect(nextSettingValue(num(), 2, 1)).toBeUndefined()
    expect(nextSettingValue(num(), 0.5, -1)).toBeUndefined()
  })
})

describe("formatSettingValue", () => {
  it("formats each type", () => {
    expect(formatSettingValue(bool, true)).toBe("ON")
    expect(formatSettingValue(select, "ramp")).toBe("ramp")
    expect(formatSettingValue(num(), 1.2)).toBe("1.2 x")
    expect(formatSettingValue(num({ unit: "" }), 1.2)).toBe("1.2")
    expect(formatSettingValue(num(), undefined)).toBe("—")
  })
})

class FakeSocket extends EventTarget {
  readyState = WebSocket.OPEN
  sent: unknown[] = []
  send(text: string) {
    this.sent.push(JSON.parse(text))
  }
  push(msg: object) {
    this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(msg) }))
  }
}

type Hook = ReturnType<typeof useAxolSettings>
let root: Root
let container: HTMLDivElement
let latest: Hook | null = null

function Probe({ ws }: { ws: FakeSocket }) {
  const wsRef = useRef(ws as unknown as WebSocket)
  latest = useAxolSettings(wsRef, true)
  return null
}

describe("useAxolSettings", () => {
  beforeEach(() => {
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    container = document.createElement("div")
    root = createRoot(container)
    latest = null
  })
  afterEach(() => act(() => root.unmount()))

  it("requests, mirrors and changes the settings", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<Probe ws={ws} />))
    expect(ws.sent).toEqual([{ type: "session-config-request" }])
    expect(latest!.settings).toBeNull()

    await act(async () =>
      ws.push({ type: "settings", value: { schema: [num()], values: { position_multiplier: 1 } } })
    )
    expect(latest!.settings?.values.position_multiplier).toBe(1)

    await act(async () => latest!.step(num(), 1))
    expect(ws.sent.at(-1)).toEqual({ type: "set", key: "position_multiplier", value: 1.1 })
    await act(async () => latest!.setSetting("box_mode", "toggle"))
    expect(ws.sent.at(-1)).toEqual({ type: "set", key: "box_mode", value: "toggle" })
  })

  it("ignores malformed and unrelated messages", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<Probe ws={ws} />))
    await act(async () => {
      ws.push({ type: "settings", value: { schema: "nope" } })
      ws.push({ type: "joints", value: {} })
      ws.dispatchEvent(new MessageEvent("message", { data: "not json" }))
    })
    expect(latest!.settings).toBeNull()
  })
})
