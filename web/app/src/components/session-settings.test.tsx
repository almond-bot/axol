import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { SessionSettings } from "./session-settings"

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

const reach = {
  key: "position_multiplier",
  label: "Reach",
  type: "number",
  help: "Hand-to-arm scale",
  options: [],
  min: 0.5,
  max: 2,
  step: 0.1,
  unit: "x",
}

let root: Root
let container: HTMLDivElement

describe("SessionSettings", () => {
  beforeEach(() => {
    ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    container = document.createElement("div")
    document.body.appendChild(container)
    root = createRoot(container)
  })
  afterEach(() => {
    act(() => root.unmount())
    container.remove()
  })

  it("renders nothing without a socket or before the server announces settings", async () => {
    await act(async () => root.render(<SessionSettings socket={null} />))
    expect(container.innerHTML).toBe("")
    const ws = new FakeSocket()
    await act(async () => root.render(<SessionSettings socket={ws as unknown as WebSocket} />))
    expect(container.innerHTML).toBe("")
  })

  it("shows the announced settings and sends changes", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<SessionSettings socket={ws as unknown as WebSocket} />))
    await act(async () =>
      ws.push({
        type: "settings",
        value: { schema: [reach], values: { position_multiplier: 1.9 } },
      })
    )
    expect(container.textContent).toContain("Session settings")
    expect(container.textContent).toContain("Reach")
    expect(container.textContent).toContain("1.9 x")

    const up = container.querySelector<HTMLButtonElement>('[aria-label="Reach up"]')!
    await act(async () => up.click())
    expect(ws.sent.at(-1)).toEqual({ type: "set", key: "position_multiplier", value: 2 })

    // At the top of the range the up button is disabled.
    await act(async () =>
      ws.push({ type: "settings", value: { schema: [reach], values: { position_multiplier: 2 } } })
    )
    expect(container.querySelector<HTMLButtonElement>('[aria-label="Reach up"]')!.disabled).toBe(
      true
    )
  })
})
