import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { SessionSettings } from "./session-settings"
import { TaskPicker } from "./task-picker"

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

const taskDef = {
  key: "task",
  label: "Task",
  type: "select",
  help: "The task the next episode records.",
  options: ["Pick up the cup", "Open the drawer"],
  min: null,
  max: null,
  step: null,
  unit: "",
}

const haptics = {
  key: "haptics",
  label: "Haptics",
  type: "boolean",
  help: "Vibrates a controller.",
  options: [],
  min: null,
  max: null,
  step: null,
  unit: "",
}

const announce = (ws: FakeSocket, current = "Pick up the cup") =>
  ws.push({
    type: "settings",
    value: { schema: [haptics, taskDef], values: { haptics: true, task: current } },
  })

// React tracks an input's value itself, so a test sets it through the
// native setter and fires an input event, as a real keystroke would.
function typeInto(input: HTMLInputElement, text: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!
  setter.call(input, text)
  input.dispatchEvent(new Event("input", { bubbles: true }))
}

let root: Root
let container: HTMLDivElement

describe("TaskPicker", () => {
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

  it("renders nothing until the session announces a task picker", async () => {
    const ws = new FakeSocket()
    await act(async () =>
      root.render(<TaskPicker socket={ws as unknown as WebSocket} locked={false} />)
    )
    expect(container.innerHTML).toBe("")
    await act(async () =>
      ws.push({ type: "settings", value: { schema: [haptics], values: { haptics: true } } })
    )
    expect(container.innerHTML).toBe("")
  })

  it("selects a listed task", async () => {
    const ws = new FakeSocket()
    await act(async () =>
      root.render(<TaskPicker socket={ws as unknown as WebSocket} locked={false} />)
    )
    await act(async () => announce(ws))
    const select = container.querySelector("select")!
    expect(select.value).toBe("Pick up the cup")
    expect([...select.options].map((o) => o.value)).toEqual(["Pick up the cup", "Open the drawer"])

    await act(async () => {
      select.value = "Open the drawer"
      select.dispatchEvent(new Event("change", { bubbles: true }))
    })
    expect(ws.sent.at(-1)).toEqual({ type: "set", key: "task", value: "Open the drawer" })
  })

  it("adds a new task", async () => {
    const ws = new FakeSocket()
    await act(async () =>
      root.render(<TaskPicker socket={ws as unknown as WebSocket} locked={false} />)
    )
    await act(async () => announce(ws))
    const input = container.querySelector<HTMLInputElement>('[aria-label="New task"]')!
    const add = [...container.querySelectorAll("button")].find((b) => b.textContent === "Add task")!
    expect(add.disabled).toBe(true)

    await act(async () => typeInto(input, "  Wipe the table "))
    expect(add.disabled).toBe(false)
    await act(async () => add.click())
    expect(ws.sent.at(-1)).toEqual({ type: "set", key: "task", value: "Wipe the table" })
    expect(input.value).toBe("")
  })

  it("is disabled while an episode records", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<TaskPicker socket={ws as unknown as WebSocket} locked />))
    await act(async () => announce(ws))
    expect(container.querySelector("select")!.disabled).toBe(true)
    expect(container.querySelector<HTMLInputElement>('[aria-label="New task"]')!.disabled).toBe(
      true
    )
    expect(container.textContent).toContain("You can change the task when this episode ends.")
  })

  it("is left out of the session settings card", async () => {
    const ws = new FakeSocket()
    await act(async () => root.render(<SessionSettings socket={ws as unknown as WebSocket} />))
    await act(async () => announce(ws))
    expect(container.textContent).toContain("Haptics")
    expect(container.textContent).not.toContain("Open the drawer")
  })
})
