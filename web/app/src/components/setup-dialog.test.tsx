import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SetupDialog } from "./setup-dialog"

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

const pressEscape = (target: Element) =>
  act(() => {
    target.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))
  })

describe("SetupDialog host history", () => {
  it("closes the recent-hosts list on Escape before closing the dialog", () => {
    const onClose = vi.fn()
    act(() =>
      root.render(
        <SetupDialog
          open
          onClose={onClose}
          host=""
          hostHistory={["192.168.1.42", "robot.local"]}
          onChangeHost={() => {}}
          conn={{ state: "idle" }}
          onConnect={() => {}}
        />
      )
    )
    const input = container.querySelector<HTMLInputElement>("#setup-server-host")!
    act(() => input.focus())
    expect(container.querySelectorAll("li")).toHaveLength(2)

    pressEscape(input)
    expect(container.querySelectorAll("li")).toHaveLength(0)
    expect(onClose).not.toHaveBeenCalled()

    pressEscape(input)
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
