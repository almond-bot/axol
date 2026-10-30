import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { parsePrompt } from "@/lib/prompt"
import { ActiveRunPanel } from "./diagnostic-actions"
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

describe("parsePrompt", () => {
  it("defaults to Continue", () => {
    expect(parsePrompt("Hold the gripper.")).toEqual({
      text: "Hold the gripper.",
      button: "Continue",
    })
  })

  it("reads a named button", () => {
    expect(parsePrompt("Support both arms. [button: Power-cycle Axol]")).toEqual({
      text: "Support both arms.",
      button: "Power-cycle Axol",
    })
  })
})

describe("ActiveRunPanel", () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    container = document.createElement("div")
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(() => {
    act(() => root.unmount())
    container.remove()
  })

  it("labels the prompt's button", () => {
    act(() =>
      root.render(
        <ActiveRunPanel
          label="Guided zero"
          since={null}
          line={null}
          prompt="Support both arms. [button: Power-cycle Axol]"
          busy={false}
          onContinue={() => {}}
          onStop={() => {}}
        />
      )
    )
    expect(container.textContent).toContain("Support both arms.")
    expect(container.textContent).not.toContain("[button:")
    const buttons = [...container.querySelectorAll("button")].map((b) => b.textContent)
    expect(buttons).toContain("Power-cycle Axol ")
  })
})
