import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { EpisodeControlSpec, PolicyState } from "@/lib/supervisor"
import { EpisodeControls } from "./operation-panel"

// Lets React's act() flush updates outside a test renderer.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  vi.useFakeTimers()
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.useRealTimers()
})

function policy(input: Partial<EpisodeControlSpec>): PolicyState {
  return {
    phase: "ready",
    episodesRecorded: 2,
    message: "Ready.",
    controls: [
      { command: "start", label: "Start recording" },
      { command: "task", label: "Set task", input: true, placeholder: "Episode 3 task", ...input },
    ],
  }
}

function render(state: PolicyState, onEpisode: (command: string) => void) {
  act(() => root.render(<EpisodeControls policy={state} onEpisode={onEpisode} hud={null} />))
}

function type(text: string) {
  const input = container.querySelector("input") as HTMLInputElement
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!
  act(() => {
    setter.call(input, text)
    input.dispatchEvent(new Event("input", { bubbles: true }))
  })
}

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((b) => b.textContent === label)
  if (!found) throw new Error(`no ${label} button`)
  return found
}

describe("episode input controls", () => {
  it("sends an autoSubmit input once typing pauses, with no button", () => {
    const sent: string[] = []
    render(policy({ autoSubmit: true }), (c) => sent.push(c))
    expect([...container.querySelectorAll("button")].map((b) => b.textContent)).toEqual([
      "Start recording",
    ])

    type("fold the")
    type("fold the towel")
    act(() => vi.advanceTimersByTime(399))
    expect(sent).toEqual([])
    act(() => vi.advanceTimersByTime(1))
    expect(sent).toEqual(["task fold the towel"])

    // Nothing is resent while the snapshot catches up, or once it has.
    act(() => vi.advanceTimersByTime(1000))
    render(policy({ autoSubmit: true, value: "fold the towel" }), (c) => sent.push(c))
    act(() => vi.advanceTimersByTime(1000))
    expect(sent).toEqual(["task fold the towel"])
    expect(container.textContent).toContain("Saved")
  })

  it("sends pending text before a button's command", () => {
    const sent: string[] = []
    render(policy({ autoSubmit: true }), (c) => sent.push(c))

    type("stack the cups")
    act(() => button("Start recording").click())
    expect(sent).toEqual(["task stack the cups", "start"])
    act(() => vi.advanceTimersByTime(1000))
    expect(sent).toEqual(["task stack the cups", "start"])
  })

  it("never sends an empty field (the op keeps its last value)", () => {
    const sent: string[] = []
    render(policy({ autoSubmit: true, value: "fold the towel" }), (c) => sent.push(c))

    type("")
    act(() => vi.advanceTimersByTime(1000))
    act(() => button("Start recording").click())
    expect(sent).toEqual(["start"])
  })

  it("keeps the submit button for a plain input", () => {
    const sent: string[] = []
    render(policy({}), (c) => sent.push(c))

    type("a note")
    act(() => vi.advanceTimersByTime(1000))
    expect(sent).toEqual([])
    act(() => button("Set task").click())
    expect(sent).toEqual(["task a note"])
  })
})
