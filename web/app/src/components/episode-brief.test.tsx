import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { EpisodeBrief, PolicyState } from "@/lib/supervisor"
import { EpisodeControls } from "./operation-panel"

vi.mock("@/components/camera-feeds", () => ({ CameraFeeds: () => null }))
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

const BRIEF: EpisodeBrief = {
  eyebrow: "Section 2 · Put away · 3/30",
  title: "Scene 14",
  headline: "Pick up the eraser and set it flat in the blue box on the left",
  items: [
    { label: "Arm", value: "Right", emphasis: true },
    { label: "Containers", value: ["Blue box L-Mid, 90°", "Clear bin C-Mid, 45° CCW"] },
    { label: "Distractors", value: "None" },
  ],
  note: "Record A, reset every object to A's start, then record B.",
  progress: { done: 12, total: 240, label: "12 of 240 scenes recorded" },
  grid: {
    cols: [
      { key: "L", label: "Left" },
      { key: "C", label: "Center" },
      { key: "R", label: "Right" },
    ],
    rows: [
      { key: "Far", label: "Far" },
      { key: "Mid", label: "Mid" },
      { key: "Near", label: "Near" },
    ],
    items: [
      { col: "L", row: "Mid", label: "Blue box", detail: "90°", role: "target", rotation: 90 },
      { col: "C", row: "Mid", label: "Clear bin", role: "container", rotation: 45 },
      { col: "R", row: "Near", label: "Eraser", detail: "felt side down", role: "object" },
    ],
    footer: "Robot",
  },
}

function state(phase: string, brief: EpisodeBrief | undefined = BRIEF): PolicyState {
  return {
    phase,
    episodesRecorded: 2,
    message: "Set up the scene.",
    controls: [{ command: "start", label: "Start recording" }],
    brief,
  }
}

function render(s: PolicyState) {
  act(() => root.render(<EpisodeControls policy={s} onEpisode={() => {}} hud={null} />))
}

function card() {
  return container.querySelector('[data-testid="episode-brief"]')
}

function toggle() {
  const b = [...container.querySelectorAll("button")].find((x) => /setup/.test(x.textContent ?? ""))
  if (!b) throw new Error("no setup toggle")
  act(() => b.click())
}

describe("EpisodeBriefCard", () => {
  it("renders nothing without a brief", () => {
    render({ ...state("ready"), brief: undefined })
    expect(card()).toBeNull()
  })

  it("shows the instruction, the setup list and the layout grid", () => {
    render(state("ready"))
    const text = card()!.textContent ?? ""
    expect(text).toContain("Scene 14")
    expect(text).toContain("Section 2 · Put away · 3/30")
    expect(text).toContain("Pick up the eraser")
    expect(text).toContain("12 of 240 scenes recorded")
    expect(text).toContain("Clear bin C-Mid, 45° CCW")
    expect(text).toContain("Record A, reset")
    const cell = container.querySelector('[data-cell="L-Mid"]')!
    expect(cell.textContent).toContain("Blue box")
    expect(cell.querySelector('[data-role="target"]')).not.toBeNull()
    expect(container.querySelector('[data-cell="R-Near"]')!.textContent).toContain("Eraser")
    expect(container.querySelector('[data-cell="C-Far"]')!.textContent).toBe("")
    expect(text).toContain("Robot")
    expect(text).toContain("Destination")
  })

  it("folds the setup away, and starts folded while recording", () => {
    render(state("ready"))
    toggle()
    expect(container.querySelector('[data-cell="L-Mid"]')).toBeNull()
    // The instruction stays visible when folded.
    expect(card()!.textContent).toContain("Pick up the eraser")

    render(state("recording"))
    expect(container.querySelector('[data-cell="L-Mid"]')).toBeNull()
    toggle()
    expect(container.querySelector('[data-cell="L-Mid"]')).not.toBeNull()

    // A new phase goes back to its default (open between takes).
    render(state("ready"))
    expect(container.querySelector('[data-cell="L-Mid"]')).not.toBeNull()
  })
})
