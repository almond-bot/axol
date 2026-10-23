import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { EpisodeControlSpec, PolicyState } from "@/lib/supervisor"
import { EpisodeControls } from "./operation-panel"

// operation-panel's camera feeds import the workspace
// @almond/axol-vr-client package, which CI runs these tests before building;
// the episode box doesn't use them.
vi.mock("@/components/camera-feeds", () => ({ CameraFeeds: () => null }))

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

describe("episode rejection notice", () => {
  it("keeps a rejected take's reason on screen after returning to ready", () => {
    const reason = "Episode 3 was REJECTED by the quality gate and discarded: 9.0% stale pose."
    render({ ...policy({}), notice: { level: "error", message: reason } }, vi.fn())

    const alert = container.querySelector('[role="alert"]')
    expect(alert?.textContent).toBe(reason)
    expect(alert?.className).toContain("text-red-200")
    // The ordinary ready status still shows alongside it.
    expect(container.textContent).toContain("Ready.")
  })

  it("shows no alert without a notice", () => {
    render(policy({}), vi.fn())
    expect(container.querySelector('[role="alert"]')).toBeNull()
  })
})

describe("Mantis tracking quality", () => {
  const clean: NonNullable<PolicyState["trackingQuality"]> = {
    status: "ready",
    reasons: [],
    faultCount: 0,
    lastFault: null,
    retakeRequired: false,
  }
  const recording: PolicyState = {
    phase: "recording",
    episode: 3,
    episodesRecorded: 2,
    controls: [
      { command: "s", label: "Save episode", confirm: true },
      { command: "r", label: "Discard and retake" },
    ],
  }

  it("keeps a rejected save visible beside recovered live tracking and permits a retake", () => {
    const onEpisode = vi.fn()
    const reason = "10.0% of control samples used a pose older than 250 ms (limit 5%)"
    render(
      {
        ...policy({}),
        episode: 14,
        episodesRecorded: 0,
        trackingQuality: clean,
        notice: { level: "error", message: reason },
      },
      onEpisode
    )
    const alert = container.querySelector('[role="alert"]')
    expect(alert?.textContent).toContain(reason)
    expect(container.textContent).toContain("Live tracking checks passed")
    expect(container.textContent).toContain("0 saved")
    expect(button("Start recording").disabled).toBe(false)
    act(() => button("Start recording").click())
    expect(onEpisode).toHaveBeenCalledWith("start")

    render({ ...recording, trackingQuality: clean }, onEpisode)
    expect(container.querySelector('[role="alert"]')).toBeNull()
    expect(container.textContent).not.toContain(reason)
  })

  it("blocks starting while quality is checking or failed, but permits a redundancy warning", () => {
    const onEpisode = vi.fn()
    for (const status of ["checking", "error"] as const) {
      render({ ...policy({}), trackingQuality: { ...clean, status } }, onEpisode)
      expect(button("Start recording").disabled).toBe(true)
      act(() => button("Start recording").click())
    }
    expect(onEpisode).not.toHaveBeenCalled()
    render({ ...policy({}), trackingQuality: { ...clean, status: "warning" } }, onEpisode)
    expect(button("Start recording").disabled).toBe(false)
    act(() => button("Start recording").click())
    expect(onEpisode).toHaveBeenCalledWith("start")
  })

  it("allows cancelling a countdown after tracking fails", () => {
    const onEpisode = vi.fn()
    render(
      {
        ...policy({}),
        phase: "countdown",
        controls: [{ command: "start", label: "Cancel countdown" }],
        trackingQuality: { ...clean, status: "error" },
      },
      onEpisode
    )
    expect(button("Cancel countdown").disabled).toBe(false)
    act(() => button("Cancel countdown").click())
    expect(onEpisode).toHaveBeenCalledWith("start")
  })

  it("explains recovered tracking with gripper control still disengaged", () => {
    const onEpisode = vi.fn()
    const recovered = {
      ...clean,
      controlState: "awaiting_engagement" as const,
      reengageRequired: true,
    }
    render({ ...policy({}), trackingQuality: recovered }, onEpisode)
    expect(container.textContent).toContain("Gripper control paused")
    expect(container.textContent).toContain("gripper control remains disengaged")
    expect(container.textContent).toContain("Squeeze both triggers together, then release both")
    expect(button("Start recording").disabled).toBe(false)
    render({ ...recording, trackingQuality: { ...recovered, retakeRequired: true } }, onEpisode)
    expect(container.textContent).toContain("Discard this take. When Mantis is ready")
    expect(button("Save episode").disabled).toBe(true)

    render({ ...policy({}), trackingQuality: { ...clean, controlState: "engaged" } }, onEpisode)
    expect(container.textContent).toContain("Live tracking checks passed")
    expect(container.textContent).not.toContain("gripper control remains disengaged")
  })

  it("shows a missing trigger input even when optical tracking is healthy", () => {
    render(
      {
        ...recording,
        trackingQuality: {
          ...clean,
          controlState: "recovering",
          triggersLive: { left: false, right: true },
        },
      },
      vi.fn()
    )
    expect(container.textContent).toContain("Gripper control paused")
    expect(container.textContent).toContain("Left trigger heartbeat is missing")
    expect(container.textContent).not.toContain("Right trigger heartbeat is missing")
  })

  it("distinguishes a stopped runtime, persistent setup faults and recoverable visibility", () => {
    const onEpisode = vi.fn()
    render(
      {
        ...policy({}),
        trackingQuality: { ...clean, status: "error", nativeFailure: true },
      },
      onEpisode
    )
    expect(container.textContent).toContain("The tracking process stopped")
    expect(container.textContent).not.toContain("Restore both trackers to view")

    render(
      {
        ...policy({}),
        trackingQuality: { ...clean, status: "error", calibrationRequired: true },
      },
      onEpisode
    )
    expect(container.textContent).toContain("Stop collection")
    expect(container.textContent).toContain("Recalibrate Lighthouse")
    expect(container.textContent).not.toContain("Validate tracking")
    expect(container.textContent).toContain("Waiting for visibility alone cannot clear this fault")
    expect(container.textContent).not.toContain("The tracking process stopped")

    render({ ...policy({}), trackingQuality: { ...clean, status: "error" } }, onEpisode)
    expect(container.textContent).toContain("Restore both trackers to view")
    expect(container.textContent).not.toContain("repair Lighthouse support")
  })

  it("keeps a failed take red and unsavable after tracking recovers; discard remains available", () => {
    const onEpisode = vi.fn()
    render({ ...recording, trackingQuality: clean }, onEpisode)
    act(() => button("Save episode").click())
    const fault = {
      ...clean,
      status: "error" as const,
      reasons: ["Left tracker position jumped"],
      faultCount: 1,
      lastFault: "Left tracker position jumped",
      retakeRequired: true,
    }
    render({ ...recording, trackingQuality: fault }, onEpisode)
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("retake required")
    expect(container.textContent).toContain(
      "1 tracking fault events since tracking started (includes setup and idle time)."
    )
    expect(button("Confirm: Save episode?").disabled).toBe(true)
    act(() => button("Confirm: Save episode?").click())
    expect(onEpisode).not.toHaveBeenCalled()

    render({ ...recording, trackingQuality: { ...fault, status: "ready", reasons: [] } }, onEpisode)
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "Left tracker position jumped"
    )
    expect(button("Confirm: Save episode?").disabled).toBe(true)
    act(() => button("Discard and retake").click())
    expect(onEpisode).toHaveBeenCalledWith("r")

    render(
      {
        ...policy({}),
        episode: 3,
        trackingQuality: { ...clean, lastEpisodeFault: fault.lastFault },
      },
      onEpisode
    )
    expect(container.querySelector('[role="alert"]')).toBeNull()
    expect(button("Start recording").disabled).toBe(false)
    expect(container.textContent).toContain(
      "Previous take needs a retake: Left tracker position jumped"
    )
    render({ ...recording, trackingQuality: clean }, onEpisode)
    expect(container.textContent).not.toContain("Previous take needs a retake")
  })

  it("shows automatic fault capture progress and the saved path without clearing a failed take", () => {
    const onEpisode = vi.fn()
    render(
      {
        ...recording,
        trackingQuality: { ...clean, diagnostics: { enabled: true, state: "armed" } },
      },
      onEpisode
    )
    expect(container.textContent).toContain("Automatic fault capture is ready")
    expect(button("Save episode").disabled).toBe(false)

    const fault = { ...clean, retakeRequired: true, lastFault: "Left tracker pose jumped" }
    for (const [state, message] of [
      ["capturing", "Keep this session open for 3 seconds"],
      ["saving", "Saving tracking diagnostics. Keep this session open"],
    ] as const) {
      render(
        {
          ...recording,
          trackingQuality: { ...fault, diagnostics: { enabled: true, state } },
        },
        onEpisode
      )
      expect(container.textContent).toContain(message)
      expect(button("Save episode").disabled).toBe(true)
      expect(button("Discard and retake").disabled).toBe(false)
    }

    const path = "/home/user/.almond/tracker/diagnostics/fault-test"
    render(
      {
        ...recording,
        trackingQuality: {
          ...fault,
          diagnostics: { enabled: true, state: "saved", latest_path: path },
        },
      },
      onEpisode
    )
    expect(container.textContent).toContain("Tracking diagnostics saved")
    expect(container.textContent).toContain(
      "Stop this collection session and share the capture path"
    )
    expect(container.textContent).toContain(path)
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("retake required")
    expect(button("Save episode").disabled).toBe(true)
    expect(onEpisode).not.toHaveBeenCalled()
  })

  it("reports a diagnostic capture error without failing tracking or blocking a valid save", () => {
    const onEpisode = vi.fn()
    render(
      {
        ...recording,
        trackingQuality: {
          ...clean,
          diagnostics: { enabled: false, state: "error", error: "Capture disk is full" },
        },
      },
      onEpisode
    )
    expect(container.textContent).toContain("Live tracking checks passed")
    expect(container.textContent).toContain("Diagnostic capture unavailable")
    expect(container.textContent).toContain("Capture disk is full")
    expect(container.querySelector('[role="alert"]')).toBeNull()
    expect(button("Save episode").disabled).toBe(false)
    act(() => button("Save episode").click())
    act(() => button("Confirm: Save episode?").click())
    expect(onEpisode).toHaveBeenCalledWith("s")
  })
})
