import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SessionInfo, TrackerBindingsSnapshot } from "@/lib/supervisor"
import { TrackerBindingPanel } from "./tracker-binding-panel"

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  start: vi.fn(),
  input: vi.fn(),
  stop: vi.fn(),
  saveQuality: vi.fn(),
  logs: { lines: [] as string[], status: null as SessionInfo | null },
  toast: { success: vi.fn(), error: vi.fn() },
}))
vi.mock("@/lib/supervisor", async (original) => ({
  ...(await original<typeof import("@/lib/supervisor")>()),
  fetchTrackerBindings: mocks.fetch,
  useSessionLogs: () => mocks.logs,
  sendSessionInput: mocks.input,
  stopSession: mocks.stop,
  saveLighthouseQualityMode: mocks.saveQuality,
}))
vi.mock("@/lib/telemetry", () => ({ startDiagnosticsRun: mocks.start }))
vi.mock("@/components/ui/toast", () => ({ useToast: () => mocks.toast }))
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement
let snapshot: TrackerBindingsSnapshot

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
  mocks.logs = { lines: [], status: null }
  const binding = { complete: true, left: "T20", right: "T21" }
  snapshot = {
    bindings: { survive: binding, ultimate: binding },
    sources: {
      lighthouse: {
        binding,
        installed: true,
        pinnedBuild: true,
        udevReady: true,
        pinnedRef: "abc123",
        issues: [],
        expectedBaseStations: 3,
        baseStations: {
          channels: { "1": ["one"], "2": ["two"], "3": ["three"] },
          clashingChannels: [],
          baseStationCount: 3,
          expectedBaseStations: 3,
          checkedAt: Date.now() / 1000,
          trackers: ["T20", "T21"],
          problems: [],
        },
        validation: { status: "missing", valid: false, problems: [] },
      },
      // Other tracker families are not rendered in these Lighthouse tests.
    } as unknown as TrackerBindingsSnapshot["sources"],
  }
  mocks.fetch.mockImplementation(async () => snapshot)
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.useRealTimers()
})

async function render(blockedReason: string | null = null, hostSession: SessionInfo | null = null) {
  await act(async () =>
    root.render(
      <TrackerBindingPanel
        source="lighthouse"
        sourceSaved
        calibrationContextSaved
        blockedReason={blockedReason}
        hostSession={hostSession}
        onHostSessionChange={() => {}}
      />
    )
  )
}

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find(
    (b) => b.textContent?.trim() === label
  )
  if (!found) throw new Error(`no ${label} button`)
  return found
}

function session(command = "tracker.lighthouse.validate"): SessionInfo {
  return {
    id: "validation-1",
    command,
    args: {},
    status: "running",
    exitCode: null,
    error: null,
    startedAt: Date.now() / 1000,
    pid: 123,
  }
}

describe("Lighthouse validation", () => {
  it("allows recalibration with legacy bindings while keeping validation blocked", async () => {
    const legacy = { complete: false, left: "WM0", right: "WM1" }
    snapshot.bindings.survive = legacy
    snapshot.sources!.lighthouse.binding = legacy
    mocks.start.mockResolvedValue({ session: session("tracker.lighthouse.calibrate") })
    await render()
    expect(button("Validate tracking").disabled).toBe(true)
    expect(button("Recalibrate Lighthouse").disabled).toBe(false)
    await act(async () => button("Recalibrate Lighthouse").click())
    expect(mocks.start).toHaveBeenCalledWith("tracker.lighthouse.calibrate", {
      expected_stations: 3,
      web_prompts: true,
    })
  })

  it("shows bootstrap calibration as requiring identification, never as a recording pass", async () => {
    snapshot.sources!.lighthouse.validation = {
      status: "needs_identification",
      valid: false,
      expectedBaseStations: 3,
      problems: ["Identify sides, then Validate tracking before recording."],
    }
    await render()
    expect(container.textContent).toContain(
      "Calibration saved — identify sides, then validate tracking"
    )
    expect(container.textContent).not.toContain("Tracking check passed")
  })

  it("uses the selected station count for the survey and highlights the actual failure", async () => {
    snapshot.sources!.lighthouse.expectedBaseStations = 2
    snapshot.sources!.lighthouse.baseStations = {
      channels: { "1": [], "2": [] },
      clashingChannels: [],
      baseStationCount: 2,
      expectedBaseStations: 2,
      checkedAt: Date.now() / 1000,
      trackers: ["LHR-ONE"],
      problems: [],
    }
    const failed = {
      ...session("tracker.lighthouse.check"),
      status: "exited" as const,
      exitCode: 1,
    }
    mocks.start.mockResolvedValue({ session: failed })
    await render()
    const select = container.querySelector<HTMLSelectElement>("select")!
    act(() => {
      select.value = "3"
      select.dispatchEvent(new Event("change", { bubbles: true }))
    })
    expect(container.textContent).toContain("Only 2 of 3 base stations seen")
    await act(async () => button("Check base stations again").click())
    expect(mocks.start).toHaveBeenCalledWith("tracker.lighthouse.check", { expected_stations: 3 })
    mocks.logs = {
      lines: ["FAIL Trackers 1 of 2 reporting", "Saved to /tmp/survey.json"],
      status: failed,
    }
    await render(null, failed)
    const error = [...container.querySelectorAll("p")].find((p) =>
      p.className.includes("text-red-300/80")
    )
    expect(error?.textContent).toBe("FAIL Trackers 1 of 2 reporting")
    expect(container.textContent).not.toContain("pair the missing one")
    expect(container.textContent).not.toContain("display on its back")
  })

  it("keeps validation and recalibration visible after device setup completes", async () => {
    await render()
    expect(container.textContent).toContain("Tracking quality not checked")
    expect(container.textContent).toContain("reference fixture is needed")
    expect(button("Validate tracking").disabled).toBe(false)
    expect(button("Recalibrate Lighthouse").disabled).toBe(false)
    expect(container.querySelector<HTMLSelectElement>("select")?.value).toBe("3")
  })

  it("does not show expired or failed reports as healthy", async () => {
    snapshot.sources!.lighthouse.validation = {
      status: "passed",
      valid: true,
      expiresAt: Date.now() / 1000 - 1,
      problems: [],
    }
    await render()
    expect(container.textContent).toContain("Validation expired or setup changed")
    expect(container.textContent).not.toContain("Tracking check passed")
    snapshot = {
      ...snapshot,
      sources: {
        ...snapshot.sources!,
        lighthouse: {
          ...snapshot.sources!.lighthouse,
          validation: {
            status: "failed",
            valid: false,
            problems: ["Right tracker noise too high; restore visibility"],
          },
        },
      },
    }
    await act(async () => {
      vi.advanceTimersByTime(3000)
    })
    expect(container.textContent).toContain("Tracking check failed")
    expect(container.textContent).toContain("Right tracker noise too high; restore visibility")
  })

  it("sends bounded numeric station count and waits for the guided stillness prompt", async () => {
    mocks.start.mockResolvedValue({ session: session() })
    await render()
    const select = container.querySelector<HTMLSelectElement>("select")!
    act(() => {
      select.value = "4"
      select.dispatchEvent(new Event("change", { bubbles: true }))
    })
    await act(async () => button("Validate tracking").click())
    expect(mocks.start).toHaveBeenCalledWith("tracker.lighthouse.validate", {
      expected_stations: 4,
      web_prompts: true,
    })
    expect(button("Recalibrate Lighthouse").disabled).toBe(true)
    expect(mocks.input).not.toHaveBeenCalled()
    mocks.logs = { lines: ["[prompt] Put both rigs still on the table."], status: session() }
    await render()
    await act(async () => button("Continue").click())
    expect(mocks.input).toHaveBeenCalledWith("validation-1")
    expect(container.textContent).toContain("Running tracking check")
    const sweep = "Move each rig slowly for 12 seconds, including wrist turns."
    mocks.logs = {
      lines: ["[prompt] Put both rigs still on the table.", `[prompt] ${sweep}`],
      status: session(),
    }
    await render()
    await act(async () => button("Continue").click())
    expect(mocks.input).toHaveBeenCalledTimes(2)
    expect(container.textContent).toContain(sweep)
    expect(container.textContent).not.toContain("Keep both rigs still.")
  })

  it("saves the tracking policy and explains what it changes", async () => {
    snapshot.sources!.lighthouse.qualityMode = "standard"
    mocks.saveQuality.mockResolvedValue({ mode: "strict" })
    await render()
    const policy = container.querySelector<HTMLSelectElement>(
      'select[aria-label="Tracking policy"]'
    )!
    expect(policy.value).toBe("standard")
    expect(container.textContent).toContain("validation is optional")
    await act(async () => {
      policy.value = "strict"
      policy.dispatchEvent(new Event("change", { bubbles: true }))
    })
    expect(mocks.saveQuality).toHaveBeenCalledWith("strict")
    expect(policy.value).toBe("strict")
    expect(container.textContent).toContain("requires a passed validation before collecting")
  })

  it("hides the tracking policy on hosts that do not report it", async () => {
    await render()
    expect(container.querySelector('select[aria-label="Tracking policy"]')).toBeNull()
  })

  it("disables setup commands while an operation owns the host", async () => {
    await render("Collect data is running")
    expect(button("Validate tracking").disabled).toBe(true)
    expect(button("Recalibrate Lighthouse").disabled).toBe(true)
    act(() => button("Recalibrate Lighthouse").click())
    expect(mocks.start).not.toHaveBeenCalled()
  })

  it("requires a new check for a changed station count and launches recalibration explicitly", async () => {
    snapshot.sources!.lighthouse.validation = {
      status: "passed",
      valid: true,
      expectedBaseStations: 3,
      problems: [],
    }
    mocks.start.mockResolvedValue({ session: session("tracker.lighthouse.calibrate") })
    await render()
    expect(container.textContent).toContain("Tracking check passed")
    const select = container.querySelector<HTMLSelectElement>("select")!
    act(() => {
      select.value = "2"
      select.dispatchEvent(new Event("change", { bubbles: true }))
    })
    expect(container.textContent).toContain("Validation expired or setup changed")
    expect(mocks.start).not.toHaveBeenCalled()
    await act(async () => button("Recalibrate Lighthouse").click())
    expect(mocks.start).toHaveBeenCalledWith("tracker.lighthouse.calibrate", {
      expected_stations: 2,
      web_prompts: true,
    })
  })

  it("keeps commands visible but disabled on a host without validation support", async () => {
    delete snapshot.sources!.lighthouse.validation
    await render()
    expect(button("Validate tracking").disabled).toBe(true)
    expect(button("Recalibrate Lighthouse").disabled).toBe(true)
    expect(container.textContent).toContain("Update Axol on the host")
  })

  it("expires an old pass even while the next host response is pending", async () => {
    snapshot.sources!.lighthouse.validation = {
      status: "passed",
      valid: true,
      expiresAt: Date.now() / 1000 + 1,
      problems: [],
    }
    await render()
    expect(container.textContent).toContain("Tracking check passed")
    mocks.fetch.mockImplementation(() => new Promise(() => {}))
    await act(async () => {
      vi.advanceTimersByTime(3000)
    })
    expect(container.textContent).toContain("Validation expired or setup changed")
    expect(container.textContent).not.toContain("Tracking check passed")
  })

  it("shows a failed command and does not accept an old green report when the host read fails", async () => {
    snapshot.sources!.lighthouse.validation = { status: "passed", valid: true, problems: [] }
    await render()
    mocks.fetch.mockRejectedValue(new Error("offline"))
    const failed = {
      ...session(),
      status: "error" as const,
      exitCode: 1,
      error: "Station 3 is missing",
    }
    mocks.logs = { lines: [], status: failed }
    await render(null, failed)
    expect(container.textContent).toContain("Station 3 is missing")
    expect(container.textContent).toContain("Validation status unavailable")
    expect(container.textContent).not.toContain("Tracking check passed")
    expect(button("Validate tracking").disabled).toBe(true)
  })
})
