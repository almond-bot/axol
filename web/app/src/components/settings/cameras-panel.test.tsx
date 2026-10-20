import { act, useState } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CameraDevice, CameraSpec } from "@/lib/supervisor"
import { CamerasPanel } from "./cameras-panel"
import { SettingsSection } from "./settings-section"

vi.mock("@/components/ui/toast", () => ({
  useToast: () => ({ success: vi.fn(), error: vi.fn() }),
}))
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const devices: CameraDevice[] = [
  { serial: 101, model: "ZED X Mini", kind: "stereo" },
  { serial: 202, model: "ZED X One", kind: "mono" },
  { serial: 203, model: "ZED X One", kind: "mono" },
]
const cameras: CameraSpec = {
  serials: { overhead: "101", left_arm: "102", right_arm: "103" },
  mantis_serials: { left_arm: "202", right_arm: "203" },
}

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

function renderPanel(initial = cameras) {
  const onChange = vi.fn()
  function Panel() {
    const [spec, setSpec] = useState(initial)
    return (
      <CamerasPanel
        spec={spec}
        mantisMode
        onChange={(next) => {
          onChange(next)
          setSpec(next)
        }}
        devices={devices}
        detecting={false}
        onRefresh={() => {}}
      />
    )
  }
  act(() => root.render(<Panel />))
  return onChange
}

function cameraSelect(label: string): HTMLSelectElement {
  return container.querySelector<HTMLSelectElement>(`select[aria-label="${label} camera"]`)!
}

function assign(label: string, serial: string) {
  const select = cameraSelect(label)
  act(() => {
    select.value = serial
    select.dispatchEvent(new Event("change", { bubbles: true }))
  })
}

function headEyes(branch: string): HTMLInputElement[] {
  const row = cameraSelect("Head").parentElement!.parentElement!
  const heading = [...row.querySelectorAll("span")].find((span) => span.textContent === branch)!
  return [...heading.parentElement!.querySelectorAll<HTMLInputElement>("input")]
}

describe("Mantis cameras", () => {
  it("shows the legacy head alongside both wrists and records both stereo eyes by default", () => {
    renderPanel()
    expect(cameraSelect("Head").value).toBe("101")
    expect(cameraSelect("Left Mantis").value).toBe("202")
    expect(cameraSelect("Right Mantis").value).toBe("203")
    expect(container.textContent).toContain("→ Head")
    expect(headEyes("Record").map((eye) => eye.checked)).toEqual([true, true])
    expect(headEyes("Stream").map((eye) => eye.checked)).toEqual([true, true])
  })

  it("assigns and unassigns the Mantis head independently of Axol", () => {
    const onChange = renderPanel({
      ...cameras,
      mantis_serials: { ...cameras.mantis_serials, overhead: "" },
    })
    expect(cameraSelect("Head").value).toBe("")
    assign("Head", "101")
    expect(onChange.mock.lastCall![0]).toMatchObject({
      serials: cameras.serials,
      mantis_serials: { overhead: "101", left_arm: "202", right_arm: "203" },
    })
    assign("Head", "")
    expect(cameraSelect("Head").value).toBe("")
    expect(onChange.mock.lastCall![0].serials.overhead).toBe("101")
  })

  it("moves a camera out of a wrist slot when it is assigned to the head", () => {
    const onChange = renderPanel()
    assign("Head", "202")
    expect(cameraSelect("Left Mantis").value).toBe("")
    expect(onChange.mock.lastCall![0].mantis_serials).toEqual({
      overhead: "202",
      left_arm: "",
      right_arm: "203",
    })
  })

  it("keeps the legacy head camera assigned only to a Mantis wrist until explicitly moved", () => {
    renderPanel({
      ...cameras,
      serials: { ...cameras.serials, overhead: " 101 " },
      mantis_serials: { left_arm: "101", right_arm: "203" },
    })
    expect(cameraSelect("Head").value).toBe("")
    expect(cameraSelect("Left Mantis").value).toBe("101")
    expect(container.textContent).not.toContain("→ Head")
    assign("Head", "101")
    expect(cameraSelect("Head").value).toBe("101")
    expect(cameraSelect("Left Mantis").value).toBe("")
  })

  it("selects head recording eyes separately from its headset stream", () => {
    const onChange = renderPanel()
    act(() => headEyes("Record")[1].click())
    expect(onChange.mock.lastCall![0].record).toEqual({ overhead: "left" })
    expect(headEyes("Record").map((eye) => eye.checked)).toEqual([true, false])
    expect(headEyes("Stream").map((eye) => eye.checked)).toEqual([true, true])
    act(() => headEyes("Record")[0].click())
    expect(onChange.mock.lastCall![0].record).toEqual({ overhead: false })
  })

  it("saves the new head with stereo defaults based on the Mantis assignment", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    act(() =>
      root.render(
        <SettingsSection
          open
          onOpenChange={() => {}}
          tab="mantis-cameras"
          onTabChange={() => {}}
          snapshot={null}
          supportError="Settings API unavailable"
          cameras={{
            ...cameras,
            serials: { ...cameras.serials, overhead: "202" },
            mantis_serials: { ...cameras.mantis_serials, overhead: "" },
          }}
          onSave={onSave}
          devices={devices}
          detecting={false}
          onRefresh={() => {}}
          usb={null}
          usbBusy={false}
          onUsbConnect={() => {}}
          actionBlocker={null}
          activeCommandSession={null}
          onCommandSessionChange={() => {}}
        />
      )
    )
    assign("Head", "101")
    const save = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save"
    )!
    expect(save.disabled).toBe(false)
    await act(async () => save.click())
    expect(onSave).toHaveBeenCalledWith({
      camerasSet: true,
      cameras: {
        serials: { overhead: "202", left_arm: "102", right_arm: "103" },
        mantis_serials: { overhead: "101", left_arm: "202", right_arm: "203" },
        stream_resolution: "SVGA",
        record_resolution: "SVGA",
        stream: { overhead: "both" },
        record: { overhead: "both" },
      },
    })
  })
})
