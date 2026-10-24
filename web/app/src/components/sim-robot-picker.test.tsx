import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SimRobotPicker, type SimRobot } from "./sim-robot-picker"

let container: HTMLDivElement
let root: Root

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

const radio = (name: string) =>
  [...container.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) =>
    b.textContent?.includes(name)
  )!

describe("SimRobotPicker", () => {
  it("offers every version unlocked, Jelly included", async () => {
    const onChange = vi.fn()
    const value: SimRobot = { model: "classic", gripper: "parallel" }
    await act(async () => root.render(<SimRobotPicker value={value} onChange={onChange} />))
    const radios = container.querySelectorAll('[role="radio"]')
    expect(radios).toHaveLength(5)
    expect(radio("Axol Legacy").getAttribute("aria-checked")).toBe("true")
    await act(async () => radio("Axol on Jelly").click())
    expect(onChange).toHaveBeenCalledWith({ model: "mobile", gripper: "parallel" })
    // The parcel gripper exists on Jelly only.
    expect(container.querySelector('[aria-label="Parcel gripper"]')).toBeNull()
  })

  it("picks the parcel gripper on Jelly", async () => {
    const onChange = vi.fn()
    const value: SimRobot = { model: "mobile", gripper: "parallel" }
    await act(async () => root.render(<SimRobotPicker value={value} onChange={onChange} />))
    await act(async () =>
      container.querySelector<HTMLButtonElement>('[aria-label="Parcel gripper"]')!.click()
    )
    expect(onChange).toHaveBeenCalledWith({ model: "mobile", gripper: "parcel" })
  })

  it("changes nothing while a run is live", async () => {
    const onChange = vi.fn()
    const value: SimRobot = { model: "owl", gripper: "parallel" }
    await act(async () =>
      root.render(<SimRobotPicker value={value} onChange={onChange} disabled />)
    )
    await act(async () => radio("Axol on Ox Cart").click())
    expect(onChange).not.toHaveBeenCalled()
  })
})
