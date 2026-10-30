import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// The stage's 3D viewer loads three.js and fetches URDFs: not under test.
vi.mock("./pose-viewer", () => ({ default: () => null }))

import { MountRequired } from "./mount-required"

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

const card = (name: string) =>
  [...container.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) =>
    b.textContent?.includes(name)
  )!
const saveButton = () =>
  [...container.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Save")!

describe("MountRequired", () => {
  it("stages a pick and saves it only on Save", async () => {
    const onSave = vi.fn(async () => {})
    await act(async () => root.render(<MountRequired locked={false} onSave={onSave} />))
    expect(saveButton().disabled).toBe(true)

    await act(async () => card("Axol on Owl Mount").click())
    expect(onSave).not.toHaveBeenCalled()
    expect(card("Axol on Owl Mount").getAttribute("aria-checked")).toBe("true")
    expect(container.textContent).toContain("Axol on Owl Mount selected")
    // The stage shows the pick, not the legacy fallback, once the pointer leaves.
    await act(async () => {
      card("Axol on Owl Mount").dispatchEvent(new MouseEvent("mouseout", { bubbles: true }))
    })
    expect(container.querySelector(".text-lg")?.textContent).toBe("Axol on Owl Mount")

    await act(async () => saveButton().click())
    expect(onSave).toHaveBeenCalledWith("owl")
  })

  it("is locked until an Axol is connected", async () => {
    const onSave = vi.fn(async () => {})
    await act(async () => root.render(<MountRequired locked onSave={onSave} />))
    await act(async () => card("Axol on Owl Mount").click())
    expect(card("Axol on Owl Mount").getAttribute("aria-checked")).toBe("false")
    expect(card("Axol on Owl Mount").querySelector('[aria-label="Locked"]')).not.toBeNull()
    expect(container.textContent).toContain("Connect Axol to choose a mount.")
    expect(saveButton().disabled).toBe(true)
    expect(onSave).not.toHaveBeenCalled()
  })
})
