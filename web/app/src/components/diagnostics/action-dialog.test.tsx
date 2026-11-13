import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CommandSpec, SchemaField } from "@/lib/supervisor"
import { ActionDialog, type ActionMode, type BusOption } from "./diagnostic-actions"
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

function field(over: Partial<SchemaField> & Pick<SchemaField, "key" | "type">): SchemaField {
  return { kind: "field", label: over.key, default: null, required: false, help: null, ...over }
}

// motor.set-zero-pos as the host advertises it: --l/--r collapse into an
// optional Arm select because --channel can stand in for them.
const ARM_FIELD = field({ key: "arm", type: "select", default: "", options: ["left", "right"] })
const SPEC: CommandSpec = {
  id: "motor.set-zero-pos",
  cli: "motor.set-zero-pos",
  label: "Set zero position",
  description: "",
  category: "Calibrate",
  simCapable: false,
  requiresHardware: true,
  available: true,
  error: null,
  schema: [
    ARM_FIELD,
    field({ key: "channel", type: "text", default: "" }),
    field({ key: "id", type: "text", default: "" }),
    field({ key: "guided", type: "boolean", default: false }),
    field({ key: "web_prompts", type: "boolean", default: false }),
  ],
  required: [],
}

const MODES: ActionMode[] = [
  { key: "single", label: "Specific motor", hideKeys: ["guided"] },
  { key: "guided", label: "Guided", armOnly: true, presetArgs: { guided: true }, hideKeys: ["id"] },
]

const BUS_OPTIONS: BusOption[] = [
  { value: "left", label: "Left arm (can_alm_axol_l)" },
  { value: "right", label: "Right arm (can_alm_axol_r)" },
  { value: "iface:can0", label: "can0" },
]

describe("ActionDialog", () => {
  let container: HTMLDivElement
  let root: Root
  const onLaunch = vi.fn()

  beforeEach(() => {
    container = document.createElement("div")
    document.body.appendChild(container)
    root = createRoot(container)
    onLaunch.mockReset()
  })

  afterEach(() => {
    act(() => root.unmount())
    container.remove()
  })

  function render(props: { modes?: ActionMode[]; busOptions?: BusOption[]; spec?: CommandSpec }) {
    act(() => {
      root.render(
        <ActionDialog
          spec={props.spec ?? SPEC}
          modes={props.modes}
          busOptions={props.busOptions}
          running={false}
          blocked={false}
          busy={false}
          disabled={false}
          onLaunch={onLaunch}
          onStop={() => {}}
          onClose={() => {}}
        />
      )
    })
  }

  function button(text: string): HTMLButtonElement {
    const found = [...container.querySelectorAll("button")].find((b) => b.textContent === text)
    if (!found) throw new Error(`no button ${text}`)
    return found as HTMLButtonElement
  }

  function choose(select: HTMLSelectElement, value: string) {
    act(() => {
      select.value = value
      select.dispatchEvent(new Event("change", { bubbles: true }))
    })
  }

  it("guided mode offers only the arms and sends the chosen arm", () => {
    render({ modes: MODES, busOptions: BUS_OPTIONS })
    act(() => button("Guided").click())

    const select = container.querySelector("#cfg-bus") as HTMLSelectElement
    expect(container.textContent).toContain("Arm")
    expect([...select.options].map((o) => o.value)).toEqual(["", "left", "right"])
    expect(select.value).toBe("")

    choose(select, "left")
    act(() => button(" Run").click())
    expect(onLaunch).toHaveBeenCalledTimes(1)
    expect(onLaunch.mock.calls[0][0]).toMatchObject({ arm: "left", guided: true })
    expect(onLaunch.mock.calls[0][0]).not.toHaveProperty("channel")
  })

  it("guided mode refuses to run without an arm", () => {
    render({ modes: MODES, busOptions: BUS_OPTIONS })
    act(() => button("Guided").click())
    act(() => button(" Run").click())
    expect(onLaunch).not.toHaveBeenCalled()
    expect(container.textContent).toContain("Required: Arm")
  })

  it("a test-stand interface picked for one motor does not carry into guided mode", () => {
    render({ modes: MODES, busOptions: BUS_OPTIONS })
    choose(container.querySelector("#cfg-bus") as HTMLSelectElement, "iface:can0")
    act(() => button("Guided").click())
    expect((container.querySelector("#cfg-bus") as HTMLSelectElement).value).toBe("")
  })

  it("an optional select starts on a blank entry, so picking the first option counts", () => {
    render({ spec: { ...SPEC, schema: [ARM_FIELD] } })
    const select = container.querySelector("select") as HTMLSelectElement
    expect(select.value).toBe("")
    expect(select.options[0].value).toBe("")

    choose(select, "left")
    act(() => button(" Run").click())
    expect(onLaunch.mock.calls[0][0]).toMatchObject({ arm: "left" })
  })
})
