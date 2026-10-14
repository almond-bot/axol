import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import type { SchemaField, SchemaGroup } from "@/lib/supervisor"
import { FlatSchemaForm } from "./config-form"

const num = (key: string, label: string, def: number): SchemaField => ({
  kind: "field",
  key,
  label,
  type: "number",
  default: def,
  required: false,
})

/** A joint group shaped like JointConfig: numbers, a nested group, a text knob. */
const joint = (name: string): SchemaGroup => ({
  kind: "group",
  key: `left.${name}`,
  label: name,
  children: [
    num(`left.${name}.kp`, "kp", 100),
    num(`left.${name}.kd`, "kd", 2),
    {
      kind: "group",
      key: `left.${name}.friction`,
      label: "friction",
      children: [num(`left.${name}.friction.fc`, "fc", 0.1)],
    },
    {
      kind: "field",
      key: `left.${name}.wire_mode`,
      label: "wire mode",
      type: "text",
      default: "mit",
      required: false,
    },
  ],
})

let root: Root
let container: HTMLDivElement

describe("FlatSchemaForm", () => {
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

  it("keeps per-joint groups with a text field in one table", () => {
    act(() =>
      root.render(
        <FlatSchemaForm
          nodes={["shoulder 1", "shoulder 2", "elbow"].map(joint)}
          overrides={{}}
          disabled={false}
          onChange={() => {}}
          onReset={() => {}}
        />
      )
    )
    const tables = container.querySelectorAll("table")
    expect(tables).toHaveLength(1)
    expect(tables[0].querySelectorAll("tbody tr")).toHaveLength(3)
    const headers = [...tables[0].querySelectorAll("th")].map((th) => th.textContent)
    expect(headers).toEqual(["", "Kp", "Kd", "Fc", "Wire mode"])
    const wire = tables[0].querySelector<HTMLInputElement>('input[title="left.elbow.wire_mode"]')
    expect(wire?.value).toBe("mit")
  })
})
