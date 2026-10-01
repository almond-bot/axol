import { afterEach, describe, expect, it, vi } from "vitest"

import {
  MYACTUATOR_SECTIONS,
  PARAM_UI,
  formatParamValue,
  groupParams,
  paramUi,
  parseParamInput,
  unitLabel,
} from "./motor-params"
import { fetchMotorConfig, writeMotorConfig, type MotorConfigParam } from "./telemetry"

function param(name: string, value: number | null = 0): MotorConfigParam {
  return {
    name,
    index: 0,
    unit: "",
    access: "read_write",
    integer: false,
    supported: true,
    value,
    error: null,
  }
}

describe("motor parameter presentation", () => {
  it("lays every listed parameter out exactly once, with a label", () => {
    const listed = MYACTUATOR_SECTIONS.flatMap((s) => s.params)
    expect(new Set(listed).size).toBe(listed.length)
    for (const name of listed) expect(PARAM_UI[name]?.label).toBeTruthy()
  })

  it("groups by panel and keeps unknown parameters under Other", () => {
    const sections = groupParams([param("LOW_VOLTAGE"), param("MAX_SPEED"), param("NEW_THING")])
    expect(sections.map((s) => s.title)).toEqual([
      "Protect parameters",
      "Motor parameters",
      "Other",
    ])
    expect(sections[2].params[0].name).toBe("NEW_THING")
    expect(paramUi("NEW_THING").label).toBe("NEW_THING")
  })

  it("formats and parses values without losing whole numbers", () => {
    expect(formatParamValue(131072)).toBe("131072")
    expect(formatParamValue(24.8799991607666)).toBe("24.88")
    expect(formatParamValue(-2)).toBe("-2")
    expect(formatParamValue(null)).toBe("")
    expect(parseParamInput(" 52.5 ")).toBe(52.5)
    expect(parseParamInput("")).toBeNull()
    expect(parseParamInput("abc")).toBeNull()
    expect(parseParamInput("Infinity")).toBeNull()
  })

  it("shows readable units", () => {
    expect(unitLabel("C")).toBe("°C")
    expect(unitLabel("mOhm")).toBe("mΩ")
    expect(unitLabel("V")).toBe("V")
  })
})

describe("motor config API client", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("reads the table and posts one write with its confirmation flag", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = []
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        calls.push({ url: String(input), init })
        const body =
          init?.method === "POST" ? { name: "KT_OUT", requested: 2.5, value: 2.5 } : { params: [] }
        return new Response(JSON.stringify(body), { status: 200 })
      })
    )
    expect(await fetchMotorConfig("left", "SHOULDER_1")).toEqual({ params: [] })
    expect(await writeMotorConfig("left", "SHOULDER_1", "KT_OUT", 2.5, true)).toEqual({
      name: "KT_OUT",
      requested: 2.5,
      value: 2.5,
    })
    expect(calls[0].url).toMatch(/\/api\/robot\/motors\/left\/SHOULDER_1\/config$/)
    expect(calls[1].init?.method).toBe("POST")
    expect(JSON.parse(String(calls[1].init?.body))).toEqual({
      param: "KT_OUT",
      value: 2.5,
      allowProtected: true,
    })
  })

  it("surfaces the server's refusal", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () => new Response(JSON.stringify({ error: "KT_OUT is protected" }), { status: 400 })
      )
    )
    await expect(writeMotorConfig("left", "SHOULDER_1", "KT_OUT", 2.5)).rejects.toThrow(
      "KT_OUT is protected"
    )
  })
})
