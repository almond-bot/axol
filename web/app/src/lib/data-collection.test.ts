import { describe, expect, it } from "vitest"

import {
  formatMoney,
  parseHuggingFaceDataset,
  priceKey,
  parseVideoUrl,
  requestSource,
  validateDataCollectionRequest,
} from "./data-collection"

const valid = {
  service: "task",
  method: null,
  hours: 10,
  task: "Fold the towel in half twice.",
  example: { kind: "video", value: "https://youtu.be/abc" },
  operation: "teleop",
  returnUrl: "https://192.168.1.50:8001/control/operations",
}

describe("parseHuggingFaceDataset", () => {
  it("accepts an id or a dataset link", () => {
    expect(parseHuggingFaceDataset("almond-bot/towels")).toBe("almond-bot/towels")
    expect(
      parseHuggingFaceDataset(" https://huggingface.co/datasets/almond-bot/towels/tree/main")
    ).toBe("almond-bot/towels")
  })

  it("rejects anything else", () => {
    expect(parseHuggingFaceDataset("towels")).toBeNull()
    expect(parseHuggingFaceDataset("https://huggingface.co/almond-bot/model")).toBeNull()
    expect(parseHuggingFaceDataset("a/b/c")).toBeNull()
  })
})

describe("parseVideoUrl", () => {
  it("accepts only http(s) links", () => {
    expect(parseVideoUrl("https://drive.google.com/file/d/1")).toBe(
      "https://drive.google.com/file/d/1"
    )
    expect(parseVideoUrl("javascript:alert(1)")).toBeNull()
    expect(parseVideoUrl("not a link")).toBeNull()
  })
})

describe("validateDataCollectionRequest", () => {
  it("accepts a complete request and trims the task", () => {
    const result = validateDataCollectionRequest({ ...valid, task: "  Fold the towel.  " })
    expect(result).toEqual({ ok: true, value: { ...valid, task: "Fold the towel." } })
  })

  it("takes a request without an example", () => {
    for (const example of [undefined, {}, { kind: "video", value: "  " }]) {
      const result = validateDataCollectionRequest({ ...valid, example })
      expect(result).toEqual({ ok: true, value: { ...valid, example: null } })
    }
  })

  it("normalizes a dataset link to its id", () => {
    const result = validateDataCollectionRequest({
      ...valid,
      example: { kind: "dataset", value: "https://huggingface.co/datasets/org/set" },
    })
    expect(result.ok && result.value.example).toEqual({ kind: "dataset", value: "org/set" })
  })

  it("accepts both services", () => {
    const task = validateDataCollectionRequest({ ...valid, method: "your_axol" })
    expect(task.ok && task.value).toMatchObject({ service: "task", method: null })
    for (const method of ["almond_axol", "your_axol"]) {
      const result = validateDataCollectionRequest({ ...valid, service: "dataset", method })
      expect(result.ok && result.value).toMatchObject({ service: "dataset", method })
    }
  })

  it.each([
    [{ service: undefined }, "one-off task or a training dataset"],
    [{ service: "policy" }, "one-off task or a training dataset"],
    [{ service: "dataset", method: undefined }, "how Almond collects"],
    [{ service: "dataset", method: "drone" }, "how Almond collects"],
    [{ service: "dataset", method: "mantis_umi" }, "how Almond collects"],
    [{ hours: 0 }, "whole number of hours"],
    [{ hours: 2.5 }, "whole number of hours"],
    [{ hours: "lots" }, "whole number of hours"],
    [{ task: "   " }, "Describe the task"],
    [{ task: "x".repeat(501) }, "500 characters"],
    [{ example: { kind: "video", value: "ftp://x" } }, "link to a video"],
    [{ example: { kind: "dataset", value: "towels" } }, "owner/name"],
    [{ example: { kind: "audio", value: "x" } }, "video link or a Hugging Face dataset"],
    [{ operation: "run-policy" }, "Unknown page"],
    [{ returnUrl: "javascript:alert(1)" }, "Unknown return page"],
  ])("rejects %j", (patch, message) => {
    const result = validateDataCollectionRequest({ ...valid, ...patch })
    expect(result.ok).toBe(false)
    expect(!result.ok && result.error).toContain(message)
  })
})

describe("requestSource", () => {
  it("keeps the operation page a card linked from, else names the request page", () => {
    expect(requestSource("teleop", "task")).toBe("teleop")
    expect(requestSource("collect-data", "dataset")).toBe("collect-data")
    expect(requestSource("overview", "task")).toBe("overview")
    expect(requestSource(null, "dataset")).toBe("request-data")
    expect(requestSource(null, "task")).toBe("request-teleoperator")
    expect(requestSource("run-policy", "dataset")).toBe("request-data")
  })

  it("is a page the checkout accepts", () => {
    for (const service of ["task", "dataset"] as const) {
      const result = validateDataCollectionRequest({
        ...valid,
        operation: requestSource(null, service),
      })
      expect(result.ok).toBe(true)
    }
  })
})

describe("priceKey", () => {
  it("charges a task at the teleoperator price and a dataset by its method", () => {
    expect(priceKey("task", null)).toBe("teleoperator")
    expect(priceKey("dataset", "almond_axol")).toBe("almond_axol")
    expect(priceKey("dataset", "your_axol")).toBe("your_axol")
  })
})

describe("formatMoney", () => {
  it("drops cents on whole amounts", () => {
    expect(formatMoney(5000, "usd")).toBe("$50")
    expect(formatMoney(5050, "usd")).toBe("$50.50")
    expect(formatMoney(5_000_000, "usd")).toBe("$50,000")
  })
})
