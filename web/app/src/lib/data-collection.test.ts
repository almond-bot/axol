import { describe, expect, it } from "vitest"

import {
  formatMoney,
  parseHuggingFaceDataset,
  parseVideoUrl,
  requestSource,
  validateDataCollectionRequest,
} from "./data-collection"

const valid = {
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

  it.each([
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
  it("keeps the operation page a card linked from, else names the page itself", () => {
    expect(requestSource("teleop")).toBe("teleop")
    expect(requestSource("collect-data")).toBe("collect-data")
    expect(requestSource("overview")).toBe("overview")
    expect(requestSource(null)).toBe("request-data")
    expect(requestSource("run-policy")).toBe("request-data")
  })

  it("is a page the checkout accepts", () => {
    const result = validateDataCollectionRequest({ ...valid, operation: requestSource(null) })
    expect(result.ok).toBe(true)
  })
})

describe("formatMoney", () => {
  it("drops cents on whole amounts", () => {
    expect(formatMoney(5000, "usd")).toBe("$50")
    expect(formatMoney(5050, "usd")).toBe("$50.50")
    expect(formatMoney(5_000_000, "usd")).toBe("$50,000")
  })
})
